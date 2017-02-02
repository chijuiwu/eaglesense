
/**
EagleSense interaction classifier

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

static void* init_numpy() {
    import_array();
    return NULL;
}

#include <sstream>
#include <iostream>
#include <chrono>

#include "topviewkinect/topviewkinect.h"
#include "topviewkinect/util.h"
#include "topviewkinect/vision/classifier.h"

namespace topviewkinect
{
    namespace vision
    {
        InteractionClassifier::InteractionClassifier() :
            p_classifier(NULL),
            p_predict_func(NULL)
        {
        }

        InteractionClassifier::~InteractionClassifier()
        {
            Py_XDECREF(this->p_classifier);
            if (this->p_classifier != NULL)
            {
                Py_DECREF(this->p_classifier);
            }
            if (this->p_predict_func != NULL)
            {
                Py_DECREF(this->p_predict_func);
            }
            Py_Finalize();
        }

        bool InteractionClassifier::initialize(const std::string& model, const std::vector<std::string>& interactions)
        {
            this->interactions = interactions;

            // Initialize Python
            topviewkinect::util::log_println("Initializing Python...");

            Py_Initialize();
            init_numpy();
            PyRun_SimpleString("import sys");

            std::ostringstream path_ss;
            path_ss << "sys.path.append(\"" << topviewkinect::EAGLESENSE_DIRECTORY << "\")";
            PyRun_SimpleString(path_ss.str().c_str());

            // Import classifier module
            topviewkinect::util::log_println("Initializing classifier...");

            PyObject* p_module = PyImport_ImportModule("classifier");
            if (p_module == NULL)
            {
                PyErr_Print();
                topviewkinect::util::log_println("Failed to load the classifier module.");
                return false;
            }

            PyObject* p_classifier_class = PyObject_GetAttrString(p_module, "TopviewInteractionClassifier");
            if (p_classifier_class == NULL)
            {
                PyErr_Print();
                topviewkinect::util::log_println("Failed to load the classifier class.");
                return false;
            }
            Py_DECREF(p_module);

            this->p_classifier = PyObject_CallObject(p_classifier_class, NULL);
            if (this->p_classifier == NULL)
            {
                PyErr_Print();
                topviewkinect::util::log_println("Failed to create a classifier object.");
                return false;
            }
            Py_DECREF(p_classifier_class);

            // Load machine learning model
            topviewkinect::util::log_println("Loading model...");

            std::string path_to_model = topviewkinect::get_model_filepath(model);
            PyObject* p_load_args = Py_BuildValue("(s)", path_to_model.c_str());
            PyObject* p_load = PyObject_GetAttrString(this->p_classifier, "load");
            PyObject* p_result = PyObject_CallObject(p_load, p_load_args);
            if (p_result == NULL)
            {
                PyErr_Print();
                topviewkinect::util::log_println("Failed to call the classifier `load` function.");
                return false;
            }
            Py_DECREF(p_load_args);
            Py_DECREF(p_load);
            Py_DECREF(p_result);

            // Load predict function
            topviewkinect::util::log_println("Loading 'predict' function...");

            this->p_predict_func = PyObject_GetAttrString(this->p_classifier, "predict");
            if (!PyCallable_Check(this->p_predict_func))
            {
                PyErr_Print();
                topviewkinect::util::log_println("Failed to load the classifier `predict` function.");
                return false;
            }

            return true;
        }

        bool InteractionClassifier::recognize_interactions(std::vector<topviewkinect::skeleton::Skeleton>& skeletons) const
        {
            if (this->p_predict_func == NULL)
            {
                topviewkinect::util::log_println("The classifier `predict` function was not initialized.");
                return false;
            }

            // Create C++ 2D array
            size_t num_skeletons = 0;
            for (const topviewkinect::skeleton::Skeleton& skeleton : skeletons)
            {
                if (skeleton.is_activity_tracked())
                {
                    ++num_skeletons;
                }
            }
            if (num_skeletons == 0)
            {
                return false;
            }

            double* c_array = new double[num_skeletons * topviewkinect::vision::NUM_TOTAL_FEATURES];
            int skeleton_counter = 0;
            for (const topviewkinect::skeleton::Skeleton& skeleton : skeletons)
            {
                if (skeleton.is_activity_tracked())
                {
                    std::array<double, topviewkinect::vision::NUM_TOTAL_FEATURES> skeleton_features = skeleton.get_features();
                    std::copy(skeleton_features.begin(), skeleton_features.end(), c_array + skeleton_counter * topviewkinect::vision::NUM_TOTAL_FEATURES);
                    ++skeleton_counter;
                }
            }

            // Create numpy arrays
            const int nd = 2;
            npy_intp dims[nd]{ static_cast<int>(num_skeletons), topviewkinect::vision::NUM_TOTAL_FEATURES };
            PyObject* p_array = PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT64, reinterpret_cast<void*>(c_array));
            if (p_array == NULL)
            {
                PyErr_Print();
                topviewkinect::util::log_println("Failed to construct numpy array.");
                return false;
            }

            // Call predict function via Python C++ API
            PyObject* p_return = PyObject_CallFunctionObjArgs(this->p_predict_func, p_array, NULL);
            if (p_return == NULL)
            {
                PyErr_Print();
                topviewkinect::util::log_println("Failed to call the classifier `predict` function.");
                return false;
            }
            Py_DECREF(p_array);

            // Convert results back to C++ array
            PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(p_return);
            int* c_out = reinterpret_cast<int*>(PyArray_DATA(np_ret));
            Py_DECREF(p_return);

            // Update skeleton activity
            skeleton_counter = 0;
            for (topviewkinect::skeleton::Skeleton& skeleton : skeletons)
            {
                if (skeleton.is_activity_tracked())
                {
                    int activity_id = c_out[skeleton_counter];
                    skeleton.set_activity_id(activity_id);
                    skeleton.set_activity(this->interactions[activity_id]);
                    ++skeleton_counter;
                }
            }

            // Clean up
            delete[] c_array;

            return true;
        }
    }
}