
/**
EagleSense interaction classifier

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright ?2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

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
            xgb(NULL),
            booster(NULL),
            booster_predict_func(NULL),
            dmatrix_class(NULL)
        {
        }

        InteractionClassifier::~InteractionClassifier()
        {
            Py_XDECREF(this->xgb);
            Py_XDECREF(this->booster);
            Py_XDECREF(this->dmatrix_class);
            Py_XDECREF(this->booster_predict_func);
            Py_Finalize();
        }

        bool InteractionClassifier::initialize(const std::string& model, const std::vector<std::string>& interactions)
        {
            this->interactions = interactions;

            topviewkinect::util::log_println("Initializing Python...");

            Py_Initialize();
            import_array();

            this->xgb = PyImport_ImportModule("xgboost"); // import xgboost as xgb
            if (!this->xgb)
            {
                topviewkinect::util::log_println("Failed to import xgboost!!");
            }

            PyObject* booster_class = PyObject_GetAttrString(this->xgb, "Booster"); // booster = xgb.Booster()
            this->booster = PyObject_CallFunctionObjArgs(booster_class, NULL);
            if (!this->booster)
            {
                topviewkinect::util::log_println("Failed to load Booster !!");
            }

            PyObject* booster_load_model_func = PyObject_GetAttrString(this->booster, "load_model"); // booster.load_model(fnmae)
            PyObject* model_fname = PyUnicode_FromString(topviewkinect::get_model_filepath(model).c_str());
            PyObject* booster_load_model_result = PyObject_CallFunctionObjArgs(booster_load_model_func, model_fname, NULL);
            if (!booster_load_model_result)
            {
                topviewkinect::util::log_println("Failed to load model !!");
            }

            this->booster_predict_func = PyObject_GetAttrString(this->booster, "predict"); // booster.predict
            if (!this->booster)
            {
                topviewkinect::util::log_println("Failed to load 'predict' function !!");
            }

            this->dmatrix_class = PyObject_GetAttrString(this->xgb, "DMatrix"); // xgb.DMatrix
            if (!this->booster)
            {
                topviewkinect::util::log_println("Failed to load 'DMatrix' structure !!");
            }

            Py_INCREF(this->xgb);
            Py_INCREF(this->booster);
            Py_INCREF(this->dmatrix_class);
            Py_INCREF(this->booster_predict_func);

            Py_DECREF(booster_class);
            Py_DECREF(model_fname);
            Py_DECREF(booster_load_model_func);
            Py_DECREF(booster_load_model_result);

            topviewkinect::util::log_println("Python initialized!!!");
            return true;
        }

        bool InteractionClassifier::recognize_interactions(std::vector<topviewkinect::skeleton::Skeleton>& skeletons) const
        {
            const int num_skeletons = static_cast<int>(std::count_if(skeletons.begin(), skeletons.end(), [](const topviewkinect::skeleton::Skeleton& skeleton) { return skeleton.is_activity_tracked(); }));
            if (num_skeletons == 0)
            {
                return false;
            }

            // Create C 2D array
            double* X_c_array = new double[num_skeletons * topviewkinect::vision::NUM_FEATURES];
            int nth_skeleton = 0;
            for (const topviewkinect::skeleton::Skeleton& skeleton : skeletons)
            {
                if (skeleton.is_activity_tracked())
                {
                    std::array<double, topviewkinect::vision::NUM_FEATURES> skeleton_features = skeleton.get_features();
                    std::copy(skeleton_features.begin(), skeleton_features.end(), X_c_array + nth_skeleton * topviewkinect::vision::NUM_FEATURES);
                    ++nth_skeleton;
                }
            }

            // Create NumPy 2D Array
            const int nrow = num_skeletons;
            const int ncol = topviewkinect::vision::NUM_FEATURES;
            const int dimension = 2;
            npy_intp shape[dimension] = { nrow, ncol };
            PyObject* X_p_array = PyArray_SimpleNewFromData(dimension, shape, NPY_FLOAT64, reinterpret_cast<void*>(X_c_array));
            if (!X_p_array)
            {
                topviewkinect::util::log_println("Failed to construct X Python Array !!");
            }

            PyArrayObject* X_np_array = reinterpret_cast<PyArrayObject*>(X_p_array);
            if (!X_np_array)
            {
                topviewkinect::util::log_println("Failed to construct X NumPy Array !!");
            }

            PyObject* X_dmatrix = PyObject_CallFunctionObjArgs(this->dmatrix_class, X_np_array, NULL); // X_DMatrix = xgb.DMatrix(X)
            if (!X_dmatrix)
            {
                topviewkinect::util::log_println("Failed to construct X DMatrix !!");
            }

            // Predict
            PyObject* y = PyObject_CallFunctionObjArgs(this->booster_predict_func, X_dmatrix, NULL); // y = booster.predict(X_DMatrix)
            if (!y)
            {
                topviewkinect::util::log_println("Failed to call predict function !!");
            }
            PyArrayObject* y_np_array = reinterpret_cast<PyArrayObject*>(y);
            float* y_c_array = reinterpret_cast<float*>(PyArray_DATA(y_np_array));

            int skeleton_idx = 0;
            for (topviewkinect::skeleton::Skeleton& skeleton : skeletons)
            {
                if (skeleton.is_activity_tracked())
                {
                    int activity_idx = static_cast<int>(y_c_array[skeleton_idx++]);
                    skeleton.set_activity_id(activity_idx);
                    skeleton.set_activity(this->interactions[activity_idx]);
                }
            }

            // Clean up
            Py_DECREF(X_np_array);
            Py_DECREF(X_dmatrix);
            Py_DECREF(y);

            delete[] X_c_array;

            return true;
        }
    }
}