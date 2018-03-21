
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

			Py_XDECREF(this->fft_features_func);
			Py_XDECREF(this->gesture_recognition_phone_model);
			Py_XDECREF(this->gesture_recognition_phone_predict_func);

			Py_Finalize();
		}

		bool InteractionClassifier::initialize()
		{
			topviewkinect::util::log_println("Initializing Python...");

			Py_Initialize();
			import_array();

			topviewkinect::util::log_println("Python initialized !");
			return true;
		}

		bool InteractionClassifier::initialize_interaction_classification(const std::string& model, const std::vector<std::string>& interactions)
		{
			this->interactions = interactions;

			this->xgb = PyImport_ImportModule("xgboost"); // import xgboost
			if (!this->xgb)
			{
				topviewkinect::util::log_println("Failed to import xgboost !!");
			}
			std::cout << "import xgboost" << std::endl;

			PyObject* booster_class = PyObject_GetAttrString(this->xgb, "Booster"); // booster = xgb.Booster()
			this->booster = PyObject_CallFunctionObjArgs(booster_class, NULL);
			if (!this->booster)
			{
				topviewkinect::util::log_println("Failed to load xgboost.Booster !!");
			}
			std::cout << "booster = xgboost.Booster()" << std::endl;

			PyObject* booster_load_model_func = PyObject_GetAttrString(this->booster, "load_model"); // booster.load_model(fnmae)
			PyObject* model_fname = PyUnicode_FromString(topviewkinect::get_model_filepath(model).c_str());
			PyObject* booster_load_model_result = PyObject_CallFunctionObjArgs(booster_load_model_func, model_fname, NULL);
			if (!booster_load_model_result)
			{
				topviewkinect::util::log_println("Failed to load model !!");
			}
			std::cout << "booster.load_model('model')" << std::endl;

			this->booster_predict_func = PyObject_GetAttrString(this->booster, "predict"); // booster.predict
			if (!this->booster)
			{
				topviewkinect::util::log_println("Failed to load 'booster.predict' function !!");
			}

			this->dmatrix_class = PyObject_GetAttrString(this->xgb, "DMatrix"); // xgboost.DMatrix
			if (!this->booster)
			{
				topviewkinect::util::log_println("Failed to load 'xgboost.DMatrix' structure !!");
			}

			Py_INCREF(this->xgb);
			Py_INCREF(this->booster);
			Py_INCREF(this->dmatrix_class);
			Py_INCREF(this->booster_predict_func);

			Py_DECREF(booster_class);
			Py_DECREF(model_fname);
			Py_DECREF(booster_load_model_func);
			Py_DECREF(booster_load_model_result);

			topviewkinect::util::log_println("Interaction Classification module initialized !");
			return true;
		}

		bool InteractionClassifier::initialize_gesture_recognition_phone()
		{
			PyObject* sys_module = PyImport_ImportModule("sys"); // import sys
			if (!sys_module)
			{
				topviewkinect::util::log_println("Failed to import sys !!");
			}
			std::cout << "import sys" << std::endl;

			PyObject *sys_path = PyObject_GetAttrString(sys_module, "path");
			PyObject* eaglesense_dir = PyUnicode_FromString(topviewkinect::EAGLESENSE_DIRECTORY.c_str());
			int sys_path_append_res = PyList_Append(sys_path, eaglesense_dir);
			if (sys_path_append_res == -1)
			{
				topviewkinect::util::log_println("Failed to add eagelsense directory to system path !!");
			}
			std::cout << "sys.path.append('/path/to/eaglesense/')" << std::endl;

			PyObject* model_module = PyImport_ImportModule("model"); // import model
			if (!model_module)
			{
				topviewkinect::util::log_println("Failed to import EagleSense model module!!");
			}
			std::cout << "import model" << std::endl;

			this->fft_features_func = PyObject_GetAttrString(model_module, "fft_features"); // model.fft_features
			if (!this->fft_features_func)
			{
				topviewkinect::util::log_println("Failed to load 'model.fft_features' function !!");
			}

			// Clean up
			Py_INCREF(this->fft_features_func);

			Py_DECREF(sys_module);
			Py_DECREF(sys_path);
			Py_DECREF(eaglesense_dir);
			Py_DECREF(model_module);

			PyObject* pickle_module = PyImport_ImportModule("pickle"); // import pickle
			if (!pickle_module)
			{
				topviewkinect::util::log_println("Failed to import pickle !!");
			}
			std::cout << "import pickle" << std::endl;

			PyObject* io_module = PyImport_ImportModule("io"); // import io
			if (!io_module)
			{
				topviewkinect::util::log_println("Failed to import io !!");
			}
			std::cout << "import io" << std::endl;

			PyObject* model_fid = PyObject_CallMethod(io_module, "open", "ss", topviewkinect::get_model_filepath("v2/3-gesture-fft-svm.pkl").c_str(), "rb"); // fid = open("model.pkl")
			if (!model_fid)
			{
				topviewkinect::util::log_println("Failed to open model file !!");
			}
			std::cout << "fid = open('model')" << std::endl;

			PyObject* pickle_load_func = PyObject_GetAttrString(pickle_module, "load"); // pickle.load
			if (!pickle_load_func)
			{
				topviewkinect::util::log_println("Failed to load 'pickle.load' function !!");
			}

			this->gesture_recognition_phone_model = PyObject_CallFunctionObjArgs(pickle_load_func, model_fid, NULL); // model = pickle.load(fid)
			if (!this->gesture_recognition_phone_model)
			{
				topviewkinect::util::log_println("Failed to load gesture recognition model (via pickle) !!");
			}
			std::cout << "model = pickle.load(fid)" << std::endl;

			this->gesture_recognition_phone_predict_func = PyObject_GetAttrString(this->gesture_recognition_phone_model, "predict"); // model.predict
			if (!this->gesture_recognition_phone_predict_func)
			{
				topviewkinect::util::log_println("Failed to load 'model.load' function !!");
			}

			// Clean up
			Py_INCREF(this->gesture_recognition_phone_model);
			Py_INCREF(this->gesture_recognition_phone_predict_func);

			Py_DECREF(pickle_module);
			Py_DECREF(io_module);
			Py_DECREF(model_fid);
			Py_DECREF(pickle_load_func);

			topviewkinect::util::log_println("Gesture Recognition module initialized !");
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

		bool InteractionClassifier::recognize_gesture_phone(std::vector<topviewkinect::AndroidSensorData>& data, int* gesture_type) const
		{
			int window_size = 200;

			if (data.size() != window_size)
			{
				return false;
			}

			// calculate elapsed time
			double initial_data_arrival_time = data[0].arrival_time;

			// accelerometer x, y, z, and time
			//const int features_length = (3);
			const int features_length = (3+1);

			// Create C 2D array
			double* X_c_array = new double[features_length * window_size];
			for (int i = 0; i < window_size; ++i)
			{
				const topviewkinect::AndroidSensorData data_point = data[i];
				//std::array<double, features_length> sensor_data_features = { data_point.linear_accel_x, data_point.linear_accel_y, data_point.linear_accel_z};
				std::array<double, features_length> sensor_data_features = { data_point.linear_accel_x, data_point.linear_accel_y, data_point.linear_accel_z, data_point.arrival_time - initial_data_arrival_time};
				std::copy(sensor_data_features.begin(), sensor_data_features.end(), X_c_array + i * sensor_data_features.size());
			}

			// Create NumPy 2D Array
			//const int nrow = 1;
			//const int ncol = features_length * window_size;
			const int nrow = window_size;
			const int ncol = features_length;
			const int ndimension = 2;
			npy_intp shape[ndimension] = { nrow, ncol };
			PyObject* X_p_array = PyArray_SimpleNewFromData(ndimension, shape, NPY_FLOAT64, reinterpret_cast<void*>(X_c_array));
			if (!X_p_array)
			{
				topviewkinect::util::log_println("Failed to construct X Python Array !!");
			}

			PyArrayObject* X_np_array = reinterpret_cast<PyArrayObject*>(X_p_array);
			if (!X_np_array)
			{
				topviewkinect::util::log_println("Failed to construct X NumPy Array !!");
			}

			// FFT Feature extract
			PyObject* X_fft = PyObject_CallFunctionObjArgs(this->fft_features_func, X_np_array, NULL);
			PyArrayObject* X_fft_np_array = reinterpret_cast<PyArrayObject*>(X_fft);

			//PyObject* y_pred = PyObject_CallFunctionObjArgs(this->gesture_recognition_phone_predict_func, X_np_array, NULL); // y_pred = model.predict(X)
			PyObject* y_pred = PyObject_CallFunctionObjArgs(this->gesture_recognition_phone_predict_func, X_fft_np_array, NULL); // y_pred = model.predict(X)
			if (!y_pred)
			{
				topviewkinect::util::log_println("Failed to call predict function !!");
			}
			PyArrayObject* y_np_array = reinterpret_cast<PyArrayObject*>(y_pred);
			int* y_c_array = reinterpret_cast<int*>(PyArray_DATA(y_np_array));

			*gesture_type = y_c_array[0];

			// Clean up
			Py_DECREF(X_np_array);
			Py_DECREF(X_fft);
			Py_DECREF(y_pred);

			return true;
		}
	}
}