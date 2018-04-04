
/**
EagleSense infrastructure

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"

// Must appear at the top
#define _USE_MATH_DEFINES
#include <cmath>

#include <math.h>
#include <limits>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <memory>
#include <exception>
#include <ctime>
#include <ctype.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/shape/shape_distance.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/math/special_functions/round.hpp>

#include "topviewkinect/topviewkinect.h"
#include "topviewkinect/color.h"
#include "topviewkinect/util.h"
#include "topviewkinect/kinect2.h"
#include "topviewkinect/vision/space.h"

//// Important to include this before flowIO.h!
//#include "thirdparty/mpi/imageLib.h"
//#include "thirdparty/mpi/flowIO.h"
//#include "thirdparty/mpi/colorcode.h"
//
//// Liu OptFlow
//#include "thirdparty/liu_optflow/OpticalFlow.h"
//#include "thirdparty/liu_optflow/project.h"
//#include "thirdparty/liu_optflow/image.h"
//
//// YOLO
//#ifdef _WIN32
//#define OPENCV
//#endif
//#define TRACK_OPTFLOW
//#include "yolo_v2_class.hpp"
//#ifdef OPENCV
//#include <opencv2/opencv.hpp>
//#include "opencv2/core/version.hpp"

//std::vector<std::string> objects_names_from_file(std::string const filename) {
//	std::ifstream file(filename);
//	std::vector<std::string> file_lines;
//	if (!file.is_open()) return file_lines;
//	for (std::string line; getline(file, line);) file_lines.push_back(line);
//	std::cout << "object names loaded \n";
//	return file_lines;
//}
//
//void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
//	int current_det_fps = -1, int current_cap_fps = -1)
//{
//	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
//
//	for (auto &i : result_vec) {
//		cv::Scalar color = obj_id_to_color(i.obj_id);
//		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
//		if (obj_names.size() > i.obj_id) {
//			std::string obj_name = obj_names[i.obj_id];
//			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
//			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
//			int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
//			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0)),
//				cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
//				color, CV_FILLED, 8, 0);
//			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
//		}
//	}
//	if (current_det_fps >= 0 && current_cap_fps >= 0) {
//		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
//		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
//	}
//}
//#endif	// OPENCV
//
//void MotionToColor(CFloatImage motim, CByteImage &colim, float maxmotion)
//{
//	CShape sh = motim.Shape();
//	int width = sh.width, height = sh.height;
//	colim.ReAllocate(CShape(width, height, 3));
//	int x, y;
//	// determine motion range:
//	float maxx = -999, maxy = -999;
//	float minx = 999, miny = 999;
//	float maxrad = -1;
//	for (y = 0; y < height; y++) {
//		for (x = 0; x < width; x++) {
//			float fx = motim.Pixel(x, y, 0);
//			float fy = motim.Pixel(x, y, 1);
//			if (unknown_flow(fx, fy))
//				continue;
//			maxx = __max(maxx, fx);
//			maxy = __max(maxy, fy);
//			minx = __min(minx, fx);
//			miny = __min(miny, fy);
//			float rad = sqrt(fx * fx + fy * fy);
//			maxrad = __max(maxrad, rad);
//		}
//	}
//	printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
//		maxrad, minx, maxx, miny, maxy);
//
//
//	if (maxmotion > 0) // i.e., specified on commandline
//		maxrad = maxmotion;
//
//	if (maxrad == 0) // if flow == 0 everywhere
//		maxrad = 1;
//
//	for (y = 0; y < height; y++) {
//		for (x = 0; x < width; x++) {
//			float fx = motim.Pixel(x, y, 0);
//			float fy = motim.Pixel(x, y, 1);
//			uchar *pix = &colim.Pixel(x, y, 0);
//			if (unknown_flow(fx, fy)) {
//				pix[0] = pix[1] = pix[2] = 0;
//			}
//			else {
//				computeColor(fx / maxrad, fy / maxrad, pix);
//			}
//		}
//	}
//}

/*
Helper functions
=============================
*/

static cv::Mat draw_color_body_layers(const cv::Mat& depth_silhouette, const cv::Mat1i& best_labels, const cv::Mat1f& centers)
{
	std::vector<float> centers_ordered;
	for (int i = 0; i < centers.rows; ++i)
	{
		centers_ordered.push_back(std::floor(centers.at<float>(0, i)));
	}
	std::sort(centers_ordered.begin(), centers_ordered.end());

	std::vector<cv::Vec3b> color_centers;
	color_centers.push_back(cv::Vec3b(0, 0, 0));
	color_centers.push_back(cv::Vec3b(static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[0][0]), static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[0][1]), static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[0][2])));
	color_centers.push_back(cv::Vec3b(static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[1][0]), static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[1][1]), static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[1][2])));
	color_centers.push_back(cv::Vec3b(static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[2][0]), static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[2][1]), static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[2][2])));
	color_centers.push_back(cv::Vec3b(static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[3][0]), static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[3][1]), static_cast<uchar>(topviewkinect::color::CV_BGR_LAYERS[3][2])));
	cv::Mat3b color_body_layers(depth_silhouette.rows, depth_silhouette.cols);
	for (int r = 0; r < color_body_layers.rows; ++r)
	{
		for (int c = 0; c < color_body_layers.cols; ++c)
		{
			float center_value = std::floor(centers(best_labels(r*depth_silhouette.cols + c)));
			int color_idx = static_cast<int>(std::find(centers_ordered.begin(), centers_ordered.end(), center_value) - centers_ordered.begin());
			color_body_layers.at<cv::Vec3b>(r, c) = color_centers[color_idx];
		}
	}

	return color_body_layers;
}

// OPTICAL FLOW
cv::Ptr<INCVMemAllocator> g_pGPUMemAllocator;
cv::Ptr<INCVMemAllocator> g_pHostMemAllocator;

namespace topviewkinect
{
	namespace vision
	{
		TopViewSpace::TopViewSpace() :
			// Kinect
			kinect_sensor(NULL),
			kinect_multisource_frame_reader(NULL),
			kinect_coordinate_mapper(NULL),
			kinect_frame_id(0),
			kinect_depth_frame_timestamp(0),
			kinect_infrared_frame_timestamp(0),
			kinect_rgb_frame_timestamp(0),

			// Optflow
			brox_optflow(cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 3, 77, 3)),
			farn_optflow(cv::cuda::FarnebackOpticalFlow::create()),
			tvl1_optflow(cv::cuda::OpticalFlowDual_TVL1::create(0.25, 0.15, 0.3, 5, 5, 0.01, 100)),

			// Background extraction
			current_num_background_frames(topviewkinect::vision::REQUIRED_BACKGROUND_FRAMES),
			foreground_mask(cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1)),
			p_background_extractor(cv::createBackgroundSubtractorMOG2()),

			// Visualizations
			depth_frame(cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1)),
			depth_foreground_frame(cv::Mat(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1, topviewkinect::color::CV_WHITE)),
			infrared_frame(cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1)),
			infrared_low_frame(cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1)),
			rgb_frame(cv::Mat::zeros(topviewkinect::kinect2::CV_COLOR_FRAME_SIZE_DOWNSAMPLED, CV_8UC4)),
			visualization_frame(cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC3)),
			android_sensor_frame(cv::Mat::zeros(cv::Size(1400, 800), CV_8UC3)),
			crossmotion_calibration_frame(cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1)),

			// RESTful client
			restful_client(NULL)
		{
		}

		TopViewSpace::~TopViewSpace()
		{
			// Free Kinect
			if (this->kinect_sensor != NULL)
			{
				this->kinect_sensor->Close();
			}
			topviewkinect::util::safe_release(&this->kinect_sensor);
			topviewkinect::util::safe_release(&this->kinect_multisource_frame_reader);
			topviewkinect::util::safe_release(&this->kinect_coordinate_mapper);

			// Stop OpenPose
			//this->op_wrapper.stop();

			// Free RESTful client
			delete this->restful_client;
		}

		bool TopViewSpace::initialize()
		{
			topviewkinect::util::log_println("Initializing...");

			try
			{
				boost::property_tree::ptree config_root;
				boost::property_tree::read_json(topviewkinect::get_config_filepath(), config_root);

				this->configuration.framerate = config_root.get<int>("tracking.framerate", 0) == 1;
				this->configuration.orientation_recognition = config_root.get<int>("tracking.orientation_recognition", 0) == 1;
				this->configuration.interaction_recognition = config_root.get<int>("tracking.interaction_recognition", 0) == 1;
				this->configuration.restful_connection = config_root.get<int>("tracking.restful_connection", 0) == 1;

				this->configuration.interaction_model = config_root.get<std::string>("interaction_model");
				this->configuration.restful_server_address = config_root.get<std::string>("restful_server.address");
				this->configuration.restful_server_port = config_root.get<int>("restful_server.port");

				this->configuration.depth = config_root.get<int>("data.depth", 0) == 1;
				this->configuration.infrared = config_root.get<int>("data.infrared", 0) == 1;
				this->configuration.color = config_root.get<int>("data.color", 0) == 1;
			}
			catch (...)
			{
				topviewkinect::util::log_println("JSON parser failed.");
				return false;
			}

			// Machine Learning module
			this->interaction_classifier.initialize();

			// Interaction Recognition
			std::vector<std::string> interactions{
				"standing", "sitting", "pointing", "phone", "tablet", "paper"
			};
			if (this->configuration.interaction_recognition)
			{
				bool classifier_loaded = this->interaction_classifier.initialize_interaction_classification(this->configuration.interaction_model, interactions);
				if (!classifier_loaded)
				{
					topviewkinect::util::log_println("Failed to load interaction classifier.");
					return false;
				}
			}

			// Gesture Recognition
			bool phone_gesture_recognition = this->interaction_classifier.initialize_gesture_recognition_phone();
			if (!phone_gesture_recognition)
			{
				topviewkinect::util::log_println("Failed to load gesture recognition (phone) classifier.");
				return false;
			}

			// RESTful connection
			if (this->configuration.restful_connection)
			{
				web::uri_builder url_builder;
				url_builder.set_host(boost::locale::conv::utf_to_utf<wchar_t>(this->configuration.restful_server_address));
				url_builder.set_port(this->configuration.restful_server_port);
				this->restful_client = new web::http::client::http_client(url_builder.to_uri());
			}

			// KINECT v2
			HRESULT hr;
			hr = GetDefaultKinectSensor(&this->kinect_sensor);
			if (FAILED(hr) || !this->kinect_sensor)
			{
				topviewkinect::util::log_println("Failed to find a Kinect device!");
				topviewkinect::util::safe_release(&this->kinect_sensor);
				return false;
			}

			hr = this->kinect_sensor->Open();
			if (FAILED(hr))
			{
				topviewkinect::util::log_println("Failed to open Kinect!");
				topviewkinect::util::safe_release(&this->kinect_sensor);
				return false;
			}

			DWORD kinect_frames_types = FrameSourceTypes_Depth | FrameSourceTypes_LongExposureInfrared;
			if (this->configuration.color)
			{
				kinect_frames_types = FrameSourceTypes_Depth | FrameSourceTypes_LongExposureInfrared | FrameSourceTypes_Color;
			}

			hr = this->kinect_sensor->get_CoordinateMapper(&this->kinect_coordinate_mapper);
			if (FAILED(hr))
			{
				topviewkinect::util::log_println("Failed to open Kinect v2 Coordinate Mapper");
				topviewkinect::util::safe_release(&this->kinect_multisource_frame_reader);
				return false;
			}

			hr = this->kinect_sensor->OpenMultiSourceFrameReader(kinect_frames_types, &this->kinect_multisource_frame_reader);
			if (FAILED(hr))
			{
				topviewkinect::util::log_println("Failed to open Kinect v2 Multisource Frame Reader");
				topviewkinect::util::safe_release(&this->kinect_multisource_frame_reader);
				return false;
			}
			topviewkinect::util::log_println("Connected to Kinect v2 !");

			// CUDA OPTICAL FLOW NVIDIA API
			int devId;
			ncvAssertCUDAReturn(cudaGetDevice(&devId), -1);
			cudaDeviceProp devProp;
			ncvAssertCUDAReturn(cudaGetDeviceProperties(&devProp, devId), -1);
			std::cout << "Using GPU: " << devId << "(" << devProp.name << "), arch=" << devProp.major << "." << devProp.minor << std::endl;

			g_pGPUMemAllocator = cv::Ptr<INCVMemAllocator>(new NCVMemNativeAllocator(NCVMemoryTypeDevice, static_cast<Ncv32u>(devProp.textureAlignment)));
			ncvAssertPrintReturn(g_pGPUMemAllocator->isInitialized(), "Device memory allocator isn't initialized", -1);

			g_pHostMemAllocator = cv::Ptr<INCVMemAllocator>(new NCVMemNativeAllocator(NCVMemoryTypeHostPageable, static_cast<Ncv32u>(devProp.textureAlignment)));
			ncvAssertPrintReturn(g_pHostMemAllocator->isInitialized(), "Host memory allocator isn't initialized", -1);

			//// OpenPose
			//const auto OP_POSE_ENABLE = true;
			//const auto OP_POSE_NET_INPUT_SIZE = op::Point<int>{ 1280, 720 };
			//const auto OP_POSE_NET_OUTPUT_SIZE = op::Point<int>{ -1, -1 };
			//const auto OP_POSE_KEY_POINT_SCALE = op::ScaleMode::InputResolution;
			//const auto OP_POSE_GPU_NUMBER = -1;
			//const auto OP_POSE_GPU_START = 0;
			//const auto OP_POSE_SCALE_NUMBER = 1;
			//const auto OP_POSE_SCALE_GAP = 0.3f;
			//const auto OP_RENDER_POSE = op::RenderMode::Gpu;
			//const auto OP_POSE_MODEL = op::PoseModel::COCO_18;
			//const auto OP_POSE_BELND_ORIGINAL_FRAME = true;
			//const auto OP_POSE_ALPHA = 0.6f;
			//const auto OP_POSE_ALPHA_HEATMAP = 0.7f;
			//const auto OP_POSE_PART_RENDER = 0;
			//const auto OP_POSE_MODEL_FOLDER = "D:\\p_eaglesense\\eaglesense\\3rdparty\\openpose\\models";
			//const std::vector<op::HeatMapType> OP_POSE_HEATMAP_TYPES = {};
			//const auto OP_POSE_HEATMAP_SCALE = op::ScaleMode::UnsignedChar;
			//const auto OP_POSE_ADD_PARTS_CANDIDATES = false;
			//const auto OP_POSE_RENDER_THRESHOLD = 0.05f;
			//const auto OP_POSE_GOOGLE_LOGGING = false;
			//const auto OP_POSE_IDENTIFICATION = false;

			//const op::WrapperStructPose wrapper_struct_pose{ OP_POSE_ENABLE, OP_POSE_NET_INPUT_SIZE, OP_POSE_NET_OUTPUT_SIZE, OP_POSE_KEY_POINT_SCALE, OP_POSE_GPU_NUMBER, OP_POSE_GPU_START, OP_POSE_SCALE_NUMBER, OP_POSE_SCALE_GAP, OP_RENDER_POSE, OP_POSE_MODEL, OP_POSE_BELND_ORIGINAL_FRAME, OP_POSE_ALPHA, OP_POSE_ALPHA_HEATMAP, OP_POSE_PART_RENDER, OP_POSE_MODEL_FOLDER, OP_POSE_HEATMAP_TYPES, OP_POSE_HEATMAP_SCALE, OP_POSE_ADD_PARTS_CANDIDATES, OP_POSE_RENDER_THRESHOLD, OP_POSE_GOOGLE_LOGGING, OP_POSE_IDENTIFICATION };

			//const op::WrapperStructFace wrapper_struct_face{};
			////const op::WrapperStructFace wrapperStructFace{ FLAGS_face, faceNetInputSize,
			////	op::flagsToRenderMode(FLAGS_face_render, FLAGS_render_pose),
			////	(float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap,
			////	(float)FLAGS_face_render_threshold };
			////

			//const op::WrapperStructHand wrapper_struct_hand{};
			////const op::WrapperStructHand wrapperStructHand{ FLAGS_hand, handNetInputSize, FLAGS_hand_scale_number,
			////	(float)FLAGS_hand_scale_range, FLAGS_hand_tracking,
			////	op::flagsToRenderMode(FLAGS_hand_render, FLAGS_render_pose),
			////	(float)FLAGS_hand_alpha_pose, (float)FLAGS_hand_alpha_heatmap,
			////	(float)FLAGS_hand_render_threshold };

			//this->op_wrapper.configure(wrapper_struct_pose, wrapper_struct_face, wrapper_struct_hand, op::WrapperStructInput{}, op::WrapperStructOutput{});
			//this->op_wrapper.start();
			//this->op_processing = false;

			return true;
		}

		bool TopViewSpace::refresh_kinect_frames()
		{
			if (!this->kinect_multisource_frame_reader)
			{
				std::cout << "Kinect multisource frame reader not initialized!" << std::endl;
				return false;
			}

			IMultiSourceFrame* p_multisource_frame = NULL;
			IDepthFrame* p_depth_frame = NULL;
			ILongExposureInfraredFrame* p_infrared_frame = NULL;
			IColorFrame* p_color_frame = NULL;

			HRESULT hr;
			hr = this->kinect_multisource_frame_reader->AcquireLatestFrame(&p_multisource_frame);
			// Since the Kinect v2 runs at 30FPS, the new frame may not be ready yet
			if (FAILED(hr))
			{
				//std::cout << ".";
				return false;
			}
			std::cout << std::endl;

			// Get depth frame
			if (SUCCEEDED(hr))
			{
				IDepthFrameReference* p_depth_frame_ref = NULL;
				hr = p_multisource_frame->get_DepthFrameReference(&p_depth_frame_ref);
				if (SUCCEEDED(hr))
				{
					hr = p_depth_frame_ref->AcquireFrame(&p_depth_frame);
				}
				topviewkinect::util::safe_release(&p_depth_frame_ref);
			}

			// Get infrared frame
			if (SUCCEEDED(hr))
			{
				ILongExposureInfraredFrameReference* p_infrared_frame_ref = NULL;
				hr = p_multisource_frame->get_LongExposureInfraredFrameReference(&p_infrared_frame_ref);
				if (SUCCEEDED(hr))
				{
					hr = p_infrared_frame_ref->AcquireFrame(&p_infrared_frame);
				}
				topviewkinect::util::safe_release(&p_infrared_frame_ref);
			}

			// Get rgb frame
			if (this->configuration.color && SUCCEEDED(hr))
			{
				IColorFrameReference* p_color_frame_ref = NULL;
				hr = p_multisource_frame->get_ColorFrameReference(&p_color_frame_ref);
				if (SUCCEEDED(hr))
				{
					hr = p_color_frame_ref->AcquireFrame(&p_color_frame);
				}
				topviewkinect::util::safe_release(&p_color_frame_ref);
			}

			// Process depth frame
			IFrameDescription* p_depth_frame_description = NULL;
			int depth_width = 0;
			int depth_height = 0;
			unsigned short depth_min_distance = 0;
			unsigned short depth_max_distance = 0;
			unsigned short* p_depth_buffer = NULL;
			if (SUCCEEDED(hr))
			{
				hr = p_depth_frame->get_FrameDescription(&p_depth_frame_description);
				if (SUCCEEDED(hr))
				{
					hr = p_depth_frame_description->get_Width(&depth_width);
				}
				if (SUCCEEDED(hr))
				{
					hr = p_depth_frame_description->get_Height(&depth_height);
				}
				if (depth_width != topviewkinect::kinect2::DEPTH_WIDTH || depth_height != topviewkinect::kinect2::DEPTH_HEIGHT)
				{
					hr = FALSE;
				}
				if (SUCCEEDED(hr))
				{
					hr = p_depth_frame->get_DepthMinReliableDistance(&depth_min_distance);
				}
				if (SUCCEEDED(hr))
				{
					hr = p_depth_frame->get_DepthMaxReliableDistance(&depth_max_distance);
				}
				if (SUCCEEDED(hr))
				{
					p_depth_buffer = new unsigned short[topviewkinect::kinect2::DEPTH_BUFFER_SIZE];
					hr = p_depth_frame->CopyFrameDataToArray(topviewkinect::kinect2::DEPTH_BUFFER_SIZE, p_depth_buffer);

					if (SUCCEEDED(hr))
					{
						p_depth_frame->get_RelativeTime(&this->kinect_depth_frame_timestamp);

						// Update depth frame
						cv::Mat depth_map(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_16UC1, p_depth_buffer);
						depth_map.convertTo(this->depth_frame, CV_8UC1, 256.0 / 8000.0);
					}
				}
			}

			// Process infrared frame
			IFrameDescription* p_infrared_frame_description = NULL;
			int infrared_width = 0;
			int infrared_height = 0;
			unsigned short* p_infrared_buffer = NULL;
			if (SUCCEEDED(hr))
			{
				hr = p_infrared_frame->get_FrameDescription(&p_infrared_frame_description);
				if (SUCCEEDED(hr))
				{
					hr = p_infrared_frame_description->get_Width(&infrared_width);
				}
				if (SUCCEEDED(hr))
				{
					hr = p_infrared_frame_description->get_Height(&infrared_height);
				}
				if (infrared_width != topviewkinect::kinect2::DEPTH_WIDTH || infrared_height != topviewkinect::kinect2::DEPTH_HEIGHT)
				{
					hr = FALSE;
				}
				if (SUCCEEDED(hr))
				{
					p_infrared_buffer = new unsigned short[topviewkinect::kinect2::DEPTH_BUFFER_SIZE];
					hr = p_infrared_frame->CopyFrameDataToArray(topviewkinect::kinect2::DEPTH_BUFFER_SIZE, p_infrared_buffer);

					if (SUCCEEDED(hr))
					{
						p_infrared_frame->get_RelativeTime(&this->kinect_infrared_frame_timestamp);

						// Update infrared frame
						cv::Mat infrared_map(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_16UC1, p_infrared_buffer);

						//this->infrared_frame.forEach<uchar>([&](uchar &p, const int* position) {
						//	p = infrared_map.at<ushort>(position[0], position[1]) >> 8;
						//});
						infrared_map.convertTo(this->infrared_frame, CV_8UC1, 256.0 / 8000.0);
						infrared_map.convertTo(this->infrared_low_frame, CV_8UC1);

						// CrossMotion Calibration
						this->infrared_frame.copyTo(this->crossmotion_calibration_frame);
						cv::cvtColor(this->crossmotion_calibration_frame, this->crossmotion_calibration_frame, CV_GRAY2BGR);
						for (int i = 0; i < this->crossmotion_calibration_points_2d.size(); ++i)
						{
							cv::Point infrared_2d = this->crossmotion_calibration_points_2d[i];
							cv::circle(this->crossmotion_calibration_frame, cv::Point(infrared_2d.x, infrared_2d.y), 3, topviewkinect::color::CV_BGR_RED, -1);
						}
					}
				}
			}

			// Process color frame
			IFrameDescription* p_color_frame_description = NULL;
			int color_width = 0;
			int color_height = 0;
			ColorImageFormat image_format = ColorImageFormat_None;
			unsigned int color_buffer_size = 0;
			RGBQUAD* p_color_buffer = NULL;
			RGBQUAD* p_color_RGBX = new RGBQUAD[topviewkinect::kinect2::COLOR_WIDTH * topviewkinect::kinect2::COLOR_HEIGHT];
			if (this->configuration.color && SUCCEEDED(hr))
			{
				hr = p_color_frame->get_FrameDescription(&p_color_frame_description);
				if (SUCCEEDED(hr))
				{
					hr = p_color_frame_description->get_Width(&color_width);
				}
				if (SUCCEEDED(hr))
				{
					hr = p_color_frame_description->get_Height(&color_height);
				}
				if (color_width != topviewkinect::kinect2::COLOR_WIDTH || color_height != topviewkinect::kinect2::COLOR_HEIGHT)
				{
					hr = FALSE;
				}
				if (SUCCEEDED(hr))
				{
					hr = p_color_frame->get_RawColorImageFormat(&image_format);
				}
				if (SUCCEEDED(hr))
				{
					hr = p_color_frame->get_RelativeTime(&this->kinect_rgb_frame_timestamp);

					// Update rgb frame
					if (image_format == ColorImageFormat_Bgra)
					{
						hr = p_color_frame->AccessRawUnderlyingBuffer(&color_buffer_size, reinterpret_cast<unsigned char**>(&p_color_buffer));
					}
					else if (p_color_RGBX)
					{
						p_color_buffer = p_color_RGBX;
						color_buffer_size = color_width * color_height * sizeof(RGBQUAD);
						hr = p_color_frame->CopyConvertedFrameDataToArray(color_buffer_size, reinterpret_cast<unsigned char*>(p_color_buffer), ColorImageFormat_Bgra);
					}
					else
					{
						hr = E_FAIL;
					}

					if (SUCCEEDED(hr))
					{
						unsigned char* byte_image = reinterpret_cast<unsigned char*>(p_color_buffer);
						cv::Mat m = cv::Mat(topviewkinect::kinect2::CV_COLOR_FRAME_SIZE, CV_8UC4, reinterpret_cast<void*>(byte_image));
						cv::pyrDown(m, this->rgb_frame, topviewkinect::kinect2::CV_COLOR_FRAME_SIZE_DOWNSAMPLED);
					}
				}
			}

			// Calibration
			if (SUCCEEDED(hr))
			{
				int num_calibration_points = this->crossmotion_calibration_points_to_add.size();

				for (int i = 0; i < num_calibration_points; ++i)
				{
					const cv::Point infrared_2d = this->crossmotion_calibration_points_to_add[i];

					DepthSpacePoint dsp = DepthSpacePoint{ static_cast<float>(infrared_2d.x), static_cast<float>(infrared_2d.y) };
					long depth_index = dsp.Y * topviewkinect::kinect2::DEPTH_WIDTH + dsp.X;
					unsigned short depth = p_depth_buffer[depth_index];
					CameraSpacePoint csp;

					HRESULT mapper_hr = this->kinect_coordinate_mapper->MapDepthPointToCameraSpace(dsp, depth, &csp);
					if (SUCCEEDED(mapper_hr))
					{
						std::cout << "mapper : success" << std::endl;
						if (csp.X != -std::numeric_limits<float>::infinity() && csp.Y != -std::numeric_limits<float>::infinity())
						{
							std::cout << "depth : " << dsp.X << " , " << dsp.Y << std::endl;
							std::cout << "csp : " << csp.X << " , " << csp.Y << " , " << csp.Z << std::endl;
							this->crossmotion_calibration_points_2d.push_back(infrared_2d);
							this->crossmotion_calibration_points_3d.push_back(csp);
						}
					}
					else
					{
						std::cout << "mapper : fail" << std::endl;
					}
				}

				this->crossmotion_calibration_points_to_add.clear();
			}

			// Clean Up
			delete[] p_depth_buffer;
			p_depth_buffer = NULL;
			topviewkinect::util::safe_release(&p_depth_frame_description);
			topviewkinect::util::safe_release(&p_depth_frame);

			delete[] p_infrared_buffer;
			p_infrared_buffer = NULL;
			topviewkinect::util::safe_release(&p_infrared_frame_description);
			topviewkinect::util::safe_release(&p_infrared_frame);

			delete[] p_color_buffer;
			p_color_buffer = NULL;
			topviewkinect::util::safe_release(&p_color_frame_description);
			topviewkinect::util::safe_release(&p_color_frame);
			topviewkinect::util::safe_release(&p_multisource_frame);

			if (SUCCEEDED(hr))
			{
				this->kinect_multisource_frame_arrived = true;
				return true;
			}
			else
			{
				return false;
			}
		}

		bool TopViewSpace::refresh_android_sensor_data(topviewkinect::AndroidSensorData& data)
		{
			this->android_sensor_mutex.lock();
			this->android_sensor_data.push_back(data);
			this->android_sensor_data_tmp.push_back(data);
			if (this->android_sensor_data.size() > 250)
			{
				this->android_sensor_data.pop_front();
			}
			this->android_sensor_mutex.unlock();
			return true;
		}

		bool TopViewSpace::refresh_android_sensor_frame()
		{
			// update sensor plot
			this->android_sensor_frame.setTo(topviewkinect::color::CV_BGR_WHITE);

			// device acceleration, X axis
			int accel_x_origin_x = 20, accel_x_origin_y = 250;
			{
				cv::putText(this->android_sensor_frame, "0 - ", cv::Point(5, 150), CV_FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_BGR_BLACK, 1);
				cv::line(this->android_sensor_frame, cv::Point(accel_x_origin_x, accel_x_origin_y), cv::Point(accel_x_origin_x, 50), topviewkinect::color::CV_BGR_BLACK, 1);
				cv::line(this->android_sensor_frame, cv::Point(accel_x_origin_x, accel_x_origin_y), cv::Point(accel_x_origin_x + 250 * 5, accel_x_origin_y), topviewkinect::color::CV_BGR_BLACK, 1);
			}

			// device acceleration, Y axis
			int accel_y_origin_x = 20, accel_y_origin_y = 500;
			{
				cv::putText(this->android_sensor_frame, "0 - ", cv::Point(5, 400), CV_FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_BGR_BLACK, 1);
				cv::line(this->android_sensor_frame, cv::Point(accel_y_origin_x, accel_y_origin_y), cv::Point(accel_y_origin_x, 300), topviewkinect::color::CV_BGR_BLACK, 1);
				cv::line(this->android_sensor_frame, cv::Point(accel_y_origin_x, accel_y_origin_y), cv::Point(accel_y_origin_x + 250 * 5, accel_y_origin_y), topviewkinect::color::CV_BGR_BLACK, 1);
			}

			// device acceleration, Z axis
			int accel_z_origin_x = 20, accel_z_origin_y = 750;
			{
				cv::putText(this->android_sensor_frame, "0 - ", cv::Point(5, 650), CV_FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_BGR_BLACK, 1);
				cv::line(this->android_sensor_frame, cv::Point(accel_z_origin_x, accel_z_origin_y), cv::Point(accel_z_origin_x, 550), topviewkinect::color::CV_BGR_BLACK, 1);
				cv::line(this->android_sensor_frame, cv::Point(accel_z_origin_x, accel_z_origin_y), cv::Point(accel_z_origin_x + 250 * 5, accel_z_origin_y), topviewkinect::color::CV_BGR_BLACK, 1);
			}

			// frame seapration
			int total_length = accel_x_origin_x + 250 * 5;
			int frame_length = total_length / 10;
			for (int i = 1; i < 10; ++i)
			{
				cv::line(this->android_sensor_frame, cv::Point(accel_x_origin_x + i * frame_length, accel_x_origin_y), cv::Point(accel_x_origin_x + i * frame_length, 50), topviewkinect::color::CV_BGR_GREY, 1);
			}

			// data
			this->android_sensor_mutex.lock();
			for (int i = 0; i < this->android_sensor_data.size(); ++i)
			{
				const topviewkinect::AndroidSensorData sensor_data = this->android_sensor_data[i];

				// device acceleration X
				cv::circle(this->android_sensor_frame, cv::Point(accel_x_origin_x + i * 5, (accel_x_origin_y - 100) - (sensor_data.linear_accel_x) * 10), 3, topviewkinect::color::CV_BGR_RED, -1);

				// device acceleration Y
				cv::circle(this->android_sensor_frame, cv::Point(accel_y_origin_x + i * 5, (accel_y_origin_y - 100) - (sensor_data.linear_accel_y) * 10), 3, topviewkinect::color::CV_BGR_GREEN, -1);

				// device acceleration Z
				cv::circle(this->android_sensor_frame, cv::Point(accel_z_origin_x + i * 5, (accel_z_origin_y - 100) - (sensor_data.linear_accel_z) * 10), 3, topviewkinect::color::CV_BGR_BLUE, -1);
			}
			this->android_sensor_mutex.unlock();

			// gesture
			std::string gesture_str = "Phone : rest";
			if (this->current_gesture == 1)
			{
				gesture_str = "Phone : inward";
			}
			else if (this->current_gesture == 2)
			{
				gesture_str = "Phone : outward";
			}

			cv::putText(this->android_sensor_frame, gesture_str, cv::Point(100, 50), CV_FONT_HERSHEY_COMPLEX, 1.3, topviewkinect::color::CV_BGR_BLACK, 2);

			return true;
		}

		bool TopViewSpace::process_android_sensor_data()
		{
			// Recognize gesture
			std::vector<topviewkinect::AndroidSensorData> android_sensor_sample;

			this->android_sensor_mutex.lock();
			int sensor_data_size = this->android_sensor_data.size();
			if (sensor_data_size >= 210)
			{
				for (int i = 0; i < 210; ++i)
				{
					int index = sensor_data_size - 210 + i;
					android_sensor_sample.push_back(this->android_sensor_data[index]);
				}
			}
			this->android_sensor_mutex.unlock();

			int gesture_type;
			bool predicted = this->interaction_classifier.recognize_gesture_phone(android_sensor_sample, &gesture_type);
			if (predicted)
			{
				this->current_gesture = gesture_type;
			}

			return true;
		}

		const int TopViewSpace::get_kinect_frame_id() const
		{
			return this->kinect_frame_id;
		}

		// Sensor Fusion

		//cv::Mat calibration_im_infrared, calibration_im_depth;
		//cv::Mat calibration_im_infrared_annotated;
		//std::string calibration_window_name = "Calibration Image", calibration_depth_window_name = "Calibration Image (Depth)";
		//std::vector<cv::Point2i> calibration_mouse_clicks;
		//cv::Point next_calibration_point;
		//bool calibration_drawing = false;

		CameraSpacePoint* infrared_to_csps = new CameraSpacePoint[topviewkinect::kinect2::DEPTH_WIDTH * topviewkinect::kinect2::DEPTH_HEIGHT];
		CameraSpacePoint* color_to_camera_space_points = new CameraSpacePoint[topviewkinect::kinect2::COLOR_WIDTH * topviewkinect::kinect2::COLOR_HEIGHT];
		DepthSpacePoint* dsps = new DepthSpacePoint[topviewkinect::kinect2::COLOR_WIDTH * topviewkinect::kinect2::COLOR_HEIGHT];

		// Must done online
		bool TopViewSpace::calibrate_sensor_fusion(const cv::Point& new_point)
		{
			this->crossmotion_calibration_points_to_add.push_back(cv::Point{ new_point.x, new_point.y });

			//// infrared
			////calibration_im_infrared = cv::imread("D:\\p_eaglesense\\eaglesense\\data\\topviewkinect\\3006\\infrared\\calibration\\451.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
			//this->infrared_frame.copyTo(this->crossmotion_calibration_frame);
			//cv::cvtColor(this->crossmotion_calibration_frame, this->crossmotion_calibration_frame, CV_GRAY2BGR);
			////cv::namedWindow(calibration_window_name);
			////cv::setMouseCallback(calibration_window_name, calibration_on_mouse, this);
			////cv::imshow(calibration_window_name, calibration_im_infrared_annotated);

			////calibration_im_depth = cv::imread("D:\\p_eaglesense\\eaglesense\\data\\topviewkinect\\3006\\depth\\calibration\\451.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
			////this->depth_frame.copyTo(calibration_im_depth);
			////cv::imshow(calibration_depth_window_name, calibration_im_depth);

			//int num_depth_points = topviewkinect::kinect2::DEPTH_WIDTH * topviewkinect::kinect2::DEPTH_HEIGHT;
			//int num_calibration_points = calibration_points.size();

			//DepthSpacePoint* calibration_dsps = new DepthSpacePoint[num_calibration_points];
			//for (int i = 0; i < calibration_points.size(); ++i)
			//{
			//	const cv::Point calibration_2d_point = calibration_points[i];
			//	DepthSpacePoint dsp = { calibration_2d_point.x, calibration_2d_point.y };
			//	calibration_dsps[i] = dsp;
			//}

			//CameraSpacePoint* calibration_csps = new CameraSpacePoint[num_calibration_points];
			//HRESULT hr = this->kinect_coordinate_mapper->MapDepthPointsToCameraSpace(num_calibration_points, calibration_dsps, num_depth_points, this->depth_frame.ptr<UINT16>(), num_calibration_points, calibration_csps);
			//if (SUCCEEDED(hr))
			//{
			//	std::cout << "hr : success" << std::endl;
			//}
			//else
			//{
			//	std::cout << "hr : failed" << std::endl;
			//}

			//for (int i = 0; i < calibration_points.size(); ++i)
			//{
			//	DepthSpacePoint dsp = calibration_dsps[i];
			//	CameraSpacePoint csp = calibration_csps[i];
			//	std::cout << "depth : " << dsp.X << " , " << dsp.Y << std::endl;
			//	std::cout << "csp : " << csp.X << " , " << csp.Y << " , " << csp.Z << std::endl;
			//	if (csp.X != -std::numeric_limits<float>::infinity() && csp.Y != -std::numeric_limits<float>::infinity())
			//	{
			//		//int depth_X = static_cast<int>(dsp.X + 0.5f);
			//		//int depth_Y = static_cast<int>(dsp.Y + 0.5f);
			//		//if ((depth_X >= 0 && depth_X < topviewkinect::kinect2::DEPTH_WIDTH) && (depth_Y >= 0 && depth_Y < topviewkinect::kinect2::DEPTH_HEIGHT))
			//		//{
			//		cv::circle(this->crossmotion_calibration_frame, cv::Point(dsp.X, dsp.Y), 3, topviewkinect::color::CV_BGR_RED, -1);
			//		//}
			//	}
			//	else
			//	{
			//		std::cout << "-inf" << std::endl;
			//	}
			//}

			//// Redraw bounding box and rectangle
			//cv::circle(calibration_im_infrared_annotated, cv::Point(x, y), 3, topviewkinect::color::CV_BGR_RED, -1);
			////cv::circle(calibration_im_infrared_annotated, next_calibration_point, 5, topviewkinect::color::CV_BGR_RED, -1);
			//cv::imshow(calibration_window_name, calibration_im_infrared_annotated);

			//cv::circle(calibration_im_depth, cv::Point(x, y), 3, topviewkinect::color::CV_WHITE, -1);
			//cv::imshow(calibration_depth_window_name, calibration_im_depth);

			////calibration_mouse_clicks.push_back(cv::Point2i(next_calibration_point.x, next_calibration_point.y));

			//calibration_drawing = false;

			//// Show on depth image
			//long infrared_index = (long)(y * topviewkinect::kinect2::DEPTH_WIDTH + x);
			//CameraSpacePoint csp = infrared_to_csps[infrared_index];
			////calibration_world_coordinates.push_back(DepthSpacePoint{ -csp.X, -csp.Y, csp.Z });

			//std::cout << "index : " << infrared_index << std::endl;


			//while (true)
			//{
			//	int ascii_keypress = cv::waitKeyEx(0);
			//	std::cout << "ascii : " << ascii_keypress << std::endl;
			//	if (ascii_keypress == 27)
			//	{
			//		break;
			//	}
			//}

			return true;
		}

		void cross_product_normal(float vec_ab[3], float vec_ac[3], float* vec_res)
		{
			float vec_n_x = vec_ab[1] * vec_ac[2] - vec_ab[2] * vec_ac[1];
			float vec_n_y = vec_ab[2] * vec_ac[0] - vec_ab[0] * vec_ac[2];
			float vec_n_z = vec_ab[0] * vec_ac[1] - vec_ab[1] * vec_ac[0];
			float sum = vec_n_x + vec_n_y + vec_n_z;

			float unit_n_x = vec_n_x / sum;
			float unit_n_y = vec_n_y / sum;
			float unit_n_z = vec_n_z / sum;

			vec_res[0] = unit_n_x;
			vec_res[1] = unit_n_y;
			vec_res[2] = unit_n_z;
		}

		bool TopViewSpace::offline_calibration()
		{
			// infrared
			cv::Mat calibration_im_depth = cv::imread("D:\\p_eaglesense\\eaglesense\\data\\topviewkinect\\3007\\depth\\calibration\\400.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat calibration_im_infrared = cv::imread("D:\\p_eaglesense\\eaglesense\\data\\topviewkinect\\3007\\infrared\\calibration\\400.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat calibration_im_rgb = cv::imread("D:\\p_eaglesense\\eaglesense\\data\\topviewkinect\\3007\\rgb\\calibration\\400.jpeg", CV_LOAD_IMAGE_COLOR);
			cv::cvtColor(calibration_im_infrared, calibration_im_infrared, CV_GRAY2BGR);

			// draw calibration points
			cv::circle(calibration_im_infrared, cv::Point(201, 184), 3, topviewkinect::color::CV_BGR_RED, -1);
			cv::circle(calibration_im_infrared, cv::Point(198, 147), 3, topviewkinect::color::CV_BGR_RED, -1);
			cv::circle(calibration_im_infrared, cv::Point(268, 145), 3, topviewkinect::color::CV_BGR_RED, -1);
			cv::circle(calibration_im_infrared, cv::Point(270, 187), 3, topviewkinect::color::CV_BGR_RED, -1);

			// three points from infrared image
			float a[3] = { -0.43404, 0.144404, 2.596 };
			float b[3] = { -0.501964,0.447939,2.867 };
			float c[3] = { 0.0445801,0.466749,2.881 };
			float vec_ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
			float vec_ac[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };

			// k_1 = n_k
			float vec_n_k[3];
			cross_product_normal(vec_ab, vec_ac, vec_n_k);

			// g_k kinect gravity
			float g_k[3] = { -0.771, 6.607, -7.206 };

			// k_2 = n_k x g_k
			float vec_k_2[3];
			cross_product_normal(vec_n_k, g_k, vec_k_2);

			// k_3 = n_k x k_2
			float vec_k_3[3];
			cross_product_normal(vec_n_k, vec_k_2, vec_k_3);

			// K
			float K[3][3] = {
				{ vec_n_k[0], vec_n_k[1], vec_n_k[2] },
				{ vec_k_2[0], vec_k_2[1], vec_k_2[2] },
				{ vec_k_3[0], vec_k_3[1], vec_k_3[2] }
			};

			std::cout << "k_1 : " << vec_n_k[0] << " , " << vec_n_k[1] << " , " << vec_n_k[2] << std::endl;
			std::cout << "k_2 : " << vec_k_2[0] << " , " << vec_k_2[1] << " , " << vec_k_2[2] << std::endl;
			std::cout << "k_2 : " << vec_k_3[0] << " , " << vec_k_3[1] << " , " << vec_k_3[2] << std::endl;

			// computer surface normal n_w
			//frame_id, depth_time, addr, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, orientation_x, orientation_y, orientation_z, linear_accel_x, linear_accel_y, linear_accel_z, gravity_x, gravity_y, gravity_z, rotation_x, rotation_y, rotation_z
			//451,735653787547,0.0.0.0,-0.006,0.040,-9.920,-0.000,-0.000,0.000,80.258,-179.806,-0.073,-0.002,0.003,-0.010,-0.012,0.033,-9.807,0.765,-0.645,-0.002
			float phone_unit_vec[3] = { 0, 0, 1 };
			float phone_orientation[3] = { 80.258,-179.806,-0.073 };

			// g_w phone gravity
			float g_w[3] = { -0.012f,0.033f,-9.807f };

			// w_1 = n_w
			float n_w[3];
			cross_product_normal(phone_unit_vec, phone_orientation, n_w);

			// w_2
			float w_2[3];
			cross_product_normal(n_w, g_w, w_2);

			// w_3
			float w_3[3];
			cross_product_normal(n_w, w_2, w_3);

			// W
			float W[3][3] = {
				{ n_w[0], n_w[1], n_w[2] },
				{ w_2[0], w_2[1], w_2[2] },
				{ w_3[0], w_3[1], w_3[2] }
			};

			cv::imshow("calibration_im_depth", calibration_im_depth);
			cv::imshow("calibration_im_infrared", calibration_im_infrared);
			cv::imshow("calibration_im_rgb", calibration_im_rgb);

			return true;
		}

		// Replay

		bool TopViewSpace::load_dataset(const int dataset_id)
		{
			topviewkinect::util::log_println("Loading Dataset (" + std::to_string(dataset_id) + ") ...");

			this->interaction_log.initialize(dataset_id);
			bool dataset_loaded = this->interaction_log.load_directories();
			if (!dataset_loaded)
			{
				topviewkinect::util::log_println("Failed to load dataset.");
				return false;
			}

			// Perform initial background subtraction
			const std::map<int, std::tuple<std::string, std::string, std::string, std::string>> dataset_frames = this->interaction_log.get_dataset_frames();
			for (int i = 0; i < topviewkinect::vision::REQUIRED_BACKGROUND_FRAMES; ++i)
			{
				std::string depth_bg_frame_filepath = std::get<0>(dataset_frames.find(i)->second);
				cv::Mat depth_bg_frame = cv::imread(depth_bg_frame_filepath, CV_LOAD_IMAGE_GRAYSCALE);

				std::string infrared_bg_frame_filepath = std::get<1>(dataset_frames.find(i)->second);
				cv::Mat infrared_bg_frame = cv::imread(infrared_bg_frame_filepath, CV_LOAD_IMAGE_GRAYSCALE);
				this->apply_kinect_multisource_frame(i, depth_bg_frame, infrared_bg_frame);
				this->interaction_log.next_frames();
			}

			// Load first frame
			const std::pair<int, std::tuple<std::string, std::string, std::string, std::string>> next_frames = this->interaction_log.current_frames();
			cv::Mat depth_frame = cv::imread(std::get<0>(next_frames.second), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat infrared_frame = cv::imread(std::get<1>(next_frames.second), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat low_infrared_frame = cv::imread(std::get<2>(next_frames.second), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat rgb_frame = cv::imread(std::get<3>(next_frames.second), CV_LOAD_IMAGE_COLOR);

			// Load Android sensor data
			//this->android_sensor_data = this->interaction_log.get_android_sensor_data();

			// BUG: cv::imread no image
			this->apply_kinect_multisource_frame(next_frames.first, depth_frame, infrared_frame, low_infrared_frame, rgb_frame);

			topviewkinect::util::log_println("Dataset (" + std::to_string(dataset_id) + ") Size: " + std::to_string(dataset_frames.size()) + ".");
			return true;
		}

		void TopViewSpace::replay_previous_frame()
		{
			const std::pair<int, std::tuple<std::string, std::string, std::string, std::string>> prev_frames = this->interaction_log.previous_frames();
			cv::Mat depth_frame = cv::imread(std::get<0>(prev_frames.second), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat infrared_frame = cv::imread(std::get<1>(prev_frames.second), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat low_infrared_frame = cv::imread(std::get<2>(prev_frames.second), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat rgb_frame = cv::imread(std::get<3>(prev_frames.second), CV_LOAD_IMAGE_COLOR);

			// Load Android sensor data
			//this->android_sensor_data = this->interaction_log.get_android_sensor_data();

			this->apply_kinect_multisource_frame(prev_frames.first, depth_frame, infrared_frame, low_infrared_frame, rgb_frame);
		}

		void TopViewSpace::replay_next_frame()
		{
			const std::pair<int, std::tuple<std::string, std::string, std::string, std::string>> next_frames = this->interaction_log.next_frames();
			cv::Mat depth_frame = cv::imread(std::get<0>(next_frames.second), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat infrared_frame = cv::imread(std::get<1>(next_frames.second), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat low_infrared_frame = cv::imread(std::get<2>(next_frames.second), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat rgb_frame = cv::imread(std::get<3>(next_frames.second), CV_LOAD_IMAGE_COLOR);

			// Load Android sensor data
			//this->android_sensor_data = this->interaction_log.get_android_sensor_data();

			this->apply_kinect_multisource_frame(next_frames.first, depth_frame, infrared_frame, low_infrared_frame, rgb_frame);
		}

		void TopViewSpace::apply_kinect_multisource_frame(const int frame_id, const cv::Mat& depth_frame, const cv::Mat& infrared_frame, const cv::Mat& infrared_low_frame, const cv::Mat& rgb_frame)
		{
			this->kinect_frame_id = frame_id;
			this->kinect_multisource_frame_arrived = true;
			depth_frame.copyTo(this->depth_frame);
			infrared_frame.copyTo(this->infrared_frame);
			if (!infrared_low_frame.empty())
			{
				infrared_low_frame.copyTo(this->infrared_low_frame);
			}
			if (!rgb_frame.empty())
			{
				rgb_frame.copyTo(this->rgb_frame);
			}

			this->process_kinect_frames();
		}

		void TopViewSpace::apply_android_sensor_data()
		{

		}

		// Capture

		bool TopViewSpace::create_dataset(const int dataset_id)
		{
			topviewkinect::util::log_println("Creating dataset (" + std::to_string(dataset_id) + ")...");

			this->interaction_log.initialize(dataset_id);
			bool dataset_created = this->interaction_log.create_directories();
			if (!dataset_created)
			{
				topviewkinect::util::log_println("Failed to create dataset.");
				return false;
			}

			return true;
		}

		bool TopViewSpace::save_kinect_frames()
		{
			if (!this->kinect_multisource_frame_arrived)
			{
				topviewkinect::util::log_println("No fKinect frames to be saved.");
				return false;
			}
			else
			{
				this->interaction_log.save_multisource_frames(this->kinect_depth_frame_timestamp, this->depth_frame, this->kinect_infrared_frame_timestamp, this->infrared_frame, this->infrared_low_frame, this->kinect_rgb_frame_timestamp, this->rgb_frame);
				return true;
			}
		}

		void TopViewSpace::set_android_sensor_label(const std::string& label)
		{
			this->android_sensor_mutex.lock();
			if (label.compare("device-inward-start") == 0)
			{
				this->android_sensor_label = 1;
			}
			else if (label.compare("device-outward-start") == 0)
			{
				this->android_sensor_label = 2;
			}
			else
			{
				this->android_sensor_label = 0;
			}
			this->android_sensor_mutex.unlock();
		}

		bool TopViewSpace::save_android_sensor_data()
		{
			this->android_sensor_mutex.lock();
			this->interaction_log.save_android_sensor_data(this->kinect_depth_frame_timestamp, android_sensor_data_tmp, this->android_sensor_label);
			this->android_sensor_data_tmp.clear();
			this->android_sensor_mutex.unlock();
			return true;
		}

		bool TopViewSpace::save_visualization()
		{
			this->interaction_log.save_visualization(this->kinect_frame_id, this->visualization_frame);
			return true;
		}

		bool TopViewSpace::save_calibration()
		{
			this->interaction_log.save_calibration(this->crossmotion_calibration_points_2d, this->crossmotion_calibration_points_3d);
			return true;
		}

		void TopViewSpace::postprocess(const std::string& dataset_name, const bool keep_label)
		{
			// Postprocessed data files
			this->interaction_log.create_postprocessed_files(keep_label);

			// Postprocess images
			const std::map<int, std::tuple<std::string, std::string, std::string, std::string>> dataset_frames = this->interaction_log.get_dataset_frames();
			const size_t num_total_frames = dataset_frames.size();
			const int progressbar_step = static_cast<int>(num_total_frames * 0.1);
			int current_frame_idx = 0;
			for (const auto& kv : dataset_frames)
			{
				// Get frames
				std::tuple<std::string, std::string, std::string, std::string> frame_paths = kv.second;
				cv::Mat depth_frame = cv::imread(std::get<0>(frame_paths), CV_LOAD_IMAGE_GRAYSCALE);
				cv::Mat infrared_frame = cv::imread(std::get<1>(frame_paths), CV_LOAD_IMAGE_GRAYSCALE);
				cv::Mat low_infrared_frame = cv::imread(std::get<2>(frame_paths), CV_LOAD_IMAGE_GRAYSCALE);

				this->apply_kinect_multisource_frame(kv.first, depth_frame, infrared_frame, low_infrared_frame); // process
				if (current_frame_idx >= topviewkinect::vision::REQUIRED_BACKGROUND_FRAMES)  // output real data
				{
					this->interaction_log.output_skeleton_features(this->kinect_frame_id, this->skeletons, keep_label);
				}

				// Show progress
				if (++current_frame_idx % progressbar_step == 0)
				{
					int progress = boost::math::iround(static_cast<float>(current_frame_idx) / num_total_frames * 100);
					topviewkinect::util::log_println(std::to_string(this->kinect_frame_id) + "/" + std::to_string(num_total_frames) + " (" + std::to_string(progress) + "%)");
				}
			}

			this->interaction_log.output_description(dataset_name);
		}

		//Detector detector("D:\\libs\\darknet-win\\darknet\\cfg\\yolo.cfg", "D:\\libs\\darknet-win\\darknet\\cfg\\yolo.weights");
		//auto obj_names = objects_names_from_file("D:\\libs\\darknet-win\\darknet\\data\\coco.names");

		// Running
		bool TopViewSpace::process_kinect_frames()
		{
			if (!this->kinect_multisource_frame_arrived)
			{
				return false;
			}

			std::cout << "Frame ID : " << this->kinect_frame_id << std::endl;

			// Per-frame processing time
			std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

			// Calibrating
			if (!this->is_calibration_ready())
			{
				this->current_num_background_frames--;
				this->p_background_extractor->apply(this->depth_frame, this->foreground_mask, -1);

				// Show calibration status
				this->visualization_frame.setTo(topviewkinect::color::CV_BGR_BLACK);
				cv::putText(this->visualization_frame, "Calibrating...", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, topviewkinect::color::CV_BGR_WHITE);
			}
			// Tracking
			else
			{
				// Background subtraction
				this->p_background_extractor->apply(this->depth_frame, this->foreground_mask, 0);
				this->depth_foreground_frame.setTo(topviewkinect::color::CV_BGR_BLACK);
				this->depth_frame.copyTo(this->depth_foreground_frame, this->foreground_mask);
				//cv::imshow("initial depth foreground", this->depth_foreground_frame);
				cv::medianBlur(this->depth_foreground_frame, this->depth_foreground_frame, topviewkinect::vision::FOREGROUND_FILTER_KERNEL);
				cv::Rect border(cv::Point(0, 0), this->depth_foreground_frame.size());
				cv::rectangle(this->depth_foreground_frame, border, topviewkinect::color::CV_BLACK, 5);
				//cv::imshow("medium-blur depth foreground", this->depth_foreground_frame);
				// Filtering time
				auto filtering_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
				std::cout << "filtering time: " << filtering_duration.count() << " msec" << std::endl;

				// Detect skeletons and features

				// Per-skeleton processing time (for evaluation dataset)
				//std::chrono::time_point<std::chrono::high_resolution_clock> features_start = std::chrono::high_resolution_clock::now();
				//long long features_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - features_start).count();

				std::chrono::time_point<std::chrono::high_resolution_clock> computation_start = std::chrono::high_resolution_clock::now();

				this->detect_skeletons();

				auto computation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - computation_start);
				std::cout << "computation time: " << computation_duration.count() << " msec" << std::endl;

				//auto skeleton_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
				//std::cout << "detect skeleton time: " << skeleton_duration.count() << std::endl;

				//// YOLO v2
				////cv::Mat m = cv::Mat::zeros(cv::Size(topviewkinect::kinect2::COLOR_WIDTH/2, topviewkinect::kinect2::COLOR_WIDTH / 2), CV_8UC3);
				////cv::resize(this->rgb_frame, m, m.size());
				//std::vector<bbox_t> result_vec = detector.detect(this->rgb_frame);
				//auto yolo_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
				//std::cout << "yolo time: " << yolo_duration.count() << std::endl;
				//result_vec = detector.tracking_id(result_vec);
				//cv::Mat yolo_result = this->rgb_frame.clone();
				//draw_boxes(yolo_result, result_vec, obj_names);
				//cv::imshow("YOLO output", yolo_result);

				//// OpenPose
				//// Push
				//if (!this->op_processing)
				//{
				//	this->op_processing = true;
				//	// Fill datum
				//	auto datumsPtr = std::make_shared<std::vector<thirdparty::openpose::UserDatum>>();
				//	datumsPtr->emplace_back();
				//	auto& datumToProcess = datumsPtr->at(0);
				//	datumToProcess.cvInputData = this->rgb_frame;
				//	this->op_wrapper.tryEmplace(datumsPtr);
				//}
				//// Pop
				//else
				//{
				//	// Get datum
				//	std::shared_ptr<std::vector<thirdparty::openpose::UserDatum>> datumProcessed;
				//	bool processed = op_wrapper.tryPop(datumProcessed);
				//	if (processed)
				//	{
				//		cv::imshow("OPENPOSE output", datumProcessed->at(0).cvOutputData);
				//		
				//		// Update OpenPose skeletons
				//		this->op_skeletons.clear();
				//		
				//		// Accesing each element of the keypoints
				//		const auto& poseKeypoints = datumProcessed->at(0).poseKeypoints;
				//		op::log("Person pose keypoints:");
				//		for (auto person = 0; person < poseKeypoints.getSize(0); person++)
				//		{
				//			op::log("Person " + std::to_string(person) + " (x, y, score):");
				//			
				//			// add detected skeleton from openpose
				//			thirdparty::openpose::Skeleton new_skeleton;
				//			new_skeleton.joints["Nose"] = { poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["Neck"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["RShoulder"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["RElbow"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["RWrist"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["LShoulder"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["LElbow"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["LWrist"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["RHip"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["RKnee"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["RAnkle"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["LHip"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["LKnee"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["LAnkle"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["REye"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["LEye"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["REar"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			new_skeleton.joints["LEar"] = thirdparty::openpose::Joint{ poseKeypoints[{person, 0, 0}], poseKeypoints[{person, 0, 1}], poseKeypoints[{person, 0, 2}] };
				//			this->op_skeletons.push_back(new_skeleton);

				//			for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
				//			{
				//				std::string valueToPrint;
				//				for (auto xyscore = 0; xyscore < poseKeypoints.getSize(2); xyscore++)
				//					valueToPrint += std::to_string(poseKeypoints[{person, bodyPart, xyscore}]) + " ";
				//				op::log(valueToPrint);
				//			}
				//		}
				//		op::log(" ");
				//		this->op_processing = false;
				//	}
				//}

				// Android sensor data
				//topviewkinect::util::log_println(this->android_sensor_data.to_str());

				//this->combine_openpose_and_depth();

				////this->compute_skeleton_features();

				//// Activity and device recognition
				//if (this->configuration.interaction_recognition)
				//{
				//	this->interaction_classifier.recognize_interactions(this->skeletons);
				//}

				// Adaptive background update/modelling if no person exists
				if (this->skeletons.size() == 0)
				{
					++this->empty_background_counter;
					if (this->empty_background_counter > 300)
					{
						this->p_background_extractor->apply(this->depth_frame, this->foreground_mask, -1);
					}
				}
				//else
				//{
				//    this->empty_background_counter = 0;
				//}

				// Record processing time
				//long long total_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
				//if (this->skeletons.size() > 0 && this->skeletons[0].is_activity_tracked())
				//{
				//    this->interaction_log.output_processing_time(this->kinect_frame_id, features_time, total_time);
				//}

				// Invalidate visualization
				cv::cvtColor(this->depth_frame, this->visualization_frame, CV_GRAY2BGR);
				//this->visualization_frame.setTo(topviewkinect::color::CV_BGR_WHITE);

				// Draw activity tracking zone
				cv::Rect2d activity_zone(topviewkinect::vision::ACTIVITY_ZONE, topviewkinect::vision::ACTIVITY_ZONE, topviewkinect::kinect2::DEPTH_WIDTH - topviewkinect::vision::ACTIVITY_ZONE * 2, topviewkinect::kinect2::DEPTH_HEIGHT - topviewkinect::vision::ACTIVITY_ZONE);
				cv::rectangle(this->visualization_frame, activity_zone, topviewkinect::color::CV_BGR_RED, 1);

				int skeleton_color_idx = 0;
				for (const topviewkinect::skeleton::Skeleton& skeleton : this->skeletons)
				{
					// Skeleton overlay
					cv::Mat skeleton_overlay = this->visualization_frame.clone();
					skeleton_overlay.setTo(topviewkinect::color::CV_BGR_SKELETON[skeleton_color_idx], skeleton.get_mask());
					cv::addWeighted(skeleton_overlay, topviewkinect::color::SKELETON_OVERLAY_ALPHA, this->visualization_frame, 1 - topviewkinect::color::SKELETON_OVERLAY_ALPHA, 0, this->visualization_frame);

					const topviewkinect::skeleton::Joint skeleton_center = skeleton.get_body_center();

					// ID
					//std::ostringstream id_ss;
					//id_ss << skeleton.get_id();
					//cv::putText(this->visualization_frame, id_ss.str(), cv::Point(skeleton_center.x, skeleton_center.y), cv::FONT_HERSHEY_COMPLEX, 1, topviewkinect::color::CV_BGR_WHITE, 2);

					if (skeleton.is_activity_tracked())
					{
						const topviewkinect::skeleton::Joint skeleton_head = skeleton.get_head();
						const topviewkinect::skeleton::Joint skeleton_center = skeleton.get_body_center();

						// Head
						cv::Point head_pt = cv::Point(skeleton_head.x, skeleton_head.y);
						cv::circle(this->visualization_frame, head_pt, 3, topviewkinect::color::CV_BGR_WHITE, -1);

						// Activity and device
						if (this->configuration.interaction_recognition)
						{
							cv::Rect activity_rect = cv::Rect(skeleton_center.x - 70, skeleton_center.y, 160, 50);
							cv::rectangle(this->visualization_frame, activity_rect, topviewkinect::color::CV_BGR_BLACK, -1);
							cv::putText(this->visualization_frame, skeleton.get_activity(), cv::Point(skeleton_center.x - 60, skeleton_center.y + 30), cv::FONT_HERSHEY_COMPLEX, 1, topviewkinect::color::CV_BGR_WHITE, 2);
						}

						// Orientation
						if (this->configuration.orientation_recognition && skeleton.is_activity_tracked())
						{
							int head_angle_pt_x = boost::math::iround(skeleton_head.x + 50 * std::cos(skeleton_head.orientation * CV_PI / 180.0));
							int head_angle_pt_y = boost::math::iround(skeleton_head.y + 50 * std::sin(skeleton_head.orientation * CV_PI / 180.0));
							cv::Point head_angle_pt = cv::Point(head_angle_pt_x, head_angle_pt_y);
							cv::arrowedLine(this->visualization_frame, head_pt, head_angle_pt, topviewkinect::color::CV_BGR_WHITE, 3, 8, 0, 0.5);
						}
					}

					// Update color index
					++skeleton_color_idx;
					if (skeleton_color_idx >= topviewkinect::color::CV_BGR_SKELETON.size())
					{
						skeleton_color_idx = 0;
					}
				}

				int framerate = boost::math::iround(this->framerate_controller.get_fps());

				// Frame rate
				if (this->configuration.framerate)
				{
					//cv::putText(this->visualization_frame, "FPS: " + std::to_string(framerate), cv::Point(topviewkinect::kinect2::DEPTH_WIDTH - 100, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, topviewkinect::color::CV_BGR_WHITE
					//);
				}

				// Send skeleton information to RESTful server
				if (this->configuration.restful_connection && this->restful_client != NULL)
				{
					this->restful_client->request(web::http::methods::POST, "/topviewkinect/skeletons", this->skeletons_json());
				}
			}

			this->kinect_multisource_frame_arrived = false;
			return true;
		}

		const std::array<double, 3> TopViewSpace::relative_3d_position(const topviewkinect::skeleton::Joint& joint1, const topviewkinect::skeleton::Joint& joint2) const
		{
			return std::array<double, 3> {static_cast<double>(joint2.x - joint1.x), static_cast<double>(joint2.y - joint1.y), static_cast<double>(joint2.z - joint1.z) };
		}

		const double TopViewSpace::relative_2d_distance(const topviewkinect::skeleton::Joint& joint1, const topviewkinect::skeleton::Joint& joint2) const
		{
			int distance_vector[2] = { joint2.x - joint1.x, joint2.y - joint1.y };
			return std::sqrt(std::inner_product(std::begin(distance_vector), std::end(distance_vector), std::begin(distance_vector), 0.0));
		}

		void TopViewSpace::find_largest_contour(const cv::Mat& m, std::vector<cv::Point>& largest_contour, double* largest_area) const
		{
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(m.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			if (contours.size() == 0)
			{
				if (largest_area) *largest_area = 0;
				largest_contour = std::vector<cv::Point>{};
				return;
			}

			double maximum_area = 0;
			for (int i = 0; i < contours.size(); ++i)
			{
				double area = cv::contourArea(contours[i]);
				if (area > maximum_area)
				{
					maximum_area = area;
					if (largest_area) *largest_area = maximum_area;
					largest_contour = contours[i];
				}
			}
		}

		void TopViewSpace::find_mass_center(const cv::Mat& src, const std::vector<cv::Point>& contour, topviewkinect::skeleton::Joint& center) const
		{
			cv::Moments contour_moments = cv::moments(contour, true);
			center.x = boost::math::iround(contour_moments.m10 / contour_moments.m00);
			center.y = boost::math::iround(contour_moments.m01 / contour_moments.m00);
			center.z = static_cast<int>(src.at<unsigned char>(center.y, center.x));
		}

		// Test whether the min depth location is inside the tracking zone
		bool TopViewSpace::inside_acitivty_tracking_zone(const cv::Mat& skeleton_depth_frame, int offset_x, int offset_y) const
		{
			cv::Point skeleton_min_loc;
			cv::minMaxLoc(skeleton_depth_frame, 0, 0, &skeleton_min_loc, 0, skeleton_depth_frame <= 50);
			skeleton_min_loc.x += offset_x;
			skeleton_min_loc.y += offset_y;
			if ((skeleton_min_loc.x <= topviewkinect::vision::ACTIVITY_ZONE || skeleton_min_loc.x >= topviewkinect::kinect2::DEPTH_WIDTH - topviewkinect::vision::ACTIVITY_ZONE) || (skeleton_min_loc.y <= topviewkinect::vision::ACTIVITY_ZONE))
			{
				return false;
			}
			return true;
		}

		// Find median value (ignoring zero pixels)
		double median(const cv::Mat& src)
		{
			int hist_size = 256;
			float range[] = { 0, 256 };
			const float* histRange = { range };
			bool uniform = true; bool accumulate = false;
			cv::Mat hist;
			cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &hist_size, &histRange, uniform, accumulate);

			float median_nonzeros = static_cast<float>(cv::countNonZero(src)) / 2;
			float bin = 0;
			for (int i = 1; i < hist_size; ++i)
			{
				bin += hist.at<float>(i);
				if (bin > median_nonzeros)
				{
					return i;
				}
			}

			return 0;
		}

		//int morph_elem = 0;
		//int morph_size = 0;
		//int morph_operator = 0;
		//int const max_operator = 4;
		//int const max_elem = 2;
		//int const max_kernel_size = 21;
		//cv::Mat src;
		//
		//void Morphology_Operations(int, void*)
		//{
		//	int operation = morph_operator + 2;
		//	cv::Mat depth_gradient;
		//	cv::Mat gradient_kernel = cv::getStructuringElement(morph_elem, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
		//	cv::morphologyEx(src, depth_gradient, operation, gradient_kernel);
		//	cv::imshow("gradient", depth_gradient);
		//}

		cv::Mat src, src_gray;
		cv::Mat dst, detected_edges;

		int edgeThresh = 1;
		int lowThreshold;
		int const max_lowThreshold = 100;
		int ratio = 3;
		int kernel_size = 3;
		char* window_name = "Edge Map";
		void CannyThreshold(int, void*)
		{
			// Reduce noise with a kernel 3x3
			//blur(src_gray, detected_edges, cv::Size(3, 3));

			// Canny detector
			Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

			// Using Canny's output as a mask, we display our result
			dst = cv::Scalar::all(0);
			src.copyTo(dst, detected_edges);
			imshow(window_name, dst);
		}

		cv::Mat depth_prev = cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1);
		cv::Mat infrared_prev = cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1);

		inline bool isFlowCorrect(cv::Point2f u)
		{
			return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
		}

		static cv::Vec3b computeColor(float fx, float fy)
		{
			static bool first = true;

			// relative lengths of color transitions:
			// these are chosen based on perceptual similarity
			// (e.g. one can distinguish more shades between red and yellow
			//  than between yellow and green)
			const int RY = 15;
			const int YG = 6;
			const int GC = 4;
			const int CB = 11;
			const int BM = 13;
			const int MR = 6;
			const int NCOLS = RY + YG + GC + CB + BM + MR;
			static cv::Vec3i colorWheel[NCOLS];

			if (first)
			{
				int k = 0;

				for (int i = 0; i < RY; ++i, ++k)
					colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

				for (int i = 0; i < YG; ++i, ++k)
					colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

				for (int i = 0; i < GC; ++i, ++k)
					colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

				for (int i = 0; i < CB; ++i, ++k)
					colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

				for (int i = 0; i < BM; ++i, ++k)
					colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

				for (int i = 0; i < MR; ++i, ++k)
					colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);

				first = false;
			}

			const float rad = sqrt(fx * fx + fy * fy);
			const float a = atan2(-fy, -fx) / (float)CV_PI;

			const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
			const int k0 = static_cast<int>(fk);
			const int k1 = (k0 + 1) % NCOLS;
			const float f = fk - k0;

			cv::Vec3b pix;

			for (int b = 0; b < 3; b++)
			{
				const float col0 = colorWheel[k0][b] / 255.0f;
				const float col1 = colorWheel[k1][b] / 255.0f;

				float col = (1 - f) * col0 + f * col1;

				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius
				else
					col *= .75; // out of range

				pix[2 - b] = static_cast<uchar>(255.0 * col);
			}

			return pix;
		}

		static void drawOpticalFlow(const cv::Mat_<float>& flowx, const  cv::Mat_<float>& flowy, cv::Mat& dst, float maxmotion = -1)
		{
			dst.create(flowx.size(), CV_8UC3);
			dst.setTo(cv::Scalar::all(0));

			// determine motion range:
			float maxrad = maxmotion;

			if (maxmotion <= 0)
			{
				maxrad = 1;
				for (int y = 0; y < flowx.rows; ++y)
				{
					for (int x = 0; x < flowx.cols; ++x)
					{
						cv::Point2f u(flowx(y, x), flowy(y, x));

						if (!isFlowCorrect(u))
							continue;

						maxrad = cv::max(maxrad, cv::sqrt(u.x * u.x + u.y * u.y));
					}
				}
			}

			for (int y = 0; y < flowx.rows; ++y)
			{
				for (int x = 0; x < flowx.cols; ++x)
				{
					cv::Point2f u(flowx(y, x), flowy(y, x));

					if (isFlowCorrect(u))
						dst.at< cv::Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
				}
			}
		}

		static void draw_optflow_arrows(const cv::Mat_<float>& flowx, const  cv::Mat_<float>& flowy, cv::Mat& dst)
		{
			for (int y = 0; y < flowx.rows; y += 8)
			{
				for (int x = 0; x < flowx.cols; x += 8)
				{
					cv::Point2f u(flowx(y, x), flowy(y, x));
					if (cv::sqrt(u.x * u.x + u.y * u.y) < 1)
					{
						continue;
					}
					cv::arrowedLine(dst, cv::Point(x, y), cv::Point(cvRound(x + u.x), cvRound(y + u.y)), topviewkinect::color::CV_BGR_RED, 1);
					cv::circle(dst, cv::Point(x, y), 1, topviewkinect::color::CV_BGR_RED, -1);
				}
			}
		}

		static void draw_optflow_heatmap(const char* name, const cv::cuda::GpuMat& d_flow)
		{
			cv::cuda::GpuMat planes[2];
			cv::cuda::split(d_flow, planes);

			cv::Mat flowx(planes[0]);
			cv::Mat flowy(planes[1]);

			cv::Mat out;
			drawOpticalFlow(flowx, flowy, out);

			cv::imshow(name, out);
		}

		cv::Mat TopViewSpace::compute_optical_flow(const cv::Mat& src_prev, const cv::Mat& src_next, cv::Mat& viz, const char* name)
		{

			cv::cuda::GpuMat d_frame0(src_prev);
			cv::cuda::GpuMat d_frame1(src_next);
			cv::cuda::GpuMat d_flow(src_prev.size(), CV_32FC2);


			// Brox
			{
				cv::cuda::GpuMat d_frame0f;
				cv::cuda::GpuMat d_frame1f;

				d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
				d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

				const int64 brox_start = cv::getTickCount();

				this->brox_optflow->calc(d_frame0f, d_frame1f, d_flow);

				const double brox_duration_sec = (cv::getTickCount() - brox_start) / cv::getTickFrequency();
				std::cout << "brox_optflow time: " << brox_duration_sec * 1000 << " ms" << " " << brox_duration_sec << " s" << std::endl;

				cv::cuda::GpuMat planes[2];
				cv::cuda::split(d_flow, planes);
				cv::Mat flowx(planes[0]);
				cv::Mat flowy(planes[1]);

				draw_optflow_heatmap(name, d_flow);
				draw_optflow_arrows(flowx, flowy, viz);
			}

			//// FARN
			//{
			//	const int64 farn_start = cv::getTickCount();

			//	this->farn_optflow->calc(d_frame0, d_frame1, d_flow);

			//	cv::cuda::GpuMat planes[2];
			//	cv::cuda::split(d_flow, planes);
			//	cv::Mat flowx(planes[0]);
			//	cv::Mat flowy(planes[1]);

			//	draw_optflow_heatmap("farn optflow", d_flow);
			//	draw_optflow_arrows(flowx, flowy, viz);

			//	const double farn_duration_sec = (cv::getTickCount() - farn_start) / cv::getTickFrequency();
			//	std::cout << "farn_optflow time: " << farn_duration_sec * 1000 << " ms" << " " << farn_duration_sec << " s" << std::endl;
			//}


			//// TVL1
			//{
			//	const int64 tvl1_start = cv::getTickCount();

			//	this->tvl1_optflow->calc(d_frame0, d_frame1, d_flow);

			//	cv::cuda::GpuMat planes[2];
			//	cv::cuda::split(d_flow, planes);
			//	cv::Mat flowx(planes[0]);
			//	cv::Mat flowy(planes[1]);

			//	draw_optflow_heatmap("tvl1 optflow", d_flow);
			//	draw_optflow_arrows(flowx, flowy, viz);

			//	const double tvl1_duration_sec = (cv::getTickCount() - tvl1_start) / cv::getTickFrequency();
			//	std::cout << "tvl1_optflow time: " << tvl1_duration_sec * 1000 << " ms" << " " << tvl1_duration_sec << " s" << std::endl;
			//}

			return viz;
		}

		template<class T>
		NCVStatus CopyData(const IplImage *image, const NCVMatrixAlloc<Ncv32f>& dst)
		{
			unsigned char *row = reinterpret_cast<unsigned char*> (image->imageData);
			T convert;
			for (int i = 0; i < image->height; ++i)
			{
				for (int j = 0; j < image->width; ++j)
				{
					if (image->nChannels < 3)
					{
						dst.ptr()[j + i*dst.stride()] = static_cast<float>(*(row + j*image->nChannels)) / 255.0f;
					}
					else
					{
						unsigned char *color = row + j * image->nChannels;
						dst.ptr()[j + i*dst.stride()] = convert(color[0], color[1], color[2]);
					}
				}
				row += image->widthStep;
			}
			return NCV_SUCCESS;
		}

		template<class T>
		NCVStatus CopyData(IplImage *image, cv::Ptr<NCVMatrixAlloc<Ncv32f>>& dst)
		{
			dst = cv::Ptr<NCVMatrixAlloc<Ncv32f>>(new NCVMatrixAlloc<Ncv32f>(*g_pHostMemAllocator, image->width, image->height));
			ncvAssertReturn(dst->isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

			unsigned char *row = reinterpret_cast<unsigned char*> (image->imageData);
			T convert;
			for (int i = 0; i < image->height; ++i)
			{
				for (int j = 0; j < image->width; ++j)
				{
					if (image->nChannels < 3)
					{
						dst->ptr()[j + i*dst->stride()] = static_cast<float> (*(row + j*image->nChannels)) / 255.0f;
					}
					else
					{
						unsigned char *color = row + j * image->nChannels;
						dst->ptr()[j + i*dst->stride()] = convert(color[0], color[1], color[2]);
					}
				}
				row += image->widthStep;
			}
			return NCV_SUCCESS;
		}

		template<typename T>
		inline T Clamp(T x, T a, T b)
		{
			return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
		}

		template<typename T>
		inline T MapValue(T x, T a, T b, T c, T d)
		{
			x = Clamp(x, a, b);
			return c + (d - c) * (x - a) / (b - a);
		}

		static NCVStatus ShowFlow(NCVMatrixAlloc<Ncv32f> &u, NCVMatrixAlloc<Ncv32f> &v, const char *name)
		{
			IplImage *flowField;

			NCVMatrixAlloc<Ncv32f> host_u(*g_pHostMemAllocator, u.width(), u.height());
			ncvAssertReturn(host_u.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

			NCVMatrixAlloc<Ncv32f> host_v(*g_pHostMemAllocator, u.width(), u.height());
			ncvAssertReturn(host_v.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

			ncvAssertReturnNcvStat(u.copySolid(host_u, 0));
			ncvAssertReturnNcvStat(v.copySolid(host_v, 0));

			float *ptr_u = host_u.ptr();
			float *ptr_v = host_v.ptr();

			float maxDisplacement = 1.0f;

			for (Ncv32u i = 0; i < u.height(); ++i)
			{
				for (Ncv32u j = 0; j < u.width(); ++j)
				{
					float d = std::max(fabsf(*ptr_u), fabsf(*ptr_v));
					if (d > maxDisplacement) maxDisplacement = d;
					++ptr_u;
					++ptr_v;
				}
				ptr_u += u.stride() - u.width();
				ptr_v += v.stride() - v.width();
			}

			CvSize image_size = cvSize(u.width(), u.height());
			flowField = cvCreateImage(image_size, IPL_DEPTH_8U, 4);
			if (flowField == 0) return NCV_NULL_PTR;

			unsigned char *row = reinterpret_cast<unsigned char *> (flowField->imageData);

			ptr_u = host_u.ptr();
			ptr_v = host_v.ptr();
			for (int i = 0; i < flowField->height; ++i)
			{
				for (int j = 0; j < flowField->width; ++j)
				{
					(row + j * flowField->nChannels)[0] = 0;
					(row + j * flowField->nChannels)[1] = static_cast<unsigned char> (MapValue(-(*ptr_v), -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
					(row + j * flowField->nChannels)[2] = static_cast<unsigned char> (MapValue(*ptr_u, -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
					(row + j * flowField->nChannels)[3] = 255;
					++ptr_u;
					++ptr_v;
				}
				row += flowField->widthStep;
				ptr_u += u.stride() - u.width();
				ptr_v += v.stride() - v.width();
			}

			cvShowImage(name, flowField);

			return NCV_SUCCESS;
		}

		static IplImage *CreateImage(NCVMatrixAlloc<Ncv32f> &h_r, NCVMatrixAlloc<Ncv32f> &h_g, NCVMatrixAlloc<Ncv32f> &h_b)
		{
			CvSize imageSize = cvSize(h_r.width(), h_r.height());
			IplImage *image = cvCreateImage(imageSize, IPL_DEPTH_8U, 4);
			if (image == 0) return 0;

			unsigned char *row = reinterpret_cast<unsigned char*> (image->imageData);

			for (int i = 0; i < image->height; ++i)
			{
				for (int j = 0; j < image->width; ++j)
				{
					int offset = j * image->nChannels;
					int pos = i * h_r.stride() + j;
					row[offset + 0] = static_cast<unsigned char> (h_b.ptr()[pos] * 255.0f);
					row[offset + 1] = static_cast<unsigned char> (h_g.ptr()[pos] * 255.0f);
					row[offset + 2] = static_cast<unsigned char> (h_r.ptr()[pos] * 255.0f);
					row[offset + 3] = 255;
				}
				row += image->widthStep;
			}
			return image;
		}

		class RgbToMonochrome
		{
		public:
			float operator ()(unsigned char b, unsigned char g, unsigned char r)
			{
				float _r = static_cast<float>(r) / 255.0f;
				float _g = static_cast<float>(g) / 255.0f;
				float _b = static_cast<float>(b) / 255.0f;
				return (_r + _g + _b) / 3.0f;
			}
		};

		class RgbToR
		{
		public:
			float operator ()(unsigned char /*b*/, unsigned char /*g*/, unsigned char r)
			{
				return static_cast<float>(r) / 255.0f;
			}
		};


		class RgbToG
		{
		public:
			float operator ()(unsigned char /*b*/, unsigned char g, unsigned char /*r*/)
			{
				return static_cast<float>(g) / 255.0f;
			}
		};

		class RgbToB
		{
		public:
			float operator ()(unsigned char b, unsigned char /*g*/, unsigned char /*r*/)
			{
				return static_cast<float>(b) / 255.0f;
			}
		};

		static NCVStatus LoadImages(const cv::Mat& frame_0, const cv::Mat& frame_1,
			int& width,
			int& height,
			cv::Ptr<NCVMatrixAlloc<Ncv32f>>& src,
			cv::Ptr<NCVMatrixAlloc<Ncv32f>>& dst,
			IplImage *&firstFrame,
			IplImage *&lastFrame)
		{
			IplImage* image_0 = new IplImage(frame_0);
			if (image_0 == 0)
			{
				std::cout << "Could not open frame 0 '\n";
				return NCV_FILE_ERROR;
			}
			firstFrame = image_0;
			ncvAssertReturnNcvStat(CopyData<RgbToMonochrome>(image_0, src));

			IplImage* image_1 = new IplImage(frame_1);
			if (image_1 == 0)
			{
				std::cout << "Could not open frame 1 '\n";
				return NCV_FILE_ERROR;
			}
			lastFrame = image_1;
			ncvAssertReturnNcvStat(CopyData<RgbToMonochrome>(image_1, dst));

			width = image_0->width;
			height = image_0->height;

			return NCV_SUCCESS;
		}

		int TopViewSpace::detect_skeletons()
		{
			// Invalidate all skeletons
			std::for_each(this->skeletons.begin(), this->skeletons.end(), [](topviewkinect::skeleton::Skeleton& skeleton) { skeleton.set_updated(false); });

			// Laplacian filter
			cv::Mat depth_laplacian, depth_laplacian_filter;
			cv::Laplacian(this->depth_foreground_frame, depth_laplacian, CV_32F, 5, 3);
			cv::convertScaleAbs(depth_laplacian, depth_laplacian_filter);
			cv::Mat depth_laplaced = this->depth_foreground_frame - depth_laplacian_filter;

			cv::imshow("Laplacian filter", depth_laplacian_filter);
			cv::imshow("Depth Laplaced", depth_laplaced);

			// Contour Segmentation
			cv::Mat depth_contours_mask;
			cv::threshold(depth_laplacian_filter, depth_contours_mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			//cv::imshow("Laplacian filter BINARY", depth_contours_mask);

			// Vertical kernel
			//cv::Mat vertical_k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 1));
			//cv::erode(depth_contours_mask, depth_contours_mask, vertical_k, cv::Point(-1, -1));
			//cv::dilate(depth_contours_mask, depth_contours_mask, vertical_k, cv::Point(-1, -1));
			//cv::imshow("Laplacian filter VERTICAL KERNEL", depth_contours_mask);

			////cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
			////cv::morphologyEx(depth_contours, depth_contours_closed, cv::MORPH_CLOSE, gradient_kernel);
			////cv::imshow("depth_contours close", depth_contours_closed);
			////cv::morphologyEx(depth_laplacian_filter, depth_laplacian_filter, cv::MORPH_GRADIENT, kernel, cv::Point(-1, -1), 1);
			////cv::morphologyEx(depth_laplacian_filter, depth_laplacian_filter, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);
			////cv::imshow("depth_laplacian_filter bold", depth_laplacian_filter);

			//cv::Mat test = depth_contours_mask.clone();
			//std::vector<std::vector<cv::Point>> c;
			//std::vector<cv::Vec4i> h;
			//cv::findContours(depth_contours_mask.clone(), c, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			//for (int i = 0; i < c.size(); ++i)
			//{
			//	if (cv::contourArea(c[i], false) >= 500)
			//	{
			//		if (h[i][3] != -1)
			//		{
			//			cv::drawContours(depth_contours_mask, c, i, 255, 1);
			//		}
			//		//if (h[i][3] != -1)
			//		//{
			//		//	cv::drawContours(test, c, i, topviewkinect::color::CV_BLACK, -1);
			//		//}

			//		// has child continue
			//		//if (depth_skeletons_hierarrchy[i][2] != -1)
			//		//{
			//		//	std::cout << "has child" << std::endl;
			//		//	continue;
			//		//}
			//		//cv::drawContours(depth_skeletons, depth_skeletons_contours, i, topviewkinect::color::CV_WHITE, 2);
			//	}
			//}
			//cv::imshow("test", depth_contours_mask);

			// Dense optical flow
			if (cv::countNonZero(depth_prev) == 0 || cv::countNonZero(infrared_prev) == 0)
			{
				depth_laplaced.copyTo(depth_prev);
				this->infrared_frame.copyTo(infrared_prev);
			}
			else
			{
				//cv::Mat optical = this->depth_foreground_frame.clone();
				//depth_laplaced.copyTo(next);
				//cv::calcOpticalFlowFarneback(depth_prev, next, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
				//cv::Mat depth_optical_viz = this->depth_foreground_frame.clone();
				//compute_optical_flow(depth_prev, depth_laplaced, depth_optical_viz, "Brox Optical Flow - Depth");
				//cv::imshow("dense optical depth", depth_optical_viz);

				cv::Mat infrared_optical_viz;
				cv::cvtColor(this->infrared_frame, infrared_optical_viz, CV_GRAY2BGR);

				compute_optical_flow(infrared_prev, this->infrared_frame.clone(), infrared_optical_viz, "Brox Optical Flow - Infrared");
				cv::imshow("Brox Optical Flow (infrared)", infrared_optical_viz);

				// Liu Optflow
				{
					// get the parameters
					//double alpha= 1;
					//double ratio=0.5;
					//int minWidth= 40;
					//int nOuterFPIterations = 3;
					//int nInnerFPIterations = 1;
					//int nSORIterations= 20;
					//
					//DImage Im1, Im2;
					//Im1.imread_opencv_gray(infrared_prev);
					//Im2.imread_opencv_gray(this->infrared_frame);
					//
					//DImage vx,vy,warpI2;

					//const int64 liu_start_sec = cv::getTickCount();
					//
					//OpticalFlow::Coarse2FineFlow(vx,vy,warpI2,Im1,Im2,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations);

					//const double liu_duration_sec = (cv::getTickCount() - liu_start_sec) / cv::getTickFrequency();
					//std::cout << "liu_optflow time: " << liu_duration_sec * 1000 << " ms" << " " << liu_duration_sec << " s" << std::endl;
				}

				// NVIDIA GPU OPTICAL FLOW
				{
					//Ncv32f timeStep = 0.25f;
					NCVBroxOpticalFlowDescriptor desc;
					desc.alpha = 0.197f;
					desc.gamma = 50.0f;
					desc.scale_factor = 0.8f;
					desc.number_of_inner_iterations = 5;
					desc.number_of_outer_iterations = 77;
					desc.number_of_solver_iterations = 10;

					int width, height;
					cv::Ptr<NCVMatrixAlloc<Ncv32f>> src_host;
					cv::Ptr<NCVMatrixAlloc<Ncv32f>> dst_host;
					IplImage *firstFrame, *lastFrame;

					// NV timer
					std::chrono::time_point<std::chrono::high_resolution_clock> nv_start = std::chrono::high_resolution_clock::now();
					const int64 nv_start_sec = cv::getTickCount();

					ncvAssertReturnNcvStat(LoadImages(infrared_prev, this->infrared_frame, width, height, src_host, dst_host, firstFrame, lastFrame));

					cv::Ptr<NCVMatrixAlloc<Ncv32f>> src(new NCVMatrixAlloc<Ncv32f>(*g_pGPUMemAllocator, src_host->width(), src_host->height()));
					ncvAssertReturn(src->isMemAllocated(), -1);

					cv::Ptr<NCVMatrixAlloc<Ncv32f>> dst(new NCVMatrixAlloc<Ncv32f>(*g_pGPUMemAllocator, src_host->width(), src_host->height()));
					ncvAssertReturn(dst->isMemAllocated(), -1);

					ncvAssertReturnNcvStat(src_host->copySolid(*src, 0));
					ncvAssertReturnNcvStat(dst_host->copySolid(*dst, 0));

					double duration = (cv::getTickCount() - nv_start_sec) / cv::getTickFrequency();
					std::cout << "nv_brox_optflow load images : " << duration << " sec" << std::endl;

#if defined SAFE_MAT_DECL
#undef SAFE_MAT_DECL
#endif
#define SAFE_MAT_DECL(name, allocator, sx, sy) \
				NCVMatrixAlloc<Ncv32f> name(*allocator, sx, sy);\
				ncvAssertReturn(name.isMemAllocated(), -1);

					SAFE_MAT_DECL(u, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(v, g_pGPUMemAllocator, width, height);

					SAFE_MAT_DECL(uBck, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(vBck, g_pGPUMemAllocator, width, height);

					SAFE_MAT_DECL(h_r, g_pHostMemAllocator, width, height);
					SAFE_MAT_DECL(h_g, g_pHostMemAllocator, width, height);
					SAFE_MAT_DECL(h_b, g_pHostMemAllocator, width, height);

					duration = (cv::getTickCount() - nv_start_sec) / cv::getTickFrequency();
					std::cout << "nv_brox_optflow forward backward before : " << duration << " sec" << std::endl;

					//std::cout << "Estimating optical flow\nForward...\n";
					if (NCV_SUCCESS != NCVBroxOpticalFlow(desc, *g_pGPUMemAllocator, *src, *dst, u, v, 0))
					{
						std::cout << "Failed\n";
						return -1;
					}

					//std::cout << "Backward...\n";
					if (NCV_SUCCESS != NCVBroxOpticalFlow(desc, *g_pGPUMemAllocator, *dst, *src, uBck, vBck, 0))
					{
						std::cout << "Failed\n";
						return -1;
					}

					duration = (cv::getTickCount() - nv_start_sec) / cv::getTickFrequency();
					std::cout << "nv_brox_optflow forward backward : " << duration << " sec" << std::endl;

					// matrix for temporary data
					SAFE_MAT_DECL(d_temp, g_pGPUMemAllocator, width, height);

					// first frame color components (GPU memory)
					SAFE_MAT_DECL(d_r, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(d_g, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(d_b, g_pGPUMemAllocator, width, height);

					// second frame color components (GPU memory)
					SAFE_MAT_DECL(d_rt, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(d_gt, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(d_bt, g_pGPUMemAllocator, width, height);

					// intermediate frame color components (GPU memory)
					SAFE_MAT_DECL(d_rNew, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(d_gNew, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(d_bNew, g_pGPUMemAllocator, width, height);

					// interpolated forward flow
					SAFE_MAT_DECL(ui, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(vi, g_pGPUMemAllocator, width, height);

					// interpolated backward flow
					SAFE_MAT_DECL(ubi, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(vbi, g_pGPUMemAllocator, width, height);

					// occlusion masks
					SAFE_MAT_DECL(occ0, g_pGPUMemAllocator, width, height);
					SAFE_MAT_DECL(occ1, g_pGPUMemAllocator, width, height);

					// prepare color components on host and copy them to device memory
					ncvAssertReturnNcvStat(CopyData<RgbToR>(firstFrame, h_r));
					ncvAssertReturnNcvStat(CopyData<RgbToG>(firstFrame, h_g));
					ncvAssertReturnNcvStat(CopyData<RgbToB>(firstFrame, h_b));

					ncvAssertReturnNcvStat(h_r.copySolid(d_r, 0));
					ncvAssertReturnNcvStat(h_g.copySolid(d_g, 0));
					ncvAssertReturnNcvStat(h_b.copySolid(d_b, 0));

					ncvAssertReturnNcvStat(CopyData<RgbToR>(lastFrame, h_r));
					ncvAssertReturnNcvStat(CopyData<RgbToG>(lastFrame, h_g));
					ncvAssertReturnNcvStat(CopyData<RgbToB>(lastFrame, h_b));

					ncvAssertReturnNcvStat(h_r.copySolid(d_rt, 0));
					ncvAssertReturnNcvStat(h_g.copySolid(d_gt, 0));
					ncvAssertReturnNcvStat(h_b.copySolid(d_bt, 0));

					duration = (cv::getTickCount() - nv_start_sec) / cv::getTickFrequency();
					std::cout << "nv_brox_optflow done: " << duration << " sec" << std::endl;

					auto nv_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - nv_start);
					std::cout << "nv_optflow time: " << nv_duration.count() << " msec" << std::endl;
					const double nv_time_sec = (cv::getTickCount() - nv_start_sec) / cv::getTickFrequency();
					std::cout << "nv_brox_optflow time : " << nv_time_sec << " sec" << std::endl;

					ShowFlow(u, v, "Forward flow");
					ShowFlow(uBck, vBck, "Backward flow");
				}


				//std::cout << "Interpolating...\n";
				//std::cout.precision(4);

				//std::vector<IplImage*> frames;
				//frames.push_back(firstFrame);

				// compute interpolated frames
				//for (Ncv32f timePos = timeStep; timePos < 1.0f; timePos += timeStep)
				//{
				//	ncvAssertCUDAReturn(cudaMemset(ui.ptr(), 0, ui.pitch() * ui.height()), NCV_CUDA_ERROR);
				//	ncvAssertCUDAReturn(cudaMemset(vi.ptr(), 0, vi.pitch() * vi.height()), NCV_CUDA_ERROR);

				//	ncvAssertCUDAReturn(cudaMemset(ubi.ptr(), 0, ubi.pitch() * ubi.height()), NCV_CUDA_ERROR);
				//	ncvAssertCUDAReturn(cudaMemset(vbi.ptr(), 0, vbi.pitch() * vbi.height()), NCV_CUDA_ERROR);

				//	ncvAssertCUDAReturn(cudaMemset(occ0.ptr(), 0, occ0.pitch() * occ0.height()), NCV_CUDA_ERROR);
				//	ncvAssertCUDAReturn(cudaMemset(occ1.ptr(), 0, occ1.pitch() * occ1.height()), NCV_CUDA_ERROR);

				//	NppStInterpolationState state;
				//	// interpolation state should be filled once except pSrcFrame0, pSrcFrame1, and pNewFrame
				//	// we will only need to reset buffers content to 0 since interpolator doesn't do this itself
				//	state.size = NcvSize32u(width, height);
				//	state.nStep = d_r.pitch();
				//	state.pSrcFrame0 = d_r.ptr();
				//	state.pSrcFrame1 = d_rt.ptr();
				//	state.pFU = u.ptr();
				//	state.pFV = v.ptr();
				//	state.pBU = uBck.ptr();
				//	state.pBV = vBck.ptr();
				//	state.pos = timePos;
				//	state.pNewFrame = d_rNew.ptr();
				//	state.ppBuffers[0] = occ0.ptr();
				//	state.ppBuffers[1] = occ1.ptr();
				//	state.ppBuffers[2] = ui.ptr();
				//	state.ppBuffers[3] = vi.ptr();
				//	state.ppBuffers[4] = ubi.ptr();
				//	state.ppBuffers[5] = vbi.ptr();

				//	// interpolate red channel
				//	nppiStInterpolateFrames(&state);

				//	// reset buffers
				//	ncvAssertCUDAReturn(cudaMemset(ui.ptr(), 0, ui.pitch() * ui.height()), NCV_CUDA_ERROR);
				//	ncvAssertCUDAReturn(cudaMemset(vi.ptr(), 0, vi.pitch() * vi.height()), NCV_CUDA_ERROR);

				//	ncvAssertCUDAReturn(cudaMemset(ubi.ptr(), 0, ubi.pitch() * ubi.height()), NCV_CUDA_ERROR);
				//	ncvAssertCUDAReturn(cudaMemset(vbi.ptr(), 0, vbi.pitch() * vbi.height()), NCV_CUDA_ERROR);

				//	ncvAssertCUDAReturn(cudaMemset(occ0.ptr(), 0, occ0.pitch() * occ0.height()), NCV_CUDA_ERROR);
				//	ncvAssertCUDAReturn(cudaMemset(occ1.ptr(), 0, occ1.pitch() * occ1.height()), NCV_CUDA_ERROR);

				//	// interpolate green channel
				//	state.pSrcFrame0 = d_g.ptr();
				//	state.pSrcFrame1 = d_gt.ptr();
				//	state.pNewFrame = d_gNew.ptr();

				//	nppiStInterpolateFrames(&state);

				//	// reset buffers
				//	ncvAssertCUDAReturn(cudaMemset(ui.ptr(), 0, ui.pitch() * ui.height()), NCV_CUDA_ERROR);
				//	ncvAssertCUDAReturn(cudaMemset(vi.ptr(), 0, vi.pitch() * vi.height()), NCV_CUDA_ERROR);

				//	ncvAssertCUDAReturn(cudaMemset(ubi.ptr(), 0, ubi.pitch() * ubi.height()), NCV_CUDA_ERROR);
				//	ncvAssertCUDAReturn(cudaMemset(vbi.ptr(), 0, vbi.pitch() * vbi.height()), NCV_CUDA_ERROR);

				//	ncvAssertCUDAReturn(cudaMemset(occ0.ptr(), 0, occ0.pitch() * occ0.height()), NCV_CUDA_ERROR);
				//	ncvAssertCUDAReturn(cudaMemset(occ1.ptr(), 0, occ1.pitch() * occ1.height()), NCV_CUDA_ERROR);

				//	// interpolate blue channel
				//	state.pSrcFrame0 = d_b.ptr();
				//	state.pSrcFrame1 = d_bt.ptr();
				//	state.pNewFrame = d_bNew.ptr();

				//	nppiStInterpolateFrames(&state);

				//	// copy to host memory
				//	ncvAssertReturnNcvStat(d_rNew.copySolid(h_r, 0));
				//	ncvAssertReturnNcvStat(d_gNew.copySolid(h_g, 0));
				//	ncvAssertReturnNcvStat(d_bNew.copySolid(h_b, 0));

				//	// convert to IplImage
				//	IplImage *newFrame = CreateImage(h_r, h_g, h_b);
				//	if (newFrame == 0)
				//	{
				//		std::cout << "Could not create new frame in host memory\n";
				//		break;
				//	}
				//	frames.push_back(newFrame);
				//	//std::cout << timePos * 100.0f << "%\r";
				//}
				////std::cout << std::setw(5) << "100%\n";

				//frames.push_back(lastFrame);

				//Ncv32u currentFrame;
				//currentFrame = 0;
				//cvShowImage("Interpolated frame", frames[0]);
				//cvShowImage("Interpolated frame 2", frames[1]);

				//cv::Mat flow_split[2];
				//cv::Mat magnitude, angle;
				//cv::split(flow, flow_split);
				//cv::cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);

				//int rows = flow.rows;
				//int cols = flow.cols;
				//CFloatImage cFlow(cols, rows, 2);
				//// Convert flow to CFLoatImage:
				//for (int i = 0; i < rows; i++) {
				//	for (int j = 0; j < cols; j++) {
				//		cFlow.Pixel(j, i, 0) = flow.at<cv::Vec2f>(i, j)[0];
				//		cFlow.Pixel(j, i, 1) = flow.at<cv::Vec2f>(i, j)[1];
				//	}
				//}
				//CByteImage cImage;
				//double mag_max;
				//cv::minMaxLoc(magnitude, 0, &mag_max);
				//MotionToColor(cFlow, cImage, mag_max);
				//cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
				//// Compute back to cv::Mat with 3 channels in BGR:
				//for (int i = 0; i < rows; i++) {
				//	for (int j = 0; j < cols; j++) {
				//		image.at<cv::Vec3b>(i, j)[0] = cImage.Pixel(j, i, 0);
				//		image.at<cv::Vec3b>(i, j)[1] = cImage.Pixel(j, i, 1);
				//		image.at<cv::Vec3b>(i, j)[2] = cImage.Pixel(j, i, 2);
				//	}
				//}
				//cv::imshow("optical color", image);

				//cv::Mat flow_split[2];
				//cv::Mat hsv_split[3], hsv, bgr;
				//cv::Mat magnitude, angle;
				//cv::split(flow, flow_split);
				//cv::cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
				//cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);
				//hsv_split[0] = angle; // already in degrees - no normalization needed
				//hsv_split[1] = 255;
				//hsv_split[2] = cv::Mat::ones(angle.size(), angle.type());
				//cv::merge(hsv_split, 3, hsv);
				//cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
				//cv::imshow("optical color", bgr);

				//cv::cartToPolar(res.u, res.t, )
				depth_laplaced.copyTo(depth_prev);
				this->infrared_frame.copyTo(infrared_prev);
			}

			//cv::Mat depth_contours_closed;
			//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
			//cv::morphologyEx(depth_contours, depth_contours_closed, cv::MORPH_CLOSE, gradient_kernel);
			//cv::imshow("depth_contours close", depth_contours_closed);
			//cv::morphologyEx(depth_contours, depth_contours, cv::MORPH_GRADIENT, kernel);
			//cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
			//cv::morphologyEx(depth_contours, depth_contours, cv::MORPH_DILATE, kernel);
			//cv::imshow("dilate", depth_contours);

			cv::Mat new_d = this->depth_foreground_frame.clone();
			new_d.setTo(topviewkinect::color::CV_BLACK, depth_contours_mask);
			//cv::imshow("new d", new_d);

			cv::Mat depth_skeletons_all = this->depth_foreground_frame.clone();
			depth_skeletons_all.setTo(0);

			std::vector<std::vector<cv::Point>> depth_skeletons_contours;
			std::vector<cv::Vec4i> depth_skeletons_hierarrchy;
			cv::findContours(new_d.clone(), depth_skeletons_contours, depth_skeletons_hierarrchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			for (int i = 0; i < depth_skeletons_contours.size(); ++i)
			{
				if (cv::contourArea(depth_skeletons_contours[i], false) >= 500)
				{
					cv::drawContours(depth_skeletons_all, depth_skeletons_contours, i, topviewkinect::color::CV_WHITE, 1);

					// has child continue
					//if (depth_skeletons_hierarrchy[i][2] != -1)
					//{
					//	std::cout << "has child" << std::endl;
					//	continue;
					//}
					//cv::drawContours(depth_skeletons, depth_skeletons_contours, i, topviewkinect::color::CV_WHITE, 2);
				}
			}
			//cv::imshow("depth skeletons all", depth_skeletons_all);

			//cv::Mat depth_contour_clean = this->depth_foreground_frame.clone();
			//std::vector<std::vector<cv::Point>> c2;
			//cv::findContours(depth_laplaced.clone(), c2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			//for (int i = 0; i < c2.size(); ++i)
			//{
			//	cv::drawContours(depth_contour_clean, c2, i, topviewkinect::color::CV_WHITE, 2);
			//}
			//cv::imshow("depth contours final", depth_contour_clean);


			// Filter

			//cv::Mat depth_bw;
			//cv::threshold(depth_laplacian_filter, depth_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			//cv::imshow("bw", depth_bw);

			//cv::Mat depth_bw2;
			//cv::threshold(depth_laplaced, depth_bw2, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			//cv::imshow("depth_bw2", depth_bw2);

			//cv::Mat depth_gradient;
			//cv::Mat gradient_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * 2 + 1, 2 * 2 + 1), cv::Point(2, 2));
			//cv::morphologyEx(depth_bw, depth_gradient, cv::MORPH_CLOSE, gradient_kernel);
			//cv::imshow("depth_gradient", depth_gradient);

			//depth_laplaced.copyTo(src);
			//depth_laplaced.copyTo(detected_edges);
			//cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
			///// Create a Trackbar for user to enter threshold
			//cv::createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
			///// Show the image
			//CannyThreshold(0, 0);

			////
			//depth_bw.copyTo(src);
			//cv::createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat",
			//	"gradient", &morph_operator, max_operator,
			//	Morphology_Operations);
			//cv::createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", "gradient",
			//	&morph_elem, max_elem,
			//	Morphology_Operations);
			//cv::createTrackbar("Kernel size:\n 2n +1", "gradient",
			//	&morph_size, max_kernel_size,
			//	Morphology_Operations);

			//cv::Mat depth_gradient;
			//cv::Mat gradient_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * 3 + 1, 2 * 3 + 1));
			//cv::morphologyEx(depth_laplaced, depth_gradient, cv::MORPH_GRADIENT, gradient_kernel);
			//cv::imshow("gradient", depth_gradient);

			//// Distance Transform
			//cv::Mat depth_bw;
			//cv::threshold(depth_laplaced, depth_bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			//cv::imshow("bw", depth_bw);
			//cv::Mat depth_dist;
			//cv::distanceTransform(depth_bw, depth_dist, CV_DIST_L2, 5);
			//cv::normalize(depth_dist, depth_dist, 0, 1, cv::NORM_MINMAX);
			//cv::imshow("distance transform", depth_dist);

			//// Dialation
			//cv::threshold(depth_dist, depth_dist, 0.4, 1, CV_THRESH_BINARY);
			////cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8UC1);
			////cv::dilate(depth_dist, depth_dist, kernel1);
			//cv::imshow("peaks", depth_dist);

			// conncected components
			//cv::Mat depth_laplaced_contours = this->depth_foreground_frame.clone();
			//std::vector<std::vector<cv::Point>> c;
			//cv::findContours(depth_laplaced.clone(), c, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
			//for (int i = 0; i < c.size(); ++i)
			//{
			//	cv::drawContours(depth_laplaced_contours, c, i, topviewkinect::color::CV_WHITE, 1);
			//}
			//cv::imshow("connected components", depth_laplaced_contours);


			// Find new skeletons (contours w/ sufficient size)
			std::vector<std::tuple<int, topviewkinect::skeleton::Joint>> new_skeleton_contours; // <contour index, contour center>

			depth_skeletons_contours.clear();
			cv::findContours(new_d.clone(), depth_skeletons_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			for (int i = 0; i < depth_skeletons_contours.size(); ++i)
			{
				if (cv::contourArea(depth_skeletons_contours[i], false) >= 500)
				{
					cv::Moments skeleton_moments = cv::moments(depth_skeletons_contours[i], true);
					topviewkinect::skeleton::Joint skeleton_center = {
						boost::math::iround(skeleton_moments.m10 / skeleton_moments.m00),
						boost::math::iround(skeleton_moments.m01 / skeleton_moments.m00) };
					new_skeleton_contours.push_back(std::make_tuple(i, skeleton_center));
				}
			}

			int next_skeleton_id = 0;
			for (int i = 0; i < this->skeletons.size(); ++i)
			{
				int skeleton_id = this->skeletons[i].get_id();
				if (skeleton_id >= next_skeleton_id)
				{
					next_skeleton_id = skeleton_id + 1;
				}
			}

			// Match skeletons based on center of mass

			std::vector<std::tuple<int, int>> skeleton_matches; // skeleton <skeleton id, contour index>
			if (this->skeletons.size() <= new_skeleton_contours.size()) // more new skeletons
			{
				// Match current skeletons
				for (topviewkinect::skeleton::Skeleton& skeleton : this->skeletons)
				{
					const topviewkinect::skeleton::Joint skeleton_center = skeleton.get_body_center();

					double shortest_distance = DBL_MAX;
					int shortest_distance_skeleton_idx = -1;
					int shortest_distance_skeleton_contour_idx = -1;

					for (int new_skeleton_idx = 0; new_skeleton_idx < new_skeleton_contours.size(); ++new_skeleton_idx)
					{
						std::tuple<int, topviewkinect::skeleton::Joint> new_skeleton_tuple = new_skeleton_contours[new_skeleton_idx];
						int new_skeleton_contour_idx = std::get<0>(new_skeleton_tuple);
						topviewkinect::skeleton::Joint new_skeleton_center = std::get<1>(new_skeleton_tuple);

						double skeleton_distance = this->relative_2d_distance(skeleton_center, new_skeleton_center);
						if (skeleton_distance <= shortest_distance)
						{
							shortest_distance = skeleton_distance;
							shortest_distance_skeleton_idx = new_skeleton_idx;
							shortest_distance_skeleton_contour_idx = new_skeleton_contour_idx;
						}
					}

					skeleton.set_body_center(std::get<1>(new_skeleton_contours[shortest_distance_skeleton_idx]));
					skeleton.set_updated(true);
					skeleton_matches.push_back(std::make_tuple(skeleton.get_id(), shortest_distance_skeleton_contour_idx));
					new_skeleton_contours.erase(new_skeleton_contours.begin() + shortest_distance_skeleton_idx);
				}

				// Create new skeleton instances
				for (auto new_skeleton_tuple : new_skeleton_contours)
				{
					skeleton_matches.push_back(std::make_tuple(next_skeleton_id, std::get<0>(new_skeleton_tuple)));
					topviewkinect::skeleton::Skeleton new_skeleton(next_skeleton_id);
					new_skeleton.set_body_center(std::get<1>(new_skeleton_tuple));
					new_skeleton.set_updated(true);
					this->skeletons.push_back(new_skeleton);
					++next_skeleton_id;
				}
			}
			else // less new skeletons
			{
				for (auto new_skeleton_contour_tuple : new_skeleton_contours)
				{
					int new_skeleton_contour_idx = std::get<0>(new_skeleton_contour_tuple);
					topviewkinect::skeleton::Joint new_skeleton_center = std::get<1>(new_skeleton_contour_tuple);

					double shortest_distance = DBL_MAX;
					int shortest_distance_skeleton_idx = -1;
					for (int i = 0; i < this->skeletons.size(); ++i)
					{
						if (this->skeletons[i].is_updated())
						{
							continue;
						}

						const topviewkinect::skeleton::Joint skeleton_center = this->skeletons[i].get_body_center();
						double skeleton_distance = this->relative_2d_distance(skeleton_center, new_skeleton_center);
						if (skeleton_distance <= shortest_distance)
						{
							shortest_distance = skeleton_distance;
							shortest_distance_skeleton_idx = i;
						}
					}

					skeleton_matches.push_back(std::make_tuple(this->skeletons[shortest_distance_skeleton_idx].get_id(), new_skeleton_contour_idx));
					this->skeletons[shortest_distance_skeleton_idx].set_body_center(new_skeleton_center);
					this->skeletons[shortest_distance_skeleton_idx].set_updated(true);
				}
			}

			// Remove skeletons who exited (not updated)
			this->skeletons.erase(std::remove_if(this->skeletons.begin(), this->skeletons.end(), [](const topviewkinect::skeleton::Skeleton& skeleton) { return !skeleton.is_updated(); }), this->skeletons.end());

			// Skeleton body segmentation
			for (topviewkinect::skeleton::Skeleton& skeleton : this->skeletons)
			{
				auto it = std::find_if(skeleton_matches.begin(), skeleton_matches.end(), [&](const std::tuple<int, int>& e) {return std::get<0>(e) == skeleton.get_id(); });
				int new_skeleton_contour_idx = std::get<1>(*it);

				std::vector<cv::Point> skeleton_contour = depth_skeletons_contours[new_skeleton_contour_idx];
				skeleton.set_contour(skeleton_contour);

				cv::RotatedRect skeleton_rect = cv::minAreaRect(cv::Mat(skeleton_contour));
				skeleton.width = static_cast<double>(skeleton_rect.size.width);
				skeleton.height = static_cast<double>(skeleton_rect.size.height);

				// Depth map
				cv::Mat skeleton_depth_frame = cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1);
				cv::Mat skeleton_occupancy_mask = cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1);
				cv::drawContours(skeleton_occupancy_mask, depth_skeletons_contours, new_skeleton_contour_idx, topviewkinect::color::CV_WHITE, -1);
				this->depth_foreground_frame.copyTo(skeleton_depth_frame, skeleton_occupancy_mask); // Get skeleton depth frame
				cv::threshold(skeleton_depth_frame, skeleton_occupancy_mask, 0, 255, cv::THRESH_BINARY); // Recover skeleton contour gaps
				skeleton.set_mask(skeleton_occupancy_mask); // Update skeleton mask

				// Infrared map
				cv::Mat skeleton_infrared_frame = cv::Mat::zeros(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE, CV_8UC1);
				this->infrared_low_frame.copyTo(skeleton_infrared_frame, skeleton_occupancy_mask);

				// Depth silhouette
				cv::Rect skeleton_bounding_rect = cv::boundingRect(skeleton_contour);
				cv::Mat skeleton_depth_silhouette = skeleton_depth_frame(skeleton_bounding_rect);
				skeleton_depth_silhouette.setTo(0, skeleton_depth_silhouette == 255);
				cv::medianBlur(skeleton_depth_silhouette, skeleton_depth_silhouette, 3);

				if (!inside_acitivty_tracking_zone(skeleton_depth_silhouette, skeleton_bounding_rect.x, skeleton_bounding_rect.y))
				{
					skeleton.set_activity_tracked(false);
					continue;
				}

				// Depth silhouette downsampled (32 * 32)
				cv::Mat skeleton_depth_silhouette_downsampled = cv::Mat::zeros(cv::Size(32, 32), CV_8UC1);
				cv::resize(skeleton_depth_silhouette, skeleton_depth_silhouette_downsampled, skeleton_depth_silhouette_downsampled.size());

				// Infrared silhouette
				cv::Mat skeleton_infrared_silhouette = skeleton_infrared_frame.clone();
				skeleton_infrared_silhouette = skeleton_infrared_silhouette(skeleton_bounding_rect);
				skeleton_infrared_silhouette.setTo(0, skeleton_infrared_silhouette == 255);
				cv::threshold(skeleton_infrared_silhouette, skeleton_infrared_silhouette, 0, 255, CV_THRESH_BINARY);

				// Infrared silhouette downsampled (32 * 32)
				cv::Mat skeleton_infrared_silhouette_downsampled = cv::Mat::zeros(cv::Size(32, 32), CV_8UC1);
				cv::resize(skeleton_infrared_silhouette, skeleton_infrared_silhouette_downsampled, skeleton_infrared_silhouette_downsampled.size());

				// Figures
				//cv::imshow("skeleton depth sil", skeleton_depth_silhouette);
				//cv::Mat skeleton_depth_silhouette_downsampled = cv::Mat::zeros(cv::Size(128, 128), CV_8UC1);
				//cv::resize(skeleton_depth_silhouette, skeleton_depth_silhouette_downsampled, skeleton_depth_silhouette_downsampled.size());
				//cv::Mat skeleton_infrared_silhouette_downsampled = cv::Mat::zeros(cv::Size(128, 128), CV_8UC1);
				//cv::resize(skeleton_infrared_silhouette, skeleton_infrared_silhouette_downsampled, skeleton_infrared_silhouette_downsampled.size());

				// K-means depth clustering of body layers
				const int attempts = 5;
				const double eps = 0.000001;
				cv::Mat1f data;
				skeleton_depth_silhouette_downsampled.convertTo(data, CV_32F);
				data = data.reshape(1, 1).t();
				cv::Mat1i best_labels(data.size(), 0);
				cv::Mat1f centers;
				// 3+2 k-means layers (Kinect depth noise + segmentation noise)
				cv::kmeans(data, topviewkinect::vision::BODY_MAX_NUM_LAYERS + 2, best_labels, cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, attempts, eps), attempts, cv::KMEANS_PP_CENTERS, centers);
				cv::Mat1b skeleton_depth_layers(skeleton_depth_silhouette_downsampled.rows, skeleton_depth_silhouette_downsampled.cols);
				for (int r = 0; r < skeleton_depth_layers.rows; ++r)
				{
					for (int c = 0; c < skeleton_depth_layers.cols; ++c)
					{
						skeleton_depth_layers(r, c) = static_cast<uchar>(centers(best_labels(r*skeleton_depth_silhouette_downsampled.cols + c)));
					}
				}

				// Sort K-means values
				std::array<int, topviewkinect::vision::BODY_MAX_NUM_LAYERS + 2> kmeans_values{};
				for (int i = 0; i < kmeans_values.size(); i++)
				{
					kmeans_values[i] = static_cast<int>(std::floor(centers.at<float>(0, i)));
				}
				std::sort(kmeans_values.begin(), kmeans_values.end());

				cv::Mat skeleton_depth_body_layers_color = draw_color_body_layers(skeleton_depth_layers, best_labels, centers);

				// Find head position and orientation
				std::vector<cv::Point> skeleton_depth_contour;
				std::vector<cv::Point> skeleton_head_contour;
				std::vector<cv::Point> skeleton_body_contour;
				std::vector<cv::Point> skeleton_bottom_contour;
				std::vector<cv::Point> skeleton_others_contour;
				cv::Mat skeleton_head_layer = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);
				cv::Mat skeleton_body_layer = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);
				cv::Mat skeleton_bottom_layer = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);
				cv::Mat skeleton_others_layer = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);

				for (int kth_idx = 1; kth_idx < kmeans_values.size(); ++kth_idx)
				{
					int kth_value = kmeans_values[kth_idx];

					cv::Mat skeleton_kth_layer = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);
					cv::Mat skeleton_kth_layer_mask = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);
					cv::threshold(skeleton_depth_layers, skeleton_kth_layer, kth_value - 1, 255, CV_THRESH_TOZERO);
					cv::threshold(skeleton_kth_layer, skeleton_kth_layer, kth_value, 255, CV_THRESH_TOZERO_INV);
					cv::threshold(skeleton_kth_layer, skeleton_kth_layer_mask, 0, 255, CV_THRESH_BINARY);
					cv::medianBlur(skeleton_kth_layer_mask, skeleton_kth_layer_mask, 3);

					// Filter
					this->find_largest_contour(skeleton_kth_layer_mask, skeleton_head_contour);
					if (skeleton_head_contour.size() < 5)
					{
						continue;
					}

					// Head contour mask and layer
					cv::Mat skeleton_head_contour_mask = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);
					cv::drawContours(skeleton_head_contour_mask, std::vector<std::vector<cv::Point>>{skeleton_head_contour}, 0, topviewkinect::color::CV_WHITE, -1);
					skeleton_depth_silhouette_downsampled.copyTo(skeleton_head_layer, skeleton_head_contour_mask); // largest contour

					// Body contour mask and layer
					if (kth_idx + 1 < kmeans_values.size())
					{
						int second_layer_kmeans_value = kmeans_values[kth_idx + 1];

						cv::Mat skeleton_body_layer_mask = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);
						cv::threshold(skeleton_depth_layers, skeleton_body_layer_mask, second_layer_kmeans_value, 255, CV_THRESH_TOZERO_INV); // Ignore everything after
						skeleton_body_layer_mask.setTo(0, skeleton_head_contour_mask); // Ignore the main head contour
						cv::threshold(skeleton_body_layer_mask, skeleton_body_layer_mask, 0, 255, CV_THRESH_BINARY); // Keep other small contours from the head layer
						cv::medianBlur(skeleton_body_layer_mask, skeleton_body_layer_mask, 3);
						skeleton_depth_silhouette_downsampled.copyTo(skeleton_body_layer, skeleton_body_layer_mask);

						this->find_largest_contour(skeleton_body_layer_mask, skeleton_body_contour);
					}

					// Bottom contour mask and layer
					if (kth_idx + 2 < kmeans_values.size())
					{
						int third_layer_kmeans_value = kmeans_values[kth_idx + 2];

						cv::Mat skeleton_bottom_layer_mask = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);
						cv::threshold(skeleton_depth_layers, skeleton_bottom_layer, third_layer_kmeans_value - 1, 255, CV_THRESH_TOZERO); // Ignore everything before
						cv::threshold(skeleton_bottom_layer, skeleton_bottom_layer, third_layer_kmeans_value, 255, CV_THRESH_TOZERO_INV); // Ignore the feet
						cv::threshold(skeleton_bottom_layer, skeleton_bottom_layer_mask, 0, 255, CV_THRESH_BINARY);
						cv::medianBlur(skeleton_bottom_layer_mask, skeleton_bottom_layer_mask, 3);

						this->find_largest_contour(skeleton_bottom_layer_mask, skeleton_bottom_contour);
					}

					if (kth_idx + 3 < kmeans_values.size())
					{
						int others_kmeans_value = kmeans_values[kth_idx + 3];

						cv::Mat skeleton_others_layer_mask = cv::Mat::zeros(skeleton_depth_silhouette_downsampled.size(), CV_8UC1);
						cv::threshold(skeleton_depth_layers, skeleton_others_layer, others_kmeans_value - 1, 255, CV_THRESH_TOZERO);
						cv::threshold(skeleton_others_layer, skeleton_others_layer, others_kmeans_value, 255, CV_THRESH_TOZERO_INV);
						cv::threshold(skeleton_others_layer, skeleton_others_layer_mask, 0, 255, CV_THRESH_BINARY);
						cv::medianBlur(skeleton_others_layer_mask, skeleton_others_layer_mask, 3);

						this->find_largest_contour(skeleton_others_layer_mask, skeleton_others_contour);
					}

					// Segment contours
					cv::copyMakeBorder(skeleton_depth_body_layers_color, skeleton_depth_body_layers_color, 5, 5, 5, 5, cv::BORDER_CONSTANT, 0);

					cv::copyMakeBorder(skeleton_depth_silhouette_downsampled, skeleton_depth_silhouette_downsampled, 5, 5, 5, 5, cv::BORDER_CONSTANT, 0);
					this->find_largest_contour(skeleton_depth_silhouette_downsampled, skeleton_depth_contour);

					cv::copyMakeBorder(skeleton_infrared_silhouette_downsampled, skeleton_infrared_silhouette_downsampled, 5, 5, 5, 5, cv::BORDER_CONSTANT, 0);

					cv::copyMakeBorder(skeleton_head_layer, skeleton_head_layer, 5, 5, 5, 5, cv::BORDER_CONSTANT, 0);
					this->find_largest_contour(skeleton_head_layer, skeleton_head_contour);

					cv::copyMakeBorder(skeleton_body_layer, skeleton_body_layer, 5, 5, 5, 5, cv::BORDER_CONSTANT, 0);
					this->find_largest_contour(skeleton_body_layer, skeleton_body_contour);

					cv::copyMakeBorder(skeleton_bottom_layer, skeleton_bottom_layer, 5, 5, 5, 5, cv::BORDER_CONSTANT, 0);
					this->find_largest_contour(skeleton_bottom_layer, skeleton_bottom_contour);

					cv::copyMakeBorder(skeleton_others_layer, skeleton_others_layer, 5, 5, 5, 5, cv::BORDER_CONSTANT, 0);
					this->find_largest_contour(skeleton_others_layer, skeleton_others_contour);

					skeleton.set_depth_silhouette(skeleton_depth_silhouette_downsampled);
					skeleton.set_infrared_silhouette(skeleton_infrared_silhouette_downsampled);
					skeleton.set_silhouette_kmeans(skeleton_depth_body_layers_color);
					skeleton.set_silhouette_head_layer(skeleton_head_layer);
					skeleton.set_silhouette_body_layer(skeleton_body_layer);
					skeleton.set_silhouette_bottom_layer(skeleton_bottom_layer);
					skeleton.set_silhouette_others_layer(skeleton_others_layer);
					skeleton.set_depth_contour(skeleton_depth_contour);
					skeleton.set_head_contour(skeleton_head_contour);
					skeleton.set_body_contour(skeleton_body_contour);
					skeleton.set_bottom_contour(skeleton_bottom_contour);
					skeleton.set_others_contour(skeleton_others_contour);

					// Figures
					//cv::Mat depth_viz = skeleton_depth_silhouette_downsampled.clone();
					//cv::Mat head_viz = skeleton_depth_silhouette_downsampled.clone();
					//cv::Mat body_viz = skeleton_depth_silhouette_downsampled.clone();
					//cv::Mat bottom_viz = skeleton_depth_silhouette_downsampled.clone();
					//cv::drawContours(depth_viz, std::vector<std::vector<cv::Point>>{skeleton_depth_contour}, 0, topviewkinect::color::CV_WHITE);
					//cv::drawContours(head_viz, std::vector<std::vector<cv::Point>>{skeleton_head_contour}, 0, topviewkinect::color::CV_WHITE);
					//cv::drawContours(body_viz, std::vector<std::vector<cv::Point>>{skeleton_body_contour}, 0, topviewkinect::color::CV_WHITE);
					//cv::drawContours(bottom_viz, std::vector<std::vector<cv::Point>>{skeleton_bottom_contour}, 0, topviewkinect::color::CV_WHITE);
					//cv::imshow("depth", depth_viz);
					//cv::imshow("infrared", skeleton_infrared_silhouette_downsampled);
					//cv::imshow("body_layers", skeleton_depth_layers);
					//cv::imshow("head", head_viz);
					//cv::imshow("body", body_viz);
					//cv::imshow("bottom", bottom_viz);
					//cv::imshow("_depth", skeleton_depth_silhouette_downsampled);
					//cv::imshow("_layers", skeleton_depth_body_layers_color);
					//cv::imshow("_infrared", skeleton_infrared_silhouette_downsampled);
					//cv::imwrite("E:\\eaglesense\\data\\topviewkinect\\2010\\depth\\paper\\figs\\new\\_depth.png", skeleton_depth_silhouette_downsampled);
					//cv::imwrite("E:\\eaglesense\\data\\topviewkinect\\2010\\depth\\paper\\figs\\new\\_layers.png", skeleton_depth_body_layers_color);
					//cv::imwrite("E:\\eaglesense\\data\\topviewkinect\\2010\\depth\\paper\\figs\\new\\_infrared.png", skeleton_infrared_silhouette_downsampled);

					break;
				}

				// Estimate head orientation (scaled up)
				std::vector<cv::Point> skeleton_resized_head_contour;
				for (int i = 0; i < skeleton_head_contour.size(); ++i)
				{
					const cv::Point head_contour_pt = skeleton_head_contour[i];
					int scaledup_x = boost::math::iround((head_contour_pt.x - 5) * static_cast<float>(skeleton_depth_silhouette.cols) / 32);
					int scaledup_y = boost::math::iround((head_contour_pt.y - 5) * static_cast<float>(skeleton_depth_silhouette.rows) / 32);
					if (scaledup_x < 0 || scaledup_x > skeleton_depth_silhouette.cols || scaledup_y < 0 || scaledup_y > skeleton_depth_silhouette.rows)
					{
						continue;
					}
					skeleton_resized_head_contour.push_back(cv::Point(scaledup_x, scaledup_y));
				}

				cv::RotatedRect skeleton_resized_head_ellipse;
				try
				{
					skeleton_resized_head_ellipse = cv::fitEllipse(skeleton_resized_head_contour);
				}
				catch (cv::Exception)
				{
					continue;
				}

				topviewkinect::skeleton::Joint skeleton_head;
				skeleton_head.x = boost::math::iround(skeleton_resized_head_ellipse.center.x + skeleton_bounding_rect.x);
				skeleton_head.y = boost::math::iround(skeleton_resized_head_ellipse.center.y + skeleton_bounding_rect.y);
				skeleton_head.z = static_cast<int>(skeleton_depth_frame.at<uchar>(skeleton_head.y, skeleton_head.x));
				skeleton_head.orientation = skeleton_resized_head_ellipse.angle;

				// Get major axis orientation
				float skeleton_head_orientation_1 = skeleton_head.orientation + 90;
				float skeleton_head_orientation_2 = skeleton_head.orientation - 90;
				skeleton_head_orientation_1 = skeleton_head_orientation_1 < 0 ? 360 + skeleton_head_orientation_1 : skeleton_head_orientation_1;
				skeleton_head_orientation_2 = skeleton_head_orientation_2 < 0 ? 360 + skeleton_head_orientation_2 : skeleton_head_orientation_2;
				std::vector<std::tuple<double, float>> orientation_distances;

				// Begin activity tracking, assume that the person is walking forward towards the Kinect sensor when entering the tracking area
				if (!skeleton.is_activity_tracked())
				{
					int orientation_1_pt_x = boost::math::iround(skeleton_head.x + 100 * std::cos(skeleton_head_orientation_1 * CV_PI / 180.0));
					int orientation_1_pt_y = boost::math::iround(skeleton_head.y + 100 * std::sin(skeleton_head_orientation_1 * CV_PI / 180.0));
					int orientation_1_vec[3] = { orientation_1_pt_x - skeleton_depth_frame.cols / 2, orientation_1_pt_y - skeleton_depth_frame.rows };
					double orientation_1_distance_to_camera = std::sqrt(std::inner_product(std::begin(orientation_1_vec), std::end(orientation_1_vec), std::begin(orientation_1_vec), 0.0));

					int orientation_2_pt_x = boost::math::iround(skeleton_head.x + 100 * std::cos(skeleton_head_orientation_2 * CV_PI / 180.0));
					int orientation_2_pt_y = boost::math::iround(skeleton_head.y + 100 * std::sin(skeleton_head_orientation_2 * CV_PI / 180.0));
					int orientation_2_vec[3] = { orientation_2_pt_x - skeleton_depth_frame.cols / 2, orientation_2_pt_y - skeleton_depth_frame.rows };
					double orientation_2_distance_to_camera = std::sqrt(std::inner_product(std::begin(orientation_2_vec), std::end(orientation_2_vec), std::begin(orientation_2_vec), 0.0));

					orientation_distances.push_back(std::make_tuple(orientation_1_distance_to_camera, skeleton_head_orientation_1));
					orientation_distances.push_back(std::make_tuple(orientation_2_distance_to_camera, skeleton_head_orientation_2));
				}
				else
				{
					// Find potential orientations
					const topviewkinect::skeleton::Joint previous_skeleton_head = skeleton.get_head();
					int head_orientation_pt_x = boost::math::iround(previous_skeleton_head.x + 100 * std::cos(previous_skeleton_head.orientation * CV_PI / 180.0));
					int head_orientation_pt_y = boost::math::iround(previous_skeleton_head.y + 100 * std::sin(previous_skeleton_head.orientation * CV_PI / 180.0));

					int orientation_1_pt_x = boost::math::iround(previous_skeleton_head.x + 100 * std::cos(skeleton_head_orientation_1 * CV_PI / 180.0));
					int orientation_1_pt_y = boost::math::iround(previous_skeleton_head.y + 100 * std::sin(skeleton_head_orientation_1 * CV_PI / 180.0));
					int orientation_1_distance_vector[2] = { head_orientation_pt_x - orientation_1_pt_x, head_orientation_pt_y - orientation_1_pt_y };
					double orientation_1_distance = std::sqrt(std::inner_product(std::begin(orientation_1_distance_vector), std::end(orientation_1_distance_vector), std::begin(orientation_1_distance_vector), 0.0));

					int orientation_2_pt_x = boost::math::iround(previous_skeleton_head.x + 100 * std::cos(skeleton_head_orientation_2 * CV_PI / 180.0));
					int orientation_2_pt_y = boost::math::iround(previous_skeleton_head.y + 100 * std::sin(skeleton_head_orientation_2 * CV_PI / 180.0));
					int orientation_2_distance_vector[2] = { head_orientation_pt_x - orientation_2_pt_x, head_orientation_pt_y - orientation_2_pt_y };
					double orientation_2_distance = std::sqrt(std::inner_product(std::begin(orientation_2_distance_vector), std::end(orientation_2_distance_vector), std::begin(orientation_2_distance_vector), 0.0));

					orientation_distances.push_back(std::make_tuple(orientation_1_distance, skeleton_head_orientation_1));
					orientation_distances.push_back(std::make_tuple(orientation_2_distance, skeleton_head_orientation_2));
				}

				// Update orientation to the corresponding smallest distance change
				std::sort(orientation_distances.begin(), orientation_distances.end(), [](const std::tuple<double, double>& t1, const std::tuple<double, double>& t2) {
					return std::get<0>(t1) < std::get<0>(t2);
				});
				skeleton_head.orientation = std::get<1>(orientation_distances[0]);

				skeleton.set_head(skeleton_head);
				skeleton.set_activity_tracked(true);

				// Figures
				//cv::Mat skeleton_head_orientation_frame = skeleton_depth_body_layers_color.clone();
				//cv::Point head_pt = cv::Point((skeleton_head.x - skeleton_bounding_rect.x) * 128 / skeleton_depth_silhouette.size().width, (skeleton_head.y - skeleton_bounding_rect.y) * 128 / skeleton_depth_silhouette.size().height);
				//int head_angle_pt_x = boost::math::iround(head_pt.x + 30 * std::cos(skeleton_head.orientation * CV_PI / 180.0));
				//int head_angle_pt_y = boost::math::iround(head_pt.y + 30 * std::sin(skeleton_head.orientation * CV_PI / 180.0));
				//cv::Point head_angle_pt = cv::Point(head_angle_pt_x, head_angle_pt_y);
				////cv::Mat skeleton_head_frame_color(skeleton_head_orientation_frame.size(), CV_8UC3);
				////cv::cvtColor(skeleton_head_orientation_frame, skeleton_head_frame_color, CV_GRAY2BGR);
				//cv::circle(skeleton_head_orientation_frame, head_pt, 3, topviewkinect::color::CV_BGR_WHITE, -1);
				//cv::arrowedLine(skeleton_head_orientation_frame, head_pt, head_angle_pt, topviewkinect::color::CV_BGR_WHITE, 2, 8, 0, 0.5);
				////cv::ellipse(skeleton_head_orientation_frame, skeleton_head_ellipse, common_colors::CV_BGR_LAYERS[2], 1);
				//cv::imshow("head", skeleton_head_orientation_frame);
				//cv::imwrite("E:\\eaglesense\\data\\topviewkinect\\2010\\depth\\paper\\figs\\new\\_head.png", skeleton_head_orientation_frame);
			}

			return true;
		}

		void TopViewSpace::combine_openpose_and_depth()
		{

		}

		// Features

		void TopViewSpace::find_contour_centers(const cv::Mat& src, std::vector<std::tuple<topviewkinect::skeleton::Joint, double, double>>& contours_centers) const
		{
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(src.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			for (int i = 0; i < contours.size(); ++i)
			{
				std::vector<cv::Point> contour_i = contours[i];
				double contour_i_area = cv::contourArea(contour_i);
				if (contour_i_area == 0 || contour_i.size() < 5) // Edge cases
				{
					continue;
				}

				topviewkinect::skeleton::Joint center_jt;
				this->find_mass_center(src, contour_i, center_jt);

				double furthest_distance = 0;
				this->find_furthest_points(contour_i, cv::Point(), cv::Point(), &furthest_distance);

				contours_centers.push_back(std::make_tuple(center_jt, contour_i_area, furthest_distance));
			}
		}

		void TopViewSpace::find_furthest_points(const cv::Mat& src, cv::Point& pt1, cv::Point& pt2, double* largest_distance) const
		{
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(src.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			std::vector<cv::Point> contour_points;
			for (int i = 0; i < contours.size(); ++i)
			{
				std::vector<cv::Point> contour_i = contours[i];
				double contour_i_area = cv::contourArea(contour_i);
				if (contour_i_area == 0 || contour_i.size() < 5) // Edge cases
				{
					continue;
				}
				for (int j = 0; j < contour_i.size(); ++j)
				{
					contour_points.push_back(contour_i[j]);
				}
			}

			this->find_furthest_points(contour_points, pt1, pt2, largest_distance);
		}

		void TopViewSpace::find_furthest_points(const std::vector<cv::Point>& contour, cv::Point& pt1, cv::Point& pt2, double* largest_distance) const
		{
			for (int i = 0; i < contour.size(); ++i)
			{
				const cv::Point contour_pt = contour[i];
				for (int j = i + 1; j < contour.size(); ++j)
				{
					const cv::Point contour_pt_2 = contour[j];
					std::array<int, 2> distance_vec = { contour_pt.x - contour_pt_2.x, contour_pt.y - contour_pt_2.y };
					double distance = std::sqrt(std::inner_product(std::begin(distance_vec), std::end(distance_vec), std::begin(distance_vec), 0.0));
					if (distance > *largest_distance)
					{
						pt1.x = contour_pt.x;
						pt1.y = contour_pt.y;
						pt2.x = contour_pt_2.x;
						pt2.y = contour_pt_2.y;
						*largest_distance = distance;
					}
				}
			}
		}

		void TopViewSpace::find_joint(const cv::Mat& src, const cv::Point& pt, int min_depth, topviewkinect::skeleton::Joint& jt, int ksize) const
		{
			int pt_x = pt.x;
			int pt_y = pt.y;
			int pt_depth = static_cast<int>(src.at<uchar>(pt.y, pt.x));

			// Correct depth (5 x 5)
			if (pt_depth < min_depth || pt_depth < 10)
			{
				cv::Mat pt_mask = cv::Mat::zeros(src.size(), CV_8UC1);
				cv::Rect pt_roi(pt_x - ksize / 2, pt_y - ksize / 2, ksize, ksize);
				cv::rectangle(pt_mask, pt_roi, 255, -1);

				cv::Point pt_max_depth_loc;
				double pt_max_depth;
				cv::minMaxLoc(src, 0, &pt_max_depth, 0, &pt_max_depth_loc, pt_mask);

				pt_x = pt_max_depth_loc.x;
				pt_y = pt_max_depth_loc.y;
				pt_depth = static_cast<int>(pt_max_depth);
			}

			jt.x = pt_x;
			jt.y = pt_y;
			jt.z = pt_depth;
		}

		void TopViewSpace::compute_skeleton_features()
		{
			// Everything is in double, for convenience
			for (topviewkinect::skeleton::Skeleton& skeleton : skeletons)
			{
				if (!skeleton.is_activity_tracked())
				{
					continue;
				}

				// Features
				std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_AREAS> f_layer_areas{};
				std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_CONTOURS> f_layer_contours{};
				std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_DISTANCES> f_layer_distances{};
				std::array<double, topviewkinect::vision::FEATURE_NUM_INTRALAYER_POSITIONS> f_intralayer_positions{};
				std::array<double, topviewkinect::vision::FEATURE_NUM_INTERLAYER_POSITIONS> f_interlayer_positions{};
				std::array<double, topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES> f_body_extremities{};
				std::array<double, topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES_INFRAREDS> f_body_extremities_infrareds{};

				const topviewkinect::skeleton::Joint skeleton_head = skeleton.get_head();
				const cv::Mat skeleton_depth_silhouette = skeleton.get_depth_silhouette();
				const cv::Mat skeleton_infrared_silhouette = skeleton.get_infrared_silhouette();
				const cv::Mat skeleton_kmeans = skeleton.get_silhouette_kmeans();
				const cv::Mat skeleton_head_layer = skeleton.get_silhouette_head_layer();
				const cv::Mat skeleton_body_layer = skeleton.get_silhouette_body_layer();
				const cv::Mat skeleton_bottom_layer = skeleton.get_silhouette_bottom_layer();
				const cv::Mat skeleton_others_layer = skeleton.get_silhouette_others_layer();
				const std::vector<cv::Point> skeleton_depth_contour = skeleton.get_depth_contour();
				const std::vector<cv::Point> skeleton_head_contour = skeleton.get_head_contour();
				const std::vector<cv::Point> skeleton_body_contour = skeleton.get_body_contour();
				const std::vector<cv::Point> skeleton_bottom_contour = skeleton.get_bottom_contour();

				// F. Layer areas
				double head_area = cv::countNonZero(skeleton_head_layer);
				double body_area = cv::countNonZero(skeleton_body_layer);
				double bottom_area = cv::countNonZero(skeleton_bottom_layer);
				double total_area = head_area + body_area + bottom_area;
				f_layer_areas[0] = head_area / total_area;
				f_layer_areas[1] = body_area / total_area;
				f_layer_areas[2] = bottom_area / total_area;
				skeleton.set_features(f_layer_areas, f_layer_contours, f_layer_distances, f_intralayer_positions, f_interlayer_positions, f_body_extremities, f_body_extremities_infrareds);

				// Compute features for sufficiently large contours
				if (skeleton_head_contour.size() < 5 || skeleton_body_contour.size() < 5)
				{
					continue;
				}

				bool contains_bottom_layer = skeleton_bottom_contour.size() > 5 && cv::contourArea(skeleton_bottom_contour) > 10;

				// Centers
				topviewkinect::skeleton::Joint body_layer_center, head_layer_center, bottom_layer_center;
				this->find_mass_center(skeleton_head_layer, skeleton_head_contour, head_layer_center);
				this->find_mass_center(skeleton_body_layer, skeleton_body_contour, body_layer_center);
				if (contains_bottom_layer) this->find_mass_center(skeleton_bottom_layer, skeleton_bottom_contour, bottom_layer_center);

				// Layer Extremities

				// (i) head
				cv::Point skeleton_head_extreme_pt_1, skeleton_head_extreme_pt_2;
				topviewkinect::skeleton::Joint skeleton_head_extreme_jt_1{}, skeleton_head_extreme_jt_2{};
				double skeleton_head_largest_distance = 0;
				this->find_furthest_points(skeleton_head_layer, skeleton_head_extreme_pt_1, skeleton_head_extreme_pt_2, &skeleton_head_largest_distance);
				this->find_joint(skeleton_head_layer, skeleton_head_extreme_pt_1, head_layer_center.z, skeleton_head_extreme_jt_1, 5);
				this->find_joint(skeleton_head_layer, skeleton_head_extreme_pt_2, head_layer_center.z, skeleton_head_extreme_jt_2, 5);

				// (ii) body
				cv::Point skeleton_body_extreme_pt_1, skeleton_body_extreme_pt_2;
				topviewkinect::skeleton::Joint skeleton_body_extreme_jt_1{}, skeleton_body_extreme_jt_2{};
				double skeleton_body_largest_distance = 0;
				this->find_furthest_points(skeleton_body_layer, skeleton_body_extreme_pt_1, skeleton_body_extreme_pt_2, &skeleton_body_largest_distance);
				this->find_joint(skeleton_body_layer, skeleton_body_extreme_pt_1, body_layer_center.z, skeleton_body_extreme_jt_1, 5);
				this->find_joint(skeleton_body_layer, skeleton_body_extreme_pt_2, body_layer_center.z, skeleton_body_extreme_jt_2, 5);

				// body other contours
				std::vector<std::tuple<topviewkinect::skeleton::Joint, double, double>> skeleton_body_all_contours;
				this->find_contour_centers(skeleton_body_layer, skeleton_body_all_contours);
				std::sort(skeleton_body_all_contours.begin(), skeleton_body_all_contours.end(), [](const auto& t1, const auto& t2) {
					return std::get<1>(t1) > std::get<1>(t2);
				});
				skeleton_body_all_contours.resize(3);

				// (iii) bottom
				cv::Point skeleton_bottom_extreme_pt_1, skeleton_bottom_extreme_pt_2;
				topviewkinect::skeleton::Joint skeleton_bottom_extreme_jt_1{}, skeleton_bottom_extreme_jt_2{};
				double skeleton_bottom_largest_distance = 0;

				std::vector<std::tuple<topviewkinect::skeleton::Joint, double, double>> skeleton_bottom_all_contours;
				if (contains_bottom_layer)
				{
					this->find_furthest_points(skeleton_bottom_layer, skeleton_bottom_extreme_pt_1, skeleton_bottom_extreme_pt_2, &skeleton_bottom_largest_distance);
					this->find_joint(skeleton_bottom_layer, skeleton_bottom_extreme_pt_1, bottom_layer_center.z, skeleton_bottom_extreme_jt_1, 5);
					this->find_joint(skeleton_bottom_layer, skeleton_bottom_extreme_pt_2, bottom_layer_center.z, skeleton_bottom_extreme_jt_2, 5);

					// bottom other contours
					this->find_contour_centers(skeleton_bottom_layer, skeleton_bottom_all_contours);
					std::sort(skeleton_bottom_all_contours.begin(), skeleton_bottom_all_contours.end(), [](const auto& t1, const auto& t2) {
						return std::get<1>(t1) > std::get<1>(t2);
					});
					skeleton_bottom_all_contours.resize(3);
				}

				// Figures
				//cv::Mat skeleton_kmeans_layer_extremities = skeleton_kmeans.clone();
				//cv::line(skeleton_kmeans_layer_extremities, cv::Point(skeleton_bottom_extreme_pt_1.x, skeleton_bottom_extreme_pt_1.y), cv::Point(skeleton_bottom_extreme_pt_2.x, skeleton_bottom_extreme_pt_2.y), common_colors::CV_BGR_WHITE, 2);
				//cv::line(skeleton_kmeans_layer_extremities, cv::Point(skeleton_head_extreme_jt_1.x, skeleton_head_extreme_jt_1.y), cv::Point(skeleton_head_extreme_jt_2.x, skeleton_head_extreme_jt_2.y), common_colors::CV_BGR_WHITE, 2);
				//cv::line(skeleton_kmeans_layer_extremities, cv::Point(skeleton_body_extreme_jt_1.x, skeleton_body_extreme_jt_1.y), cv::Point(skeleton_body_extreme_jt_2.x, skeleton_body_extreme_jt_2.y), common_colors::CV_BGR_WHITE, 2);
				//cv::circle(skeleton_kmeans_layer_extremities, cv::Point(skeleton_head_extreme_jt_1.x, skeleton_head_extreme_jt_1.y), 7, common_colors::CV_BGR_WHITE, -1);
				//cv::circle(skeleton_kmeans_layer_extremities, cv::Point(skeleton_head_extreme_jt_2.x, skeleton_head_extreme_jt_2.y), 7, common_colors::CV_BGR_WHITE, -1);
				//cv::circle(skeleton_kmeans_layer_extremities, cv::Point(skeleton_body_extreme_jt_1.x, skeleton_body_extreme_jt_1.y), 7, common_colors::CV_BGR_WHITE, -1);
				//cv::circle(skeleton_kmeans_layer_extremities, cv::Point(skeleton_body_extreme_jt_2.x, skeleton_body_extreme_jt_2.y), 7, common_colors::CV_BGR_WHITE, -1);
				//cv::circle(skeleton_kmeans_layer_extremities, cv::Point(skeleton_bottom_extreme_pt_1.x, skeleton_bottom_extreme_pt_1.y), 7, common_colors::CV_BGR_WHITE, -1);
				//cv::circle(skeleton_kmeans_layer_extremities, cv::Point(skeleton_bottom_extreme_pt_2.x, skeleton_bottom_extreme_pt_2.y), 7, common_colors::CV_BGR_WHITE, -1);
				//cv::imshow("layer_ext", skeleton_kmeans_layer_extremities);
				//cv::imwrite("E:\\eaglesense\\data\\topviewkinect\\2010\\depth\\paper\\figs\\new\\_layer_ext.png", skeleton_kmeans_layer_extremities);

				// Body extremities
				std::vector<topviewkinect::skeleton::Joint> body_convexity_extremities;
				//cv::Mat body_defects_frame = skeleton_kmeans.clone();
				std::vector<cv::Point> body_hull_points;
				cv::convexHull(skeleton_depth_contour, body_hull_points);

				std::vector<int> body_hull;
				std::vector<cv::Vec4i> body_defects;
				cv::convexHull(skeleton_depth_contour, body_hull);
				cv::convexityDefects(skeleton_depth_contour, body_hull, body_defects);
				float max_depth = 0;
				for (const cv::Vec4i& defect : body_defects)
				{
					int start_idx = defect[0];
					int end_idx = defect[1];
					cv::Point start_pt(skeleton_depth_contour[start_idx]);
					//cv::circle(body_defects_frame, cv::Point(start_pt.x, start_pt.y), 7, common_colors::CV_BGR_WHITE, 2);

					topviewkinect::skeleton::Joint start_jt;
					this->find_joint(skeleton_depth_silhouette, start_pt, body_layer_center.z, start_jt, 5);
					body_convexity_extremities.push_back(start_jt);
				}
				//cv::drawContours(body_defects_frame, std::vector<std::vector<cv::Point>>{body_hull_points}, 0, common_colors::CV_BGR_WHITE, 2);
				//cv::imshow("_defects", body_defects_frame);
				//cv::imwrite("E:\\eaglesense\\data\\topviewkinect\\2010\\depth\\paper\\figs\\new\\_defects.png", body_defects_frame);

				// Mask the head and feet
				cv::Mat head_mask_dilated = cv::Mat::zeros(skeleton_depth_silhouette.size(), CV_8UC1);
				cv::threshold(skeleton_head_layer, head_mask_dilated, 0, 255, CV_THRESH_BINARY);
				cv::dilate(head_mask_dilated, head_mask_dilated, cv::Mat(), cv::Point(-1, -1), 3); // we make the head bigger
				cv::threshold(head_mask_dilated, head_mask_dilated, 0, 255, CV_THRESH_BINARY_INV); // so we can ignore points that lie on the head
				cv::Mat others_mask_dilated = cv::Mat::zeros(skeleton_depth_silhouette.size(), CV_8UC1);
				cv::threshold(skeleton_others_layer, others_mask_dilated, 0, 255, CV_THRESH_BINARY);
				cv::threshold(others_mask_dilated, others_mask_dilated, 0, 255, CV_THRESH_BINARY_INV);

				// Ignore body convedxity extremities near the head and the feet
				std::vector<topviewkinect::skeleton::Joint> vec_convexity_extremities_to_remove;
				for (const topviewkinect::skeleton::Joint& extreme_pt : body_convexity_extremities)
				{
					if (head_mask_dilated.at<uchar>(extreme_pt.y, extreme_pt.x) == 0 || others_mask_dilated.at<uchar>(extreme_pt.y, extreme_pt.x) == 0)
					{
						vec_convexity_extremities_to_remove.push_back(extreme_pt);
					}
				}
				for (const topviewkinect::skeleton::Joint& convexity_extreme_to_remove : vec_convexity_extremities_to_remove)
				{
					body_convexity_extremities.erase(std::remove(body_convexity_extremities.begin(), body_convexity_extremities.end(), convexity_extreme_to_remove), body_convexity_extremities.end());
				}

				// Retain at most 5 body convexity extremities
				while (body_convexity_extremities.size() > 5)
				{
					topviewkinect::skeleton::Joint convexity_extreme_to_remove;
					double shortest_distance = DBL_MAX;
					for (int i = 0; i < body_convexity_extremities.size(); ++i)
					{
						topviewkinect::skeleton::Joint current = body_convexity_extremities[i];
						topviewkinect::skeleton::Joint next = i == body_convexity_extremities.size() - 1 ? body_convexity_extremities[0] : body_convexity_extremities[i + 1];
						double distance = this->relative_2d_distance(current, next);
						if (distance < shortest_distance)
						{
							convexity_extreme_to_remove = body_convexity_extremities[i];
							shortest_distance = distance;
						}
					}
					body_convexity_extremities.erase(std::remove(body_convexity_extremities.begin(), body_convexity_extremities.end(), convexity_extreme_to_remove), body_convexity_extremities.end());
				}

				// Figures
				//cv::Mat body_defects_filtered_frame = skeleton_kmeans.clone();
				//// Show body extremities
				//for (const std::tuple<topviewkinect::skeleton::Joint, float>& extreme_pt_tuple : body_convexity_extremities)
				//{
				//    const topviewkinect::skeleton::Joint defect_pt = std::get<0>(extreme_pt_tuple);
				//    cv::circle(body_defects_filtered_frame, cv::Point(defect_pt.x, defect_pt.y), 7, common_colors::CV_BGR_WHITE, -1);
				//}
				//cv::drawContours(body_defects_filtered_frame, std::vector<std::vector<cv::Point>>{body_hull_points}, 0, common_colors::CV_BGR_WHITE, 2);
				//cv::imshow("_defects_filtered", body_defects_filtered_frame);
				//cv::imwrite("E:\\eaglesense\\data\\topviewkinect\\2010\\depth\\paper\\figs\\new\\_defects_filtered.png", body_defects_filtered_frame);

				// F. Layer contours
				f_layer_contours[0] = static_cast<double>(skeleton_body_all_contours.size());
				f_layer_contours[1] = static_cast<double>(skeleton_bottom_all_contours.size());

				// F. Layer maximal distances
				// (i) three layers
				f_layer_distances[0] = skeleton_head_largest_distance;
				f_layer_distances[1] = skeleton_body_largest_distance;
				f_layer_distances[2] = skeleton_bottom_largest_distance;
				// (ii) body layer
				for (int i = 0; i < 3 && i < skeleton_body_all_contours.size(); ++i)
				{
					f_layer_distances[3 + i * 2] = std::get<2>(skeleton_body_all_contours[i]); // distance
					f_layer_distances[3 + i * 2 + 1] = std::get<1>(skeleton_body_all_contours[i]); // area
				}
				// (iii) bottom layer
				for (int i = 0; i < 3 && i < skeleton_bottom_all_contours.size(); ++i)
				{
					f_layer_distances[9 + i * 2] = std::get<2>(skeleton_bottom_all_contours[i]);
					f_layer_distances[9 + i * 2 + 1] = std::get<1>(skeleton_bottom_all_contours[i]);
				}

				// F. Intralayer positions
				int intralayer_position_idx = 0;
				// (i) head
				{
					std::array<double, 3> head_extreme_1_postion = this->relative_3d_position(head_layer_center, skeleton_head_extreme_jt_1);
					std::copy(head_extreme_1_postion.begin(), head_extreme_1_postion.end(), f_intralayer_positions.begin() + intralayer_position_idx);

					std::array<double, 3> head_extreme_2_postion = this->relative_3d_position(head_layer_center, skeleton_head_extreme_jt_2);
					std::copy(head_extreme_2_postion.begin(), head_extreme_2_postion.end(), f_intralayer_positions.begin() + intralayer_position_idx + 3);

					std::array<double, 3> head_extremes_postion = this->relative_3d_position(skeleton_head_extreme_jt_1, skeleton_head_extreme_jt_2);
					std::copy(head_extremes_postion.begin(), head_extremes_postion.end(), f_intralayer_positions.begin() + intralayer_position_idx + 6);
				}
				intralayer_position_idx += 9;
				// (ii) body
				{
					std::array<double, 3> body_extreme_1_postion = this->relative_3d_position(body_layer_center, skeleton_body_extreme_jt_1);
					std::copy(body_extreme_1_postion.begin(), body_extreme_1_postion.end(), f_intralayer_positions.begin() + intralayer_position_idx);

					std::array<double, 3> body_extreme_2_postion = this->relative_3d_position(body_layer_center, skeleton_body_extreme_jt_2);
					std::copy(body_extreme_2_postion.begin(), body_extreme_2_postion.end(), f_intralayer_positions.begin() + intralayer_position_idx + 3);

					std::array<double, 3> body_extremes_postion = this->relative_3d_position(skeleton_body_extreme_jt_1, skeleton_body_extreme_jt_2);
					std::copy(body_extremes_postion.begin(), body_extremes_postion.end(), f_intralayer_positions.begin() + intralayer_position_idx + 6);
				}
				intralayer_position_idx += 9;
				// (iii) bottonm
				if (contains_bottom_layer)
				{
					std::array<double, 3> bottom_extreme_1_postion = this->relative_3d_position(bottom_layer_center, skeleton_bottom_extreme_jt_1);
					std::copy(bottom_extreme_1_postion.begin(), bottom_extreme_1_postion.end(), f_intralayer_positions.begin() + intralayer_position_idx);

					std::array<double, 3> bottom_extreme_2_postion = this->relative_3d_position(bottom_layer_center, skeleton_bottom_extreme_jt_2);
					std::copy(bottom_extreme_2_postion.begin(), bottom_extreme_2_postion.end(), f_intralayer_positions.begin() + intralayer_position_idx + 3);

					std::array<double, 3> bottom_extremes_postion = this->relative_3d_position(skeleton_bottom_extreme_jt_1, skeleton_bottom_extreme_jt_2);
					std::copy(bottom_extremes_postion.begin(), bottom_extremes_postion.end(), f_intralayer_positions.begin() + intralayer_position_idx + 6);
				}

				// F. Interlayer positions
				for (int i = 0; i < 3 && i < skeleton_body_all_contours.size(); ++i)
				{
					const topviewkinect::skeleton::Joint body_contour_center = std::get<0>(skeleton_body_all_contours[i]);

					std::array<double, 3> head_body_postion = this->relative_3d_position(head_layer_center, body_contour_center);
					std::copy(head_body_postion.begin(), head_body_postion.end(), f_interlayer_positions.begin() + i * 6);

					std::array<double, 3> body_bottom_postion = this->relative_3d_position(bottom_layer_center, body_contour_center);
					std::copy(body_bottom_postion.begin(), body_bottom_postion.end(), f_interlayer_positions.begin() + i * 6 + 3);
				}

				cv::Mat infrared_silhouette_mask = cv::Mat::zeros(skeleton_depth_silhouette.size(), CV_8UC1);
				skeleton_infrared_silhouette.copyTo(infrared_silhouette_mask, head_mask_dilated);
				cv::Mat others_mask_dilated_inv;
				cv::threshold(others_mask_dilated, others_mask_dilated_inv, 0, 255, CV_THRESH_BINARY_INV);
				infrared_silhouette_mask.setTo(0, others_mask_dilated_inv);

				// F. Body extremities depths and infrareds
				f_body_extremities[0] = static_cast<double>(body_convexity_extremities.size());

				// Ensure enough space
				cv::Mat extreme_infrared_area_padded_16;
				cv::copyMakeBorder(infrared_silhouette_mask, extreme_infrared_area_padded_16, 16, 16, 16, 16, cv::BORDER_CONSTANT, 0);
				cv::Mat extreme_infrared_area_mask_16 = cv::Mat::zeros(extreme_infrared_area_padded_16.size(), CV_8UC1);
				cv::Mat extreme_infrared_area_16 = cv::Mat::zeros(extreme_infrared_area_padded_16.size(), CV_8UC1);
				cv::Mat extreme_infrared_area_16_total = cv::Mat::zeros(extreme_infrared_area_padded_16.size(), CV_8UC1);
				//cv::Mat extreme_infrared_colors = cv::Mat::zeros(extreme_infrared_area_padded_16.size(), CV_8UC3);

				for (int i = 0; i < body_convexity_extremities.size(); ++i)
				{
					const topviewkinect::skeleton::Joint convexity_extreme_pt = body_convexity_extremities[i];

					// Infrared area
					extreme_infrared_area_mask_16.setTo(0);
					cv::Rect object_infrared_roi_16(convexity_extreme_pt.x - 8 + 16, convexity_extreme_pt.y - 8 + 16, 16, 16);
					cv::rectangle(extreme_infrared_area_mask_16, object_infrared_roi_16, 255, -1);
					extreme_infrared_area_16.setTo(0);
					extreme_infrared_area_padded_16.copyTo(extreme_infrared_area_16, extreme_infrared_area_mask_16);
					extreme_infrared_area_padded_16.copyTo(extreme_infrared_area_16_total, extreme_infrared_area_mask_16);
					std::vector<cv::Point> infrared_contour;
					double infrared_contour_area;
					this->find_largest_contour(extreme_infrared_area_16, infrared_contour, &infrared_contour_area);
					f_body_extremities_infrareds[i] = infrared_contour_area;

					//std::cout << i << " - area: " << extreme_infrared_areas[i] << std::endl;
					//cv::threshold(extreme_infrared_area, extreme_infrared_area, 0, 255, CV_THRESH_BINARY);
					//cv::imshow(std::to_string(i) + " area", extreme_infrared_area);
					//extreme_infrared_colors.setTo(0);
					//cv::rectangle(extreme_infrared_colors, object_infrared_roi_16, common_colors::CV_BGR_WHITE, 1);
					//extreme_infrared_colors.setTo(common_colors::CV_BGR_RED, extreme_infrared_area_16);
				}
				std::vector<cv::Point> largest_infrared_contour;
				double largest_infrared_contour_area;
				this->find_largest_contour(extreme_infrared_area_16_total, largest_infrared_contour, &largest_infrared_contour_area);
				f_body_extremities_infrareds[5] = largest_infrared_contour_area;

				//cv::Mat largest_infrared_contour_mask = cv::Mat::zeros(extreme_infrared_area_padded_16.size(), CV_8UC1);
				//cv::drawContours(largest_infrared_contour_mask, std::vector<std::vector<cv::Point>>{largest_infrared_contour}, 0, 255, -1);
				//std::cout << "size: " << f_body_extremities_infrareds[0] << std::endl;
				//cv::imshow("infrareds", extreme_infrared_colors);
				//cv::imwrite("E:\\eaglesense\\data\\topviewkinect\\2010\\depth\\paper\\figs\\new\\_infrareds.png", extreme_infrared_colors);

				// Update features
				skeleton.set_features(f_layer_areas, f_layer_contours, f_layer_distances, f_intralayer_positions, f_interlayer_positions, f_body_extremities, f_body_extremities_infrareds);
			}
		}

		const bool TopViewSpace::is_calibration_ready() const
		{
			return this->current_num_background_frames == 0;
		}

		const size_t TopViewSpace::num_people() const
		{
			return this->skeletons.size();
		}

		const std::vector<topviewkinect::skeleton::Skeleton> TopViewSpace::get_skeletons() const
		{
			return this->skeletons;
		}

		const std::string TopViewSpace::skeletons_json() const
		{
			/*
			{
			"timestamp": 0,
			"skeletons": [
			{
			"head...activity..."
			}
			]
			}
			*/

			std::ostringstream buffer;
			buffer << "{";
			buffer << "\"timestamp\": " << this->kinect_depth_frame_timestamp << ",";
			buffer << "\"skeletons\": [";
			for (int i = 0; i < this->skeletons.size(); ++i)
			{
				buffer << skeletons[i].json();
				if (i + 1 < this->skeletons.size())
				{
					buffer << ",";
				}
			}
			buffer << "]}";
			return buffer.str();
		}

		cv::Mat TopViewSpace::get_depth_frame() const
		{
			return this->depth_frame.clone();
		}

		cv::Mat TopViewSpace::get_infrared_frame() const
		{
			return this->infrared_frame.clone();
		}

		cv::Mat TopViewSpace::get_low_infrared_frame() const
		{
			return this->infrared_low_frame.clone();
		}

		cv::Mat TopViewSpace::get_rgb_frame() const
		{
			return this->rgb_frame.clone();
		}

		cv::Mat TopViewSpace::get_visualization_frame() const
		{
			return this->visualization_frame.clone();
		}

		cv::Mat TopViewSpace::get_android_sensor_frame() const
		{
			return this->android_sensor_frame.clone();
		}

		cv::Mat TopViewSpace::get_crossmotion_calibration_frame() const
		{
			return this->crossmotion_calibration_frame.clone();
		}
	}
}
