
/**
EagleSense infrastructure

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright ?2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include "opencv2/cudalegacy.hpp"

#include <Kinect.h>

#include <cpprest/http_client.h>
#undef U

#include <string>
#include <vector>

#include "thirdparty/openpose/openpose_wrapper.h"
#include "topviewkinect/topviewkinect.h"
#include "topviewkinect/skeleton/skeleton.h"
#include "topviewkinect/vision/classifier.h"
#include "topviewkinect/vision/framerate.h"
#include "topviewkinect/vision/log.h"

//void onMouse(int event, int x, int y);
//static void onMouse(int event, int x, int y, int, void* userdata);

namespace topviewkinect
{
    namespace vision
    {
        class TopViewSpace
        {
        private:
            // EagleSense tracking configuration
            topviewkinect::Configuration configuration;

            // Kinect sensor
            IKinectSensor* kinect_sensor;
            IMultiSourceFrameReader* kinect_multisource_frame_reader;
            bool kinect_multisource_frame_arrived = false;
            int kinect_frame_id;
            signed long long kinect_depth_frame_timestamp;
            signed long long kinect_infrared_frame_timestamp;
            signed long long kinect_rgb_frame_timestamp;
            void apply_kinect_multisource_frame(const int frame_id, const cv::Mat& depth_frame, const cv::Mat& infrared_frame, const cv::Mat& infrared_low_frame = cv::Mat(), const cv::Mat& rgb_frame = cv::Mat());

			// Calibration
			cv::Mat crossmotion_calibration_frame;
			std::vector<cv::Point> crossmotion_calibration_points_to_add;
			std::vector<cv::Point> crossmotion_calibration_points_2d;
			std::vector<CameraSpacePoint> crossmotion_calibration_points_3d;
			//CameraSpacePoint* color_to_camera_space_points;
			//void calibration_on_mouse(int event, int x, int y, int, void* user_data);

			// Android sensors
			void apply_android_sensor_data();

			// Optical Flow
			cv::Ptr<cv::cuda::BroxOpticalFlow> brox_optflow;
			cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn_optflow;
			cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> tvl1_optflow;
			cv::Mat compute_optical_flow(const cv::Mat& src_prev, const cv::Mat& src_next, cv::Mat& viz, const char* name);

			// OpenPose wrapper
			op::Wrapper<std::vector<thirdparty::openpose::UserDatum>> op_wrapper{ op::ThreadManagerMode::Asynchronous };
			bool op_processing;
			std::vector<thirdparty::openpose::Skeleton> op_skeletons;

            // Background extraction
            int current_num_background_frames;
            cv::Mat foreground_mask;
            cv::Ptr<cv::BackgroundSubtractor> p_background_extractor;
            int empty_background_counter;

            // Data and Visualizations
            topviewkinect::vision::FrameRateController framerate_controller;
            cv::Mat depth_frame;
            cv::Mat depth_foreground_frame;
            cv::Mat infrared_frame;
            cv::Mat infrared_low_frame;
            cv::Mat rgb_frame;
            cv::Mat visualization_frame;
			cv::Mat android_sensor_frame;

			// Sensor data
			std::mutex android_sensor_mutex;
			std::deque<topviewkinect::AndroidSensorData> android_sensor_data;
			std::deque<topviewkinect::AndroidSensorData> android_sensor_data_tmp;
			int android_sensor_label = 0;

            // Skeletons
            std::vector<topviewkinect::skeleton::Skeleton> skeletons;
            topviewkinect::vision::InteractionClassifier interaction_classifier;
            topviewkinect::vision::InteractionLog interaction_log;

            // RESTful client
            web::http::client::http_client* restful_client;

            bool inside_acitivty_tracking_zone(const cv::Mat& skeleton_depth_frame, int offset_x, int offset_y) const;
			int detect_skeletons();
			void combine_openpose_and_depth();
            void compute_skeleton_features();

            // Features
            const std::array<double, 3> relative_3d_position(const topviewkinect::skeleton::Joint& joint1, const topviewkinect::skeleton::Joint& joint2) const;
            const double relative_2d_distance(const topviewkinect::skeleton::Joint& joint1, const topviewkinect::skeleton::Joint& joint2) const;
            void find_largest_contour(const cv::Mat& src, std::vector<cv::Point>& largest_contour, double* largest_area = 0) const;
            void find_mass_center(const cv::Mat& src, const std::vector<cv::Point>& contour, topviewkinect::skeleton::Joint& center) const;
            void find_contour_centers(const cv::Mat& src, std::vector<std::tuple<topviewkinect::skeleton::Joint, double, double>>& contours_centers) const;
            void find_furthest_points(const cv::Mat& src, cv::Point& pt1, cv::Point& pt2, double* largest_distance) const;
            void find_furthest_points(const std::vector<cv::Point>& contour, cv::Point& pt1, cv::Point& pt2, double* largest_distance) const;
            void find_joint(const cv::Mat& src, const cv::Point& pt, int min_depth, topviewkinect::skeleton::Joint& jt, int ksize) const;

        public:
            TopViewSpace();
            ~TopViewSpace();

			ICoordinateMapper* kinect_coordinate_mapper;

            bool initialize();

            // Tracking
            bool refresh_kinect_frames();
            bool process_kinect_frames();
            const int get_kinect_frame_id() const;
            const bool is_calibration_ready() const;
            const size_t num_people() const;
            const std::vector<topviewkinect::skeleton::Skeleton> get_skeletons() const;
            const std::string skeletons_json() const;

			// Sensor Fusion
			bool refresh_android_sensor_data(topviewkinect::AndroidSensorData& data);
			bool refresh_android_sensor_frame();
			bool process_android_sensor_data();
			void set_android_sensor_label(const std::string& label);
			bool calibrate_sensor_fusion(const cv::Point& new_point);
			bool offline_calibration();

            // Replay
            bool load_dataset(const int dataset_id);
            void replay_previous_frame();
            void replay_next_frame();

            // Capture
            bool create_dataset(const int dataset_id);
            bool save_kinect_frames();
			bool save_android_sensor_data();
            bool save_visualization();
			bool save_calibration();

            // Postprocess
            void postprocess(const std::string& dataset_name, const bool keep_label);

            // Visualizations
            cv::Mat get_depth_frame() const;
			cv::Mat get_infrared_frame() const;
            cv::Mat get_low_infrared_frame() const;
            cv::Mat get_rgb_frame() const;
            cv::Mat get_visualization_frame() const;
			cv::Mat get_android_sensor_frame() const;
			cv::Mat get_crossmotion_calibration_frame() const;
        };
    }
}
