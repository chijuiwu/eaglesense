
/**
EagleSense data

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <opencv2/core.hpp>

#include <vector>
#include <fstream>

#include "topviewkinect/skeleton/skeleton.h"

namespace topviewkinect
{
    namespace vision
    {
        class InteractionLog
        {
        private:
            // Dataset
            int dataset_id;
            int dataset_size;
            std::string dataset_directory;

            // Data
            std::string depth_directory;
            std::string infrared_directory;
            std::string low_infrared_directory;
            std::string rgb_directory;

            // TODO: real-time tracking log
            //std::ofstream tracking_csv;

            // Replay
            std::map<int, std::tuple<std::string, std::string>> dataset_frames;
            std::map<int, std::tuple<std::string, std::string>>::iterator dataset_frames_it;

            // Capture
            std::ofstream timeseries_csv;

            // Postprocess
            std::ofstream features_csv;
            std::ofstream labels_csv;
            std::ofstream description_json;
            std::ofstream processing_csv;

        public:
            InteractionLog();
            ~InteractionLog();

            const int get_size() const;

            void initialize(const int dataset_id);
            // TODO: real-time tracking log
            //void output_tracking_state(const int frame_id, const int fps, const std::vector<topviewkinect::skeleton::Skeleton>& skeletons);

            // Replay
            bool load_directories();
            const std::map<int, std::tuple<std::string, std::string>> get_dataset_frames() const;
            const std::pair<int, std::tuple<std::string, std::string>> current_frames();
            const std::pair<int, std::tuple<std::string, std::string>> next_frames();
            const std::pair<int, std::tuple<std::string, std::string>> previous_frames();

            // Capture
            bool create_directories();
            void save_multisource_frames(signed long long depth_frame_timestamp, const cv::Mat& depth_frame, signed long long infrared_frame_timestamp = -1, const cv::Mat& infrared_frame = cv::Mat(), const cv::Mat& low_infrared_frame = cv::Mat(), signed long long rgb_frame_timestamp = -1, const cv::Mat& rgb_frame = cv::Mat());
            void save_visualization(const int frame_id, const cv::Mat& visualization_frame);

            // Postprocess
            void create_postprocessed_files(const bool relabel);
            void output_skeleton_features(const int frame_id, const std::vector<topviewkinect::skeleton::Skeleton>& skeletons, const bool keep_label);
            void output_processing_time(const int frame_id, const long long features_time, const long long total_time);
            void output_description(const std::string& dataset_name);
        };
    }
}
