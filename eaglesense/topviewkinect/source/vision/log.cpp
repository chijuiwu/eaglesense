
/**
EagleSense data

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"

#include <opencv2/highgui.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/filesystem.hpp>

#include <numeric>

#include "topviewkinect/topviewkinect.h"
#include "topviewkinect/util.h"
#include "topviewkinect/vision/log.h"

namespace topviewkinect
{
    namespace vision
    {
        InteractionLog::InteractionLog()
        {
        }

        InteractionLog::~InteractionLog()
        {
            this->timeseries_csv.close();
            this->features_csv.close();
            this->labels_csv.close();
            this->description_json.close();
        }

        static const std::string get_depth_directory(const std::string& dataset_directory)
        {
            return dataset_directory + "/depth";
        }

        static const std::string get_infrared_directory(const std::string& dataset_directory)
        {
            return dataset_directory + "/infrared";
        }

        static const std::string get_low_infrared_directory(const std::string& dataset_directory)
        {
            return dataset_directory + "/low_infrared";
        }

        static const std::string get_rgb_directory(const std::string& dataset_directory)
        {
            return dataset_directory + "/rgb";
        }

        static const std::string get_visualization_directory(const std::string& dataset_directory)
        {
            return dataset_directory + "/visualization";
        }

        const int InteractionLog::get_size() const
        {
            return this->dataset_size;
        }

        void InteractionLog::initialize(const int dataset_id)
        {
            this->dataset_id = dataset_id;
            this->dataset_size = 0;
            this->dataset_directory = topviewkinect::get_dataset_directory(dataset_id);

            this->depth_directory = get_depth_directory(this->dataset_directory);
            this->infrared_directory = get_infrared_directory(this->dataset_directory);
            this->low_infrared_directory = get_low_infrared_directory(this->dataset_directory);
            this->rgb_directory = get_rgb_directory(this->dataset_directory);
        }

        // TODO: real-time tracking log
        //void InteractionLog::output_tracking_state(const int frame_id, const int fps, const std::vector<topviewkinect::skeleton::Skeleton>& skeletons)
        //{
        //    this->tracking_csv << frame_id << "," << fps;
        //    for (int i = 0; i < 6; ++i)
        //    {
        //        if (i + 1 > skeletons.size())
        //        {
        //            this->tracking_csv << ",-1,-1,-1" << std::endl;
        //        }
        //        else
        //        {
        //            const topviewkinect::skeleton::Joint head = skeletons[i].get_head();
        //            const int activity_id = skeletons[i].get_activity_id();
        //            this->tracking_csv << "," << head.x << "," << head.y << "," << activity_id << std::endl;
        //        }
        //    }
        //}

        // Replay

        bool InteractionLog::load_directories()
        {
            if (!boost::filesystem::is_directory(this->depth_directory))
            {
                topviewkinect::util::log_println("No depth directory found @ " + this->depth_directory);
                return false;
            }
            if (!boost::filesystem::is_directory(this->low_infrared_directory))
            {
                topviewkinect::util::log_println("No infrared directory found @ " + this->low_infrared_directory);
                return false;
            }

            this->dataset_frames.clear();

            // Load dataset
            for (auto i = boost::filesystem::directory_iterator(this->depth_directory); i != boost::filesystem::directory_iterator(); i++)
            {
                if (boost::filesystem::is_regular_file(i->path()))
                {
                    std::string depth_frame_filepath = i->path().string();

                    int frame_id = std::stoi(depth_frame_filepath.substr(depth_frame_filepath.find_last_of("/\\") + 1, depth_frame_filepath.find(".jpeg")));

                    std::ostringstream infrared_frame_filepath_ss;
                    infrared_frame_filepath_ss << this->low_infrared_directory << "/" << frame_id << ".jpeg";
                    std::string infrared_frame_filepath = infrared_frame_filepath_ss.str();

                    this->dataset_frames[frame_id] = std::make_tuple(depth_frame_filepath, infrared_frame_filepath);
                }
            }
            this->dataset_frames_it = this->dataset_frames.begin();

            this->processing_csv = std::ofstream(this->dataset_directory + "/processing.csv");
            this->processing_csv << "frame_id,features_time,total_time" << std::endl;

            return true;
        }

        const std::map<int, std::tuple<std::string, std::string>> InteractionLog::get_dataset_frames() const
        {
            return this->dataset_frames;
        }

        const std::pair<int, std::tuple<std::string, std::string>> InteractionLog::current_frames()
        {
            return std::make_pair(this->dataset_frames_it->first, this->dataset_frames_it->second);
        }

        const std::pair<int, std::tuple<std::string, std::string>> InteractionLog::next_frames()
        {
            if (!(this->dataset_frames_it != this->dataset_frames.end() && this->dataset_frames_it == --this->dataset_frames.end()))
            {
                ++this->dataset_frames_it;
            }
            return std::make_pair(this->dataset_frames_it->first, this->dataset_frames_it->second);
        }

        const std::pair<int, std::tuple<std::string, std::string>> InteractionLog::previous_frames()
        {
            if (this->dataset_frames_it != this->dataset_frames.begin())
            {
                --this->dataset_frames_it;
            }
            return std::make_pair(this->dataset_frames_it->first, this->dataset_frames_it->second);
        }

        // Capture

        bool InteractionLog::create_directories()
        {
            if (boost::filesystem::is_directory(this->dataset_directory))
            {
                topviewkinect::util::log_println("Dataset exists already @ " + this->dataset_directory);
                return false;
            }

            boost::filesystem::create_directory(boost::filesystem::path(this->dataset_directory));
            boost::filesystem::create_directory(boost::filesystem::path(this->depth_directory));
            boost::filesystem::create_directory(boost::filesystem::path(this->infrared_directory));
            boost::filesystem::create_directory(boost::filesystem::path(this->low_infrared_directory));
            boost::filesystem::create_directory(boost::filesystem::path(this->rgb_directory));

            this->timeseries_csv = std::ofstream(this->dataset_directory + "/timeseries.csv");
            this->timeseries_csv << "frame_id,depth_time,infrared_time,rgb_time" << std::endl;

            return true;
        }

        void InteractionLog::save_multisource_frames(signed long long depth_frame_timestamp, const cv::Mat& depth_frame, signed long long infrared_frame_timestamp, const cv::Mat& infrared_frame, const cv::Mat& low_infrared_frame, signed long long rgb_frame_timestamp, const cv::Mat& rgb_frame)
        {
            if (!depth_frame.empty())
            {
                std::ostringstream depth_image_ss;
                depth_image_ss << this->depth_directory << "/" << this->dataset_size << ".jpeg";
                cv::imwrite(depth_image_ss.str(), depth_frame);
            }

            if (!infrared_frame.empty())
            {
                std::ostringstream infrared_image_ss;
                infrared_image_ss << this->infrared_directory << "/" << this->dataset_size << ".jpeg";
                cv::imwrite(infrared_image_ss.str(), infrared_frame);
            }

            if (!low_infrared_frame.empty())
            {
                std::ostringstream infrared_image_ss;
                infrared_image_ss << this->low_infrared_directory << "/" << this->dataset_size << ".jpeg";
                cv::imwrite(infrared_image_ss.str(), low_infrared_frame);
            }

            if (!rgb_frame.empty())
            {
                std::ostringstream rgb_image_ss;
                rgb_image_ss << this->rgb_directory << "/" << this->dataset_size << ".jpeg";
                cv::imwrite(rgb_image_ss.str(), rgb_frame);
            }

            this->timeseries_csv << this->dataset_size << "," << depth_frame_timestamp << "," << infrared_frame_timestamp << "," << rgb_frame_timestamp << std::endl;

            topviewkinect::util::log_println("Captured " + this->dataset_size);
            ++this->dataset_size;
        }

        void InteractionLog::save_visualization(const int frame_id, const cv::Mat& visualization_frame)
        {
            std::ostringstream visualization_image_ss;
            visualization_image_ss << get_visualization_directory(this->dataset_directory) << "/" << frame_id << ".jpeg";
            cv::imwrite(visualization_image_ss.str(), visualization_frame);
        }

        // Postprocess

        void InteractionLog::create_postprocessed_files(const bool relabel)
        {
            // Create features.csv
            this->features_csv = std::ofstream(this->dataset_directory + "/features.csv");
            this->features_csv << "frame_id,skeleton_id,x,y,z,";

            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_LAYER_AREAS; ++i)
            {
                this->features_csv << "layer_area_" << i << ",";
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_LAYER_CONTOURS; ++i)
            {
                this->features_csv << "layer_contours_" << i << ",";
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_LAYER_DISTANCES; ++i)
            {
                this->features_csv << "layer_distance_" << i << ",";
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_INTRALAYER_POSITIONS; ++i)
            {
                this->features_csv << "intralayer_pos_" << i << ",";
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_INTERLAYER_POSITIONS; ++i)
            {
                this->features_csv << "interlayer_pos_" << i << ",";
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES; ++i)
            {
                this->features_csv << "extremities" << i << ",";
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES_INFRAREDS; ++i)
            {
                this->features_csv << "extreme_infrared_" << i;
                if (i + 1 < topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES_INFRAREDS)
                {
                    this->features_csv << ",";
                }
            }
            this->features_csv << std::endl;

            // Create labels.csv
            if (relabel)
            {
                this->labels_csv = std::ofstream(this->dataset_directory + "/labels.csv");
                this->labels_csv << "frame_id,skeleton_id,activity,orientation,orientation_accurate" << std::endl;
            }
        }

        void InteractionLog::output_skeleton_features(const int frame_id, const std::vector<topviewkinect::skeleton::Skeleton>& skeletons, const bool relabel)
        {
            for (const topviewkinect::skeleton::Skeleton& skeleton : skeletons)
            {
                const topviewkinect::skeleton::Joint head = skeleton.get_head();

                // Features
                if (skeleton.is_activity_tracked())
                {
                    const std::array<double, topviewkinect::vision::NUM_FEATURES> skeleton_features = skeleton.get_features();

                    this->features_csv << frame_id << "," << skeleton.get_id() << "," << head.x << "," << head.y << "," << head.z << ",";
                    for (size_t i = 0; i < skeleton_features.size(); ++i)
                    {
                        this->features_csv << skeleton_features[i];
                        if (i + 1 < skeleton_features.size())
                        {
                            this->features_csv << ",";
                        }
                    }
                    this->features_csv << std::endl;
                }

                // Label
                if (relabel)
                {
                    this->labels_csv << frame_id << "," << skeleton.get_id() << ",-1," << std::setprecision(2) << head.orientation << ",-1" << std::endl;
                }
            }

            if (skeletons.size() == 0 && relabel)
            {
                this->labels_csv << frame_id << ",-1,-1,-1,-1" << std::endl;
            }

            ++this->dataset_size;
        }

        void InteractionLog::output_processing_time(const int frame_id, const long long features_time, const long long total_time)
        {
            this->processing_csv << frame_id << "," << features_time << "," << total_time << std::endl;
        }

        void InteractionLog::output_description(const std::string& dataset_name)
        {
            this->description_json = std::ofstream(this->dataset_directory + "/description.json");
            this->description_json << "{" << std::endl;
            this->description_json << "    \"id\": " << this->dataset_id << "," << std::endl;
            this->description_json << "    \"name\": \"" << dataset_name << "\"," << std::endl;
            this->description_json << "    \"type\": \"Activity and Device Recognition\"," << std::endl;
            this->description_json << "    \"datetime\": \"" << topviewkinect::util::get_current_datetime() << "\"," << std::endl;
            this->description_json << "    \"size\": " << this->dataset_size << "," << std::endl;
            this->description_json << "    \"labels\": [" << std::endl;
            this->description_json << "        {" << std::endl;
            this->description_json << "            \"name\": \"Standing\"" << std::endl;
            this->description_json << "        }," << std::endl;
            this->description_json << "        {" << std::endl;
            this->description_json << "            \"name\": \"Sitting\"" << std::endl;
            this->description_json << "        }," << std::endl;
            this->description_json << "        {" << std::endl;
            this->description_json << "            \"name\": \"Pointing\"" << std::endl;
            this->description_json << "        }," << std::endl;
            this->description_json << "        {" << std::endl;
            this->description_json << "            \"name\": \"Phone\"" << std::endl;
            this->description_json << "        }," << std::endl;
            this->description_json << "        {" << std::endl;
            this->description_json << "            \"name\": \"Tablet\"" << std::endl;
            this->description_json << "        }," << std::endl;
            this->description_json << "        {" << std::endl;
            this->description_json << "            \"name\": \"Paper\"" << std::endl;
            this->description_json << "        }," << std::endl;
            this->description_json << "        {" << std::endl;
            this->description_json << "            \"name\": \"Empty\"" << std::endl;
            this->description_json << "        }" << std::endl;
            this->description_json << "    ]" << std::endl;
            this->description_json << "}" << std::endl;
        }
    }
}
