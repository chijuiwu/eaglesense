
/**
EagleSense skeleton

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"

#include <iostream>
#include <functional>

#include "topviewkinect/skeleton/skeleton.h"
#include "topviewkinect/vision/space.h"

namespace topviewkinect_vision = topviewkinect::vision;

namespace topviewkinect
{
    namespace skeleton
    {
        bool operator== (const Joint& lhs, const Joint& rhs)
        {
            return (lhs.x == rhs.x && lhs.y == rhs.y);
        }

        Skeleton::Skeleton(const int id) :
            id(id),
            activity("...")
        {
        }

        Skeleton::~Skeleton()
        {
        }

        int Skeleton::get_id() const
        {
            return this->id;
        }

        const bool Skeleton::is_updated() const
        {
            return this->updated;
        }

        void Skeleton::set_updated(bool updated)
        {
            this->updated = updated;
        }

        const bool Skeleton::is_activity_tracked() const
        {
            return this->activity_tracked;
        }

        void Skeleton::set_activity_tracked(bool activity_tracked)
        {
            this->activity_tracked = activity_tracked;
        }

        // Body
        const Joint Skeleton::get_body_center() const
        {
            return this->body_center;
        }

        void Skeleton::set_body_center(const Joint& body_center)
        {
            this->body_center = body_center;
        }

        const Joint Skeleton::get_head() const
        {
            return this->head;
        }

        void Skeleton::set_head(const Joint& head)
        {
            this->head = head;
        }

        const std::vector<cv::Point> Skeleton::get_depth_contour() const
        {
            return this->depth_contour;
        }

        void Skeleton::set_depth_contour(const std::vector<cv::Point>& depth_contour)
        {
            this->depth_contour = depth_contour;
        }

        const std::vector<cv::Point> Skeleton::get_contour() const
        {
            return this->contour;
        }

        void Skeleton::set_contour(const std::vector<cv::Point>& contour)
        {
            this->contour = contour;
        }

        const std::vector<cv::Point> Skeleton::get_head_contour() const
        {
            return this->head_contour;
        }

        void Skeleton::set_head_contour(const std::vector<cv::Point>& head_contour)
        {
            this->head_contour = head_contour;
        }

        const std::vector<cv::Point> Skeleton::get_body_contour() const
        {
            return this->body_contour;
        }

        void Skeleton::set_body_contour(const std::vector<cv::Point>& body_contour)
        {
            this->body_contour = body_contour;
        }

        const std::vector<cv::Point> Skeleton::get_bottom_contour() const
        {
            return this->bottom_contour;
        }

        void Skeleton::set_bottom_contour(const std::vector<cv::Point>& bottom_contour)
        {
            this->bottom_contour = bottom_contour;
        }

        const std::vector<cv::Point> Skeleton::get_others_contour() const
        {
            return this->others_contour;
        }

        void Skeleton::set_others_contour(const std::vector<cv::Point>& others_contour)
        {
            this->others_contour = others_contour;
        }

        // Features

        cv::Mat Skeleton::get_depth_silhouette() const
        {
            return this->depth_silhouette.clone();
        }

        void Skeleton::set_depth_silhouette(const cv::Mat& m)
        {
            m.copyTo(this->depth_silhouette);
        }

        cv::Mat Skeleton::get_infrared_silhouette() const
        {
            return this->infrared_silhouette.clone();
        }

        void Skeleton::set_infrared_silhouette(const cv::Mat& m)
        {
            m.copyTo(this->infrared_silhouette);
        }

        cv::Mat Skeleton::get_silhouette_kmeans() const
        {
            return this->silhouette_kmeans_color.clone();
        }

        void Skeleton::set_silhouette_kmeans(const cv::Mat& m)
        {
            m.copyTo(this->silhouette_kmeans_color);
        }

        cv::Mat Skeleton::get_silhouette_head_layer() const
        {
            return this->silhouette_head_layer.clone();
        }

        void Skeleton::set_silhouette_head_layer(const cv::Mat& m)
        {
            m.copyTo(this->silhouette_head_layer);
        }

        cv::Mat Skeleton::get_silhouette_body_layer() const
        {
            return this->silhouette_body_layer.clone();
        }

        void Skeleton::set_silhouette_body_layer(const cv::Mat& m)
        {
            m.copyTo(this->silhouette_body_layer);
        }

        cv::Mat Skeleton::get_silhouette_bottom_layer() const
        {
            return this->silhouette_bottom_layer.clone();
        }

        void Skeleton::set_silhouette_bottom_layer(const cv::Mat& m)
        {
            m.copyTo(this->silhouette_bottom_layer);
        }

        cv::Mat Skeleton::get_silhouette_others_layer() const
        {
            return this->silhouette_others_layer.clone();
        }

        void Skeleton::set_silhouette_others_layer(const cv::Mat& m)
        {
            m.copyTo(this->silhouette_others_layer);
        }

        const std::array<double, topviewkinect::vision::NUM_FEATURES> Skeleton::get_features() const
        {
            std::array<double, topviewkinect::vision::NUM_FEATURES> skeleton_features{};
            int feature_idx = 0;
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_LAYER_AREAS; ++i)
            {
                skeleton_features[feature_idx++] = this->f_layer_areas[i];
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_LAYER_CONTOURS; ++i)
            {
                skeleton_features[feature_idx++] = this->f_layer_contours[i];
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_LAYER_DISTANCES; ++i)
            {
                skeleton_features[feature_idx++] = this->f_layer_distances[i];
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_INTRALAYER_POSITIONS; ++i)
            {
                skeleton_features[feature_idx++] = this->f_intralayer_positions[i];
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_INTERLAYER_POSITIONS; ++i)
            {
                skeleton_features[feature_idx++] = this->f_interlayer_positions[i];
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES; ++i)
            {
                skeleton_features[feature_idx++] = this->f_body_extremities[i];
            }
            for (int i = 0; i < topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES_INFRAREDS; ++i)
            {
                skeleton_features[feature_idx++] = this->f_body_extremities_infrareds[i];
            }
            return skeleton_features;
        }

        void Skeleton::set_features(const std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_AREAS>& f_layer_areas, const std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_CONTOURS>& f_layer_contours, const std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_DISTANCES>& f_layer_distances, const std::array<double, topviewkinect::vision::FEATURE_NUM_INTRALAYER_POSITIONS>& f_intralayer_positions, const std::array<double, topviewkinect::vision::FEATURE_NUM_INTERLAYER_POSITIONS>& f_interlayer_positions, const std::array<double, topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES>& f_body_extremities, const std::array<double, topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES_INFRAREDS>& f_body_extremities_infrareds)
        {
            this->f_layer_areas = f_layer_areas;
            this->f_layer_contours = f_layer_contours;
            this->f_layer_distances = f_layer_distances;
            this->f_intralayer_positions = f_intralayer_positions;
            this->f_interlayer_positions = f_interlayer_positions;
            this->f_body_extremities = f_body_extremities;
            this->f_body_extremities_infrareds = f_body_extremities_infrareds;
        }

        // Visualizations
        cv::Mat Skeleton::get_mask() const
        {
            return this->mask.clone();
        }

        cv::Mat Skeleton::get_depth_frame() const
        {
            return this->depth_frame.clone();
        }

        cv::Mat Skeleton::get_infrared_frame() const
        {
            return this->infrared_frame.clone();
        }

        void Skeleton::set_mask(const cv::Mat& m)
        {
            m.copyTo(this->mask);
        }

        void Skeleton::set_depth_frame(const cv::Mat& m)
        {
            m.copyTo(this->depth_frame);
        }

        void Skeleton::set_infrared_frame(const cv::Mat& m)
        {
            m.copyTo(this->infrared_frame);
        }

        // Activity
        const int Skeleton::get_activity_id() const
        {
            return this->activity_id;
        }

        const std::string Skeleton::get_activity() const
        {
            return this->activity;
        }

        void Skeleton::set_activity_id(const int id)
        {
            this->activity_id = id;
        }


        void Skeleton::set_activity(const std::string &activity)
        {
            this->activity = activity;
        }

        // JSONify
        const std::string Skeleton::json() const
        {
            /*
            {
            "id": 0,
            "head": {
            "x": 0,
            "y": 0,
            "z": 0,
            "orientation": 0
            },
            "activity": ""
            },
            "activity_tracked": 0
            }
            */

            std::ostringstream skeleton_json;
            skeleton_json << "{";
            skeleton_json << "\"id\": " << this->id << ",";
            skeleton_json << "\"head\": {";
            skeleton_json << "\"x\": " << this->head.x << ",";
            skeleton_json << "\"y\": " << this->head.y << ",";
            skeleton_json << "\"z\": " << (int)this->head.z << ",";
            skeleton_json << "\"orientation\": " << this->head.orientation << "},";
            skeleton_json << "\"activity\": \"" << this->activity << "\",";
            skeleton_json << "\"activity_tracked\": " << static_cast<int>(this->activity_tracked) << "";
            skeleton_json << "}";
            return skeleton_json.str();
        }
    }
}
