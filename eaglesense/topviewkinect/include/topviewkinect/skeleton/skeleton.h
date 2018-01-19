
/**
EagleSense skeleton

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

#include "topviewkinect/skeleton/geodesic.h"
#include "topviewkinect/vision/parameters.h"

namespace topviewkinect
{
    namespace skeleton
    {
        struct Joint {
            int x;
            int y;
            int z;
            float orientation;
        };

        bool operator== (const Joint& lhs, const Joint& rhs);

        class Skeleton
        {
        private:
            int id;
            bool updated = false;
            bool activity_tracked = false;

            // Body
            Joint body_center;
            Joint head;
            std::vector<cv::Point> contour;

            // Features
            cv::Mat depth_silhouette;
            cv::Mat infrared_silhouette;
            cv::Mat silhouette_kmeans_color;
            cv::Mat silhouette_head_layer;
            cv::Mat silhouette_body_layer;
            cv::Mat silhouette_bottom_layer;
            cv::Mat silhouette_others_layer;
            std::vector<cv::Point> depth_contour;
            std::vector<cv::Point> head_contour;
            std::vector<cv::Point> body_contour;
            std::vector<cv::Point> bottom_contour;
            std::vector<cv::Point> others_contour;

            std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_AREAS> f_layer_areas;
            std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_CONTOURS> f_layer_contours;
            std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_DISTANCES> f_layer_distances;
            std::array<double, topviewkinect::vision::FEATURE_NUM_INTRALAYER_POSITIONS> f_intralayer_positions;
            std::array<double, topviewkinect::vision::FEATURE_NUM_INTERLAYER_POSITIONS> f_interlayer_positions;
            std::array<double, topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES> f_body_extremities;
            std::array<double, topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES_INFRAREDS> f_body_extremities_infrareds;

            // Visualizations
            cv::Mat mask;
            cv::Mat depth_frame;
            cv::Mat infrared_frame;

            // Activity
            int activity_id;
            std::string activity;

        public:
            Skeleton(const int id);
            ~Skeleton();

            int get_id() const;
            const bool is_updated() const;
            void set_updated(bool updated);
            const bool is_activity_tracked() const;
            void set_activity_tracked(bool updated);

            // Joints
            double width;
            double height;

            const Joint get_body_center() const;
            void set_body_center(const Joint& body_center);
            const std::vector<cv::Point> get_contour() const;
            void set_contour(const std::vector<cv::Point>& contour);
            const Joint get_head() const;
            void set_head(const Joint& head);

            // Features
            cv::Mat get_depth_silhouette() const;
            void set_depth_silhouette(const cv::Mat& m);
            cv::Mat get_infrared_silhouette() const;
            void set_silhouette_kmeans(const cv::Mat& m);
            cv::Mat get_silhouette_kmeans() const;
            void set_infrared_silhouette(const cv::Mat& m);
            cv::Mat get_silhouette_head_layer() const;
            void set_silhouette_head_layer(const cv::Mat& m);
            cv::Mat get_silhouette_body_layer() const;
            void set_silhouette_body_layer(const cv::Mat& m);
            cv::Mat get_silhouette_bottom_layer() const;
            void set_silhouette_bottom_layer(const cv::Mat& m);
            cv::Mat get_silhouette_others_layer() const;
            void set_silhouette_others_layer(const cv::Mat& m);
            const std::vector<cv::Point> get_depth_contour() const;
            void set_depth_contour(const std::vector<cv::Point>& contour);
            const std::vector<cv::Point> get_head_contour() const;
            void set_head_contour(const std::vector<cv::Point>& contour);
            const std::vector<cv::Point> get_body_contour() const;
            void set_body_contour(const std::vector<cv::Point>& contour);
            const std::vector<cv::Point> get_bottom_contour() const;
            void set_bottom_contour(const std::vector<cv::Point>& contour);
            const std::vector<cv::Point> get_others_contour() const;
            void set_others_contour(const std::vector<cv::Point>& contour);

            const std::array<double, topviewkinect::vision::NUM_FEATURES> get_features() const;
            void set_features(const std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_AREAS>& f_layer_areas, const std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_CONTOURS>& f_layer_contours, const std::array<double, topviewkinect::vision::FEATURE_NUM_LAYER_DISTANCES>& f_layer_distances, const std::array<double, topviewkinect::vision::FEATURE_NUM_INTRALAYER_POSITIONS>& f_intralayer_positions, const std::array<double, topviewkinect::vision::FEATURE_NUM_INTERLAYER_POSITIONS>& f_interlayer_positions, const std::array<double, topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES>& f_body_extremities, const std::array<double, topviewkinect::vision::FEATURE_NUM_BODY_EXTREMITIES_INFRAREDS>& f_body_extremities_infrareds);

            // Visualizations
            cv::Mat get_mask() const;
            cv::Mat get_depth_frame() const;
            cv::Mat get_infrared_frame() const;
            void set_mask(const cv::Mat& m);
            void set_depth_frame(const cv::Mat& m);
            void set_infrared_frame(const cv::Mat& m);

            // Activity
            const int get_activity_id() const;
            const std::string get_activity() const;
            void set_activity_id(const int id);
            void set_activity(const std::string& activity);

            // JSONify
            const std::string json() const;
        };
    }
}
