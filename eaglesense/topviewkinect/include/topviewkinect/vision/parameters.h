
/**
EagleSense parameters

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/core.hpp>

namespace topviewkinect
{
    namespace vision
    {
        // Kinect FOV
        static const double area_cm = 4 * 250 * 250 * ((std::tan(35 * M_PI / 180) * std::tan(30 * M_PI / 180)) / (std::cos(30 * M_PI / 180) * std::cos(35 * M_PI / 180)));
        static const double area_pixel_ratio = area_cm / (512 * 424);

        // Background subtraction
        static const int REQUIRED_BACKGROUND_FRAMES = 120;
        static const int FOREGROUND_MEDIAN_FILTER_SIZE = 5;

        // Skeleton
        static const double BODY_CONTOUR_MIN_AREA = area_pixel_ratio * M_PI * 50 * 50;

        // Activity zone
        static const double ACTIVITY_ZONE = area_pixel_ratio * 15;

        // Skeleton features
        static const int BODY_MAX_NUM_LAYERS = 3;

        static const int FEATURE_NUM_LAYER_AREAS = 3;
        static const int FEATURE_NUM_LAYER_CONTOURS = 2;
        static const int FEATURE_NUM_LAYER_DISTANCES = 15;
        static const int FEATURE_NUM_INTRALAYER_POSITIONS = 27;
        static const int FEATURE_NUM_INTERLAYER_POSITIONS = 18;
        static const int FEATURE_NUM_BODY_EXTREMITIES = 1;
        static const int FEATURE_NUM_BODY_EXTREMITIES_INFRAREDS = 6;

        static const int NUM_TOTAL_FEATURES = FEATURE_NUM_LAYER_AREAS + FEATURE_NUM_LAYER_CONTOURS + FEATURE_NUM_LAYER_DISTANCES + FEATURE_NUM_INTRALAYER_POSITIONS + FEATURE_NUM_INTERLAYER_POSITIONS + FEATURE_NUM_BODY_EXTREMITIES + FEATURE_NUM_BODY_EXTREMITIES_INFRAREDS;
    }
}