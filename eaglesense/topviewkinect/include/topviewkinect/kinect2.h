
/**
Kinect v2 parameters

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <opencv2/core.hpp>

#include <climits>

namespace topviewkinect
{
    namespace kinect2
    {
        // Depth
        static const int DEPTH_WIDTH = 512;
        static const int DEPTH_HEIGHT = 424;
        static const int DEPTH_BUFFER_SIZE = DEPTH_WIDTH * DEPTH_HEIGHT;
        static const cv::Size CV_DEPTH_FRAME_SIZE(DEPTH_WIDTH, DEPTH_HEIGHT);

        static const int COLOR_WIDTH = 1920;
        static const int COLOR_HEIGHT = 1080;
        static const int COLOR_BUFFER_SIZE = COLOR_WIDTH * COLOR_HEIGHT;
        static const cv::Size CV_COLOR_FRAME_SIZE(COLOR_WIDTH, COLOR_HEIGHT);

        static const int COLOR_WIDTH_DOWNSAMPLED = COLOR_WIDTH / 2;
        static const int COLOR_HEIGHT_DOWNSAMPLED = COLOR_HEIGHT / 2;
        static const cv::Size CV_COLOR_FRAME_SIZE_DOWNSAMPLED(COLOR_WIDTH_DOWNSAMPLED, COLOR_HEIGHT_DOWNSAMPLED);

        // Infrared
        static const float INFRARED_SOURCE_MAX_VALUE = static_cast<float>(USHRT_MAX);
        static const float INFRARED_OUTPUT_MAX_VALUE = 1.0f;
        static const float INFRARED_OUTPUT_MIN_VALUE = 0.01f;
    }
}
