
/**
EagleSense tracking application visualization

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright ?2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <opencv2/core.hpp>

namespace topviewkinect
{
    namespace color
    {
        static const cv::Scalar CV_WHITE(255);
        static const cv::Scalar CV_BLACK(0);
        static const cv::Scalar CV_BGR_WHITE(255, 255, 255);
        static const cv::Scalar CV_BGR_BLACK(0, 0, 0);
        static const cv::Scalar CV_BGR_GREY(150, 150, 150);
        static const cv::Scalar CV_BGR_RED(82, 68, 204);

        static const float SKELETON_OVERLAY_ALPHA = 0.5f;
        static const std::vector<cv::Scalar> CV_BGR_SKELETON{ cv::Scalar(217,161,0), cv::Scalar(241,90,90), cv::Scalar(78,186,111), cv::Scalar(149,91,165), cv::Scalar(240,196,25), cv::Scalar(238,62,32) };
        static const std::vector<cv::Scalar> CV_BGR_LAYERS{ cv::Scalar(76,69,62), cv::Scalar(197,133,33), cv::Scalar(82,68,204), cv::Scalar(241,90,90) };
    }
}