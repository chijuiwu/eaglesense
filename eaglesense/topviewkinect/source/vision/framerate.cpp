
/**
EagleSense framerate

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"

#include "topviewkinect/vision/framerate.h"

namespace topviewkinect
{
    namespace vision
    {
        FrameRateController::FrameRateController() :
            num_last_counter(0),
            num_frames_since_update(0),
            frame_frequency(0)
        {
            LARGE_INTEGER qpf = { 0 };
            if (QueryPerformanceFrequency(&qpf))
            {
                this->frame_frequency = double(qpf.QuadPart);
            }
        }

        double FrameRateController::get_fps()
        {
            double fps = 0.0;
            LARGE_INTEGER qpcNow = { 0 };
            if (this->frame_frequency)
            {
                if (QueryPerformanceCounter(&qpcNow))
                {
                    if (this->num_last_counter)
                    {
                        this->num_frames_since_update++;
                        fps = this->frame_frequency * this->num_frames_since_update / double(qpcNow.QuadPart - this->num_last_counter);
                    }
                }
            }
            this->num_last_counter = qpcNow.QuadPart;
            this->num_frames_since_update = 0;
            return fps;
        }
    }
}
