
/**
EagleSense framerate

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <windows.h>

namespace topviewkinect
{
    namespace vision
    {
        class FrameRateController
        {
        private:
            INT64 num_last_counter;
            DWORD num_frames_since_update;
            double frame_frequency;

        public:
            FrameRateController();
            ~FrameRateController() {}

            double get_fps();
        };
    }
}
