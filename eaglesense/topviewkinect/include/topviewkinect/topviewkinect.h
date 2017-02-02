
/**
EagleSense top-view processing pipeline

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <string>

namespace topviewkinect
{
    // EagleSense version
    static const std::string VERSION = "v1.0";

    // EagleSense directories
    extern std::string EAGLESENSE_DIRECTORY;
    void set_eaglesense_directory(const std::string& directory);
    std::string get_config_filepath();
    std::string get_dataset_directory(const int dataset_id);
    std::string get_model_filepath(const std::string& model);

    // EagleSense configuration
    struct Configuration
    {
        bool framerate;
        bool orientation_recognition;
        bool interaction_recognition;
        bool restful_connection;

        std::string interaction_model;

        std::string restful_server_address;
        int restful_server_port;

        bool depth;
        bool infrared;
        bool color;
    };
}
