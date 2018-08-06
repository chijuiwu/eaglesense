
/**
EagleSense top-view processing pipeline

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"

#include "topviewkinect/topviewkinect.h"

namespace topviewkinect
{
    std::string EAGLESENSE_DIRECTORY = "";

    void set_eaglesense_directory(const std::string& directory)
    {
        EAGLESENSE_DIRECTORY = directory;
    }

    std::string get_config_filepath()
    {
        return EAGLESENSE_DIRECTORY + "/config.json";
    }

    std::string get_dataset_directory(const int dataset_id)
    {
		//return EAGLESENSE_DIRECTORY + "/../eaglesense-exp/data/topviewkinect/" + std::to_string(dataset_id);
        return EAGLESENSE_DIRECTORY + "/data/topviewkinect/" + std::to_string(dataset_id);
    }
    
    std::string get_model_filepath(const std::string& model)
    {
        return EAGLESENSE_DIRECTORY + "/models/" + model;
    }
}
