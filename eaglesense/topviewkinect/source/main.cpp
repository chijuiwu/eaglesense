
/**
EagleSense top-view tracking application

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <boost/program_options.hpp>

#include <string>
#include <sstream>
#include <vector>

#include "topviewkinect/topviewkinect.h"
#include "topviewkinect/color.h"
#include "topviewkinect/util.h"
#include "topviewkinect/vision/space.h"

static constexpr const char* DEPTH_WINDOW_NAME = "Top-view Kinect Depth";
static constexpr const char* INFRARED_WINDOW_NAME = "Top-view Kinect Infrared";
static constexpr const char* RGB_WINDOW_NAME = "Top-view Kinect RGB";
static constexpr const char* SPACE_WINDOW_NAME = "Top-view Interactive Space";

static int start();
static int replay(const int dataset_id);
static int capture(const int dataset_id);
static int postprocess(const int dataset_id, const std::string& dataset_name, const bool relabel);

int main(int argc, char* argv[])
{
    // Update EagleSense root directory
    std::string args = *argv;
    std::vector<std::string> delimited = topviewkinect::util::string_split(args, '\\');
    std::ostringstream eaglesense_directory_ss;
    for (auto i = delimited.begin(); i != delimited.end(); ++i)
    {
        std::string dir = *i;
        eaglesense_directory_ss << dir << "/";
        if (dir == "eaglesense")
        {
            break;
        }
    }
    topviewkinect::set_eaglesense_directory(eaglesense_directory_ss.str());

    // Define program options
    boost::program_options::options_description all_opts("Help");

    boost::program_options::options_description general_opts("General");
    general_opts.add_options()
        ("version,v", "Version")
        ("help,h", "Help");

    boost::program_options::options_description advanced_opts("Advanced (w/ Datasets)");
    int dataset_id;
    std::string dataset_name;
    advanced_opts.add_options()
        ("replay,r", "Replay")
        ("capture,c", "Capture")
        ("postprocess,p", "Postprocess")
        ("features,f", "Postprocess (features ONLY)")
        ("dataset_id,d", boost::program_options::value<int>(&dataset_id), "Dataset ID (Required for advanced options)")
        ("dataset_name,n", boost::program_options::value<std::string>(&dataset_name)->default_value("Untitled"), "Dataset name");

    all_opts.add(general_opts).add(advanced_opts);

    // Parse program options
    boost::program_options::variables_map program_options;
    try
    {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, all_opts), program_options);
        boost::program_options::notify(program_options);
    }
    catch (boost::program_options::error& e)
    {
        std::cout << "Invalid." << " " << e.what() << std::endl;
        std::cout << all_opts << std::endl;
        return EXIT_FAILURE;
    }

    // Start
    std::cout << "EagleSense topviewkinect " << topviewkinect::VERSION << std::endl;

    // -v
    if (program_options.count("version"))
    {
        return EXIT_SUCCESS;
    }

    // -h
    if (program_options.count("help"))
    {
        std::cout << all_opts << std::endl;
        return EXIT_SUCCESS;
    }

    if (argc == 1)
    {
        return start();
    }

    // Advanced (w/ Datasets)

    // Check dataset id
    if (!program_options.count("dataset_id"))
    {
        std::cout << "Missing dataset ID." << std::endl;
        return EXIT_FAILURE;
    }

    // -r
    if (program_options.count("replay"))
    {
        return replay(dataset_id);
    }

    // -c
    if (program_options.count("capture"))
    {
        return capture(dataset_id);
    }
    
    // -p
    if (program_options.count("postprocess"))
    {
        bool relabel = program_options.count("features") == 0 ? true : false;
        return postprocess(dataset_id, dataset_name, relabel);
    }

    return EXIT_SUCCESS;
}

static int start()
{
    topviewkinect::util::log_println("Tracking!!!");

    // Create interactive space
    topviewkinect::vision::TopViewSpace m_space;
    bool space_initialized = m_space.initialize();
    if (!space_initialized)
    {
        topviewkinect::util::log_println("Failed. Exiting...");
        return EXIT_FAILURE;
    }

    // Tracking
    while (true)
    {
        cv::imshow(DEPTH_WINDOW_NAME, m_space.get_depth_frame());
        cv::imshow(INFRARED_WINDOW_NAME, m_space.get_infrared_frame());
        cv::imshow(SPACE_WINDOW_NAME, m_space.get_visualization_frame());

        int ascii_keypress = cv::waitKey(1); // any key to exit
        if (ascii_keypress != -1)
        {
            topviewkinect::util::log_println("System terminating...");
            break;
        }

        bool frame_received = m_space.refresh_kinect_frames();
        if (frame_received)
        {
            m_space.process_kinect_frames();
        }
    }

    topviewkinect::util::log_println("Done!!!");
    return EXIT_SUCCESS;
}

static int replay(const int dataset_id)
{
    topviewkinect::util::log_println("Replaying!!!");

    // Create interactive space
    topviewkinect::vision::TopViewSpace m_space;
    bool space_initialized = m_space.initialize();
    if (!space_initialized)
    {
        topviewkinect::util::log_println("Failed. Exiting...");
        return EXIT_FAILURE;
    }

    // Load dataset
    bool dataset_loaded = m_space.load_dataset(dataset_id);
    if (!dataset_loaded)
    {
        topviewkinect::util::log_println("Failed. Exiting...");
        return EXIT_FAILURE;
    }

    // Replay
    while (true)
    {
        cv::Mat depth_frame = m_space.get_depth_frame();
        cv::Mat infrared_frame = m_space.get_infrared_frame();
        cv::Mat visualization_frame = m_space.get_visualization_frame();

        // Visualize tracking
        const int frame_id = m_space.get_kinect_frame_id();
        cv::putText(depth_frame, std::to_string(frame_id), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_WHITE);
        cv::putText(infrared_frame, std::to_string(frame_id), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_WHITE);
        cv::putText(visualization_frame, std::to_string(frame_id), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_BGR_WHITE);
        cv::imshow(DEPTH_WINDOW_NAME, depth_frame);
        cv::imshow(INFRARED_WINDOW_NAME, infrared_frame);
        cv::imshow(SPACE_WINDOW_NAME, visualization_frame);

        int ascii_keypress = cv::waitKey(0);
        if (ascii_keypress == 2555904) // right arrow
        {
            m_space.replay_next_frame();
        }
        else if (ascii_keypress == 2424832) // left arrow
        {
            m_space.replay_previous_frame();
        }
        else // anything else
        {
            break;
        }
    }

    topviewkinect::util::log_println("Done!!!");
    return EXIT_SUCCESS;
}

static int capture(const int dataset_id)
{
    topviewkinect::util::log_println("Capturing!!!");

    // Create interactive space
    topviewkinect::vision::TopViewSpace m_space;
    bool space_initialized = m_space.initialize();
    if (!space_initialized)
    {
        topviewkinect::util::log_println("Failed. Exiting...");
        return EXIT_FAILURE;
    }

    // Create dataset
    bool dataset_created = m_space.create_dataset(dataset_id);
    if (!dataset_created)
    {
        topviewkinect::util::log_println("Failed. Exiting...");
        return EXIT_FAILURE;
    }

    // Capture
    while (true)
    {
        cv::imshow(DEPTH_WINDOW_NAME, m_space.get_depth_frame());
        cv::imshow(INFRARED_WINDOW_NAME, m_space.get_infrared_frame());
        cv::imshow(SPACE_WINDOW_NAME, m_space.get_visualization_frame());

        int ascii_keypress = cv::waitKey(30); // 30 fps
        if (ascii_keypress != -1)
        {
            topviewkinect::util::log_println("System terminating...");
            break;
        }

        bool frame_received = m_space.refresh_kinect_frames();
        if (frame_received)
        {
            m_space.save_kinect_frames();
        }
    }

    topviewkinect::util::log_println("Done!!!");
    return EXIT_SUCCESS;
}

static int postprocess(const int dataset_id, const std::string& dataset_name, const bool relabel)
{
    topviewkinect::util::log_println("Postprocessing!!!");
    !relabel ? topviewkinect::util::log_println("**Features Only**") : 0;

    // Create interactive space
    topviewkinect::vision::TopViewSpace m_space;
    // Skip tracking initialization

    // Load dataset
    bool dataset_loaded = m_space.load_dataset(dataset_id);
    if (!dataset_loaded)
    {
        topviewkinect::util::log_println("Failed. Exiting...");
        return EXIT_FAILURE;
    }

    // Postprocess
    m_space.postprocess(dataset_name, relabel);

    topviewkinect::util::log_println("Done!!!");
    return EXIT_SUCCESS;
}