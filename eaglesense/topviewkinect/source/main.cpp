
/**
EagleSense

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright � 2016 Chi-Jui Wu

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
#include "topviewkinect/kinect2.h"
#include "topviewkinect/color.h"
#include "topviewkinect/util.h"
#include "topviewkinect/vision/space.h"

static constexpr const char* DEPTH_WINDOW_NAME = "Top-View Kinect Depth";
static constexpr const char* INFRARED_WINDOW_NAME = "Top-View Kinect Infrared";
static constexpr const char* INFRARED_LOW_WINDOW_NAME = "Top-View Kinect Infrared (Low)";
static constexpr const char* RGB_WINDOW_NAME = "Top-View Kinect RGB";
static constexpr const char* SPACE_WINDOW_NAME = "Top-View Interactive Space";

static int start();
static int replay(const int dataset_id);
static int capture(const int dataset_id);
static int postprocess(const int dataset_id, const std::string& dataset_name, const int dataset_label, const bool keep_label);

int main(int argc, char* argv[])
{
    // Update EagleSense root directory
    std::string args = *argv;
    std::vector<std::string> delimited = topviewkinect::util::string_split(args, '\\');
    std::ostringstream eaglesense_directory_ss;
    for (auto dir = delimited.begin(); dir != delimited.end(); ++dir)
    {
        eaglesense_directory_ss << *dir;
        if (*dir == "eaglesense")
        {
            break;
        }
        eaglesense_directory_ss << "/";
    }
    topviewkinect::set_eaglesense_directory(eaglesense_directory_ss.str());

    // Define program options
    boost::program_options::options_description all_opts("Help");

    boost::program_options::options_description general_opts("General");
    general_opts.add_options()
        ("help,h", "Help");

    boost::program_options::options_description advanced_opts("Advanced (working with datasets)");
    int dataset_id;
    std::string dataset_name;
	int dataset_label = -1;
    advanced_opts.add_options()
        ("replay,r", "Replay")
        ("capture,c", "Capture")
        ("postprocess,p", "Postprocess")
        ("keep_label,k", "Keep labels during postprocessing")
        ("dataset_id,d", boost::program_options::value<int>(&dataset_id), "Dataset ID (required)")
        ("dataset_name,n", boost::program_options::value<std::string>(&dataset_name)->default_value("Untitled"), "Dataset name")
        ("dataset_label,l", boost::program_options::value<int>(&dataset_label), "Dataset label");

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

	// EagleSense welcome message
    std::cout << "EagleSense topviewkinect " << topviewkinect::VERSION << std::endl;

    // -h help
    if (program_options.count("help"))
    {
        std::cout << all_opts << std::endl;
        return EXIT_SUCCESS;
    }

    if (argc == 1)
    {
        return start();
    }

    // Advanced (working with datasets)
    if (!program_options.count("dataset_id"))
    {
        std::cout << "Missing dataset ID." << std::endl;
        return EXIT_FAILURE;
    }

    // -r replay
    if (program_options.count("replay"))
    {
        return replay(dataset_id);
    }

    // -c capture
    if (program_options.count("capture"))
    {
        return capture(dataset_id);
    }
    
    // -p postprocess
    if (program_options.count("postprocess"))
    {
        bool keep_label = program_options.count("keep_label") == 1;
        return postprocess(dataset_id, dataset_name, dataset_label, keep_label);
    }

    return EXIT_SUCCESS;
}

static int start()
{
    topviewkinect::util::log_println("Tracking ...");

    // Create interactive space
    topviewkinect::vision::TopViewSpace m_space;
    bool space_initialized = m_space.initialize();
    if (!space_initialized)
    {
        topviewkinect::util::log_println("Failed. Exiting...");
        return EXIT_FAILURE;
    }

	// Create windows
	cv::namedWindow(DEPTH_WINDOW_NAME);
	cv::namedWindow(INFRARED_WINDOW_NAME);
	cv::namedWindow(SPACE_WINDOW_NAME);
	const cv::Size enlarged_size = cv::Size(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.width * 2, topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.height * 2);
	cv::Mat visualization_frame_enlarged = cv::Mat::zeros(enlarged_size, CV_8UC3);

    // Tracking
    while (true)
    {
		cv::Mat depth_frame = m_space.get_depth_frame();
		cv::Mat infrared_frame = m_space.get_infrared_frame();
		cv::Mat visualization_frame = m_space.get_visualization_frame();
		cv::resize(visualization_frame, visualization_frame_enlarged, visualization_frame_enlarged.size(), 0, 0, cv::INTER_LINEAR);

        cv::imshow(DEPTH_WINDOW_NAME, depth_frame);
        cv::imshow(INFRARED_WINDOW_NAME, infrared_frame);
        cv::imshow(SPACE_WINDOW_NAME, visualization_frame_enlarged);

        int ascii_keypress = cv::waitKey(1); // any key to exit
        if (ascii_keypress != -1)
        {
            topviewkinect::util::log_println("EagleSense topviewkinect exiting...");
            break;
        }

        bool frame_received = m_space.refresh_kinect_frames();
        if (frame_received)
        {
            m_space.process_kinect_frames();
        }
    }

    topviewkinect::util::log_println("Done!");
    return EXIT_SUCCESS;
}

static int replay(const int dataset_id)
{
    topviewkinect::util::log_println("Replaying ...");

    // Create interactive space
    topviewkinect::vision::TopViewSpace m_space;
    bool space_initialized = m_space.initialize();
    if (!space_initialized)
    {
        topviewkinect::util::log_println("Failed. Exiting ...");
        return EXIT_FAILURE;
    }

    // Load dataset
    bool dataset_loaded = m_space.load_dataset(dataset_id);
    if (!dataset_loaded)
    {
        topviewkinect::util::log_println("Failed. Exiting ...");
        return EXIT_FAILURE;
    }

	// Create windows
	cv::namedWindow(DEPTH_WINDOW_NAME);
	cv::namedWindow(INFRARED_WINDOW_NAME);
	cv::namedWindow(SPACE_WINDOW_NAME);
	const cv::Size enlarged_size = cv::Size(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.width * 2, topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.height * 2);
	cv::Mat visualization_frame_enlarged = cv::Mat::zeros(enlarged_size, CV_8UC3);

    // Replay
    while (true)
    {
        cv::Mat depth_frame = m_space.get_depth_frame();
        cv::Mat infrared_frame = m_space.get_infrared_frame();
		cv::Mat infrared_low_frame = m_space.get_low_infrared_frame();
        cv::Mat visualization_frame = m_space.get_visualization_frame();
		cv::resize(visualization_frame, visualization_frame_enlarged, visualization_frame_enlarged.size(), 0, 0, cv::INTER_LINEAR);

        // Visualize tracking
        const int frame_id = m_space.get_kinect_frame_id();
        cv::putText(depth_frame, std::to_string(frame_id), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_WHITE);
        cv::putText(infrared_frame, std::to_string(frame_id), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_WHITE);
        cv::putText(visualization_frame_enlarged, std::to_string(frame_id), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_BGR_WHITE);
        cv::imshow(DEPTH_WINDOW_NAME, depth_frame);
        cv::imshow(INFRARED_WINDOW_NAME, infrared_frame);
		cv::imshow(INFRARED_LOW_WINDOW_NAME, infrared_low_frame);
        cv::imshow(SPACE_WINDOW_NAME, visualization_frame_enlarged);

        // Windows
        int ascii_keypress = cv::waitKeyEx(0);
        if (ascii_keypress == 2555904) // Win10 right arrow
        {
            m_space.replay_next_frame();
        }
        else if (ascii_keypress == 2424832) // Win10 left arrow
        {
            m_space.replay_previous_frame();
        }
        else if (ascii_keypress == 115) // 's'
        {
            m_space.save_visualization();
        }
        else // anything else
        {
            break;
        }
    }

    topviewkinect::util::log_println("Done!");
    return EXIT_SUCCESS;
}

static int capture(const int dataset_id)
{
    topviewkinect::util::log_println("Capturing ...");

    // Create interactive space
    topviewkinect::vision::TopViewSpace m_space;
    bool space_initialized = m_space.initialize();
    if (!space_initialized)
    {
        topviewkinect::util::log_println("Failed. Exiting ...");
        return EXIT_FAILURE;
    }

    // Create dataset
    bool dataset_created = m_space.create_dataset(dataset_id);
    if (!dataset_created)
    {
        topviewkinect::util::log_println("Failed. Exiting ...");
        return EXIT_FAILURE;
    }

    // Capture
    while (true)
    {
        cv::imshow(DEPTH_WINDOW_NAME, m_space.get_depth_frame());
        cv::imshow(INFRARED_WINDOW_NAME, m_space.get_infrared_frame());
		cv::imshow(INFRARED_LOW_WINDOW_NAME, m_space.get_low_infrared_frame());
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

    topviewkinect::util::log_println("Done!");
    return EXIT_SUCCESS;
}

static int postprocess(const int dataset_id, const std::string& dataset_name, const int dataset_label, const bool keep_label)
{
    std::string info;
    keep_label ? info = "(keep labels)" : "";
    topviewkinect::util::log_println("Postprocessing " + info + " ... ");

    // Create interactive space
    topviewkinect::vision::TopViewSpace m_space;
    // Skip tracking initialization

    // Load dataset
    bool dataset_loaded = m_space.load_dataset(dataset_id);
    if (!dataset_loaded)
    {
        topviewkinect::util::log_println("Failed. Exiting ...");
        return EXIT_FAILURE;
    }

    // Postprocess
    m_space.postprocess(dataset_name, dataset_label, keep_label);

    topviewkinect::util::log_println("Done!");
    return EXIT_SUCCESS;
}