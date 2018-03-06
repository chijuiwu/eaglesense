
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
#include <boost/asio.hpp>
#include <boost/bind.hpp>

#include <iostream>
#include <thread>
#include <string>
#include <sstream>
#include <vector>
#include <deque>
#include <winsock2.h>
#include <Ws2tcpip.h>
#include <stdio.h>
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define DATA_BUFSIZE 4096

#include "topviewkinect/topviewkinect.h"
#include "topviewkinect/vision/space.h"
#include "topviewkinect/vision/log.h"
#include "topviewkinect/kinect2.h"
#include "topviewkinect/color.h"
#include "topviewkinect/util.h"

static constexpr const char* DEPTH_WINDOW_NAME = "Top-View Kinect Depth";
static constexpr const char* INFRARED_WINDOW_NAME = "Top-View Kinect Infrared";
static constexpr const char* RGB_WINDOW_NAME = "Top-View Kinect RGB";
static constexpr const char* SPACE_WINDOW_NAME = "Top-View Interactive Space";
static constexpr const char* SPACE_ENLARGED_WINDOW_NAME = "Top-View Interactive Space (Zoom-in)";
static constexpr const char* ANDROID_SENSOR_WINDOW_NAME = "Android Sensor Stream";

static int start();
static int replay(const int dataset_id);
static int capture(const int dataset_id);
static int postprocess(const int dataset_id, const std::string& dataset_name, const bool keep_label);

class AndroidSensorServer {
	public:
		AndroidSensorServer(boost::asio::io_service& io_service)
			: _socket(io_service, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 8888))
		{
			try_receive();
		}

		float accel_x, accel_y, accel_z;
		float gyro_x, gyro_y, gyro_z;
		float orientation_x, orientation_y, orientation_z;
		float linear_accel_x, linear_accel_y, linear_accel_z;
		float rotation_vec_x, rotation_vec_y, rotation_vec_z;
		topviewkinect::AndroidSensorData sensor_data_current;
		std::deque<topviewkinect::AndroidSensorData> sensor_data_history;

		const topviewkinect::AndroidSensorData get_sensor_data() const
		{
			return sensor_data_current;
		}

		const std::deque<topviewkinect::AndroidSensorData> get_sensor_data_history() const
		{
			return this->sensor_data_history;
		}

	private:
		void try_receive() {
			_socket.async_receive_from(
				boost::asio::buffer(_data, buffer_size-1), _sender_endpoint,
				boost::bind(&AndroidSensorServer::handle_receive, this,
					boost::asio::placeholders::error,
					boost::asio::placeholders::bytes_transferred));
		}

		void handle_receive(const boost::system::error_code& error,
			std::size_t bytes_transferred) {
			if (!error || error == boost::asio::error::message_size) {
				std::string sensor_data_str(&_data[0], &_data[0] + bytes_transferred);
				
				std::vector<float> sensor_data_delimited;
				std::stringstream ss(sensor_data_str);

				while (ss.good())
				{
					std::string substr;
					std::getline(ss, substr, ',');
					sensor_data_delimited.push_back(std::stof(substr));
				}

				accel_x = sensor_data_delimited[2], accel_y = sensor_data_delimited[3], accel_z = sensor_data_delimited[4];
				gyro_x = sensor_data_delimited[6], gyro_y = sensor_data_delimited[7], gyro_z = sensor_data_delimited[8];
				orientation_x = sensor_data_delimited[14], orientation_y = sensor_data_delimited[15], orientation_z = sensor_data_delimited[16];
				linear_accel_x = sensor_data_delimited[18], linear_accel_y = sensor_data_delimited[19], linear_accel_z = sensor_data_delimited[20];
				rotation_vec_x = sensor_data_delimited[22], rotation_vec_y = sensor_data_delimited[23], rotation_vec_z = sensor_data_delimited[24];
				sensor_data_current = { accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, orientation_x, orientation_y, orientation_z, linear_accel_x, linear_accel_y, linear_accel_z, rotation_vec_x, rotation_vec_y, rotation_vec_z };

				this->sensor_data_history.push_back(sensor_data_current);
				if (this->sensor_data_history.size() > 300)
				{
					this->sensor_data_history.pop_front();
				}

				topviewkinect::util::log_println(sensor_data_current.to_str());

				try_receive();
			}
		}

		void handle_send(std::shared_ptr<std::string> message,
			const boost::system::error_code& ec,
			std::size_t bytes_transferred) {
			try_receive();
		}

		boost::asio::ip::udp::socket _socket;
		boost::asio::ip::udp::endpoint _sender_endpoint;
		const static int buffer_size = 1024;
		std::array<char, buffer_size> _data;
};

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
    advanced_opts.add_options()
        ("replay,r", "Replay")
        ("capture,c", "Capture")
        ("postprocess,p", "Postprocess")
        ("keep_label,k", "Keep labels during postprocessing")
        ("dataset_id,d", boost::program_options::value<int>(&dataset_id), "Dataset ID (required)")
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

	// EagleSense welcome message
    std::cout << "EagleSense topviewkinect " << topviewkinect::VERSION << std::endl;

    // -h help
    if (program_options.count("help"))
    {
        std::cout << all_opts << std::endl;
        return EXIT_SUCCESS;
    }

	// OpenCv
	std::cout << cv::getBuildInformation() << std::endl;

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
        return postprocess(dataset_id, dataset_name, keep_label);
    }

    return EXIT_SUCCESS;
}

static int start()
{
	topviewkinect::util::log_println("Start server ...");

	boost::asio::io_service io_service;
	AndroidSensorServer server{ io_service };
	std::thread thread1([&io_service]() { io_service.run(); });

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
	cv::namedWindow(ANDROID_SENSOR_WINDOW_NAME);
	const cv::Size enlarged_size = cv::Size(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.width * 2, topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.height * 2);
	cv::Mat visualization_frame_enlarged = cv::Mat::zeros(enlarged_size, CV_8UC3);

    // Tracking
    while (true)
    {
		cv::Mat depth_frame = m_space.get_depth_frame();
		cv::Mat infrared_frame = m_space.get_infrared_frame();
		cv::Mat visualization_frame = m_space.get_visualization_frame();
		cv::Mat android_sensor_frame = m_space.get_android_sensor_frame();
		cv::resize(visualization_frame, visualization_frame_enlarged, visualization_frame_enlarged.size(), 0, 0, cv::INTER_LINEAR);

        cv::imshow(DEPTH_WINDOW_NAME, depth_frame);
        cv::imshow(INFRARED_WINDOW_NAME, infrared_frame);
        cv::imshow(SPACE_WINDOW_NAME, visualization_frame_enlarged);
		cv::imshow(ANDROID_SENSOR_WINDOW_NAME, android_sensor_frame);

        int ascii_keypress = cv::waitKey(1); // any key to exit
        if (ascii_keypress != -1)
        {
            topviewkinect::util::log_println("EagleSense topviewkinect exiting...");
            break;
        }

		m_space.refresh_android_sensor_data(server.get_sensor_data_history());

        bool frame_received = m_space.refresh_kinect_frames();
        if (frame_received)
		{
            //m_space.process_kinect_frames();
        }
    }

	io_service.stop();
	thread1.join();
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
	//cv::namedWindow(SPACE_ENLARGED_WINDOW_NAME);
	//const cv::Size enlarged_size = cv::Size(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.width * 2, topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.height * 2);
	//cv::Mat visualization_frame_enlarged = cv::Mat::zeros(enlarged_size, CV_8UC3);

    // Replay
    while (true)
    {
        cv::Mat depth_frame = m_space.get_depth_frame();
        cv::Mat infrared_frame = m_space.get_infrared_frame();
        cv::Mat visualization_frame = m_space.get_visualization_frame();
		//cv::resize(visualization_frame, visualization_frame_enlarged, visualization_frame_enlarged.size(), 0, 0, cv::INTER_LINEAR);

        // Visualize tracking
        const int frame_id = m_space.get_kinect_frame_id();
        cv::putText(depth_frame, std::to_string(frame_id), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_WHITE);
        cv::putText(infrared_frame, std::to_string(frame_id), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_WHITE);
        cv::putText(visualization_frame, std::to_string(frame_id), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, topviewkinect::color::CV_BGR_WHITE);
        cv::imshow(DEPTH_WINDOW_NAME, depth_frame);
        cv::imshow(INFRARED_WINDOW_NAME, infrared_frame);
		cv::imshow(SPACE_WINDOW_NAME, visualization_frame);
        //cv::imshow(SPACE_ENLARGED_WINDOW_NAME, visualization_frame_enlarged);

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
	topviewkinect::util::log_println("Start server ...");

	boost::asio::io_service io_service;
	AndroidSensorServer server{ io_service };
	std::thread thread1([&io_service]() { io_service.run(); });

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
        cv::imshow(INFRARED_WINDOW_NAME, m_space.get_low_infrared_frame());
        cv::imshow(SPACE_WINDOW_NAME, m_space.get_rgb_frame());

        int ascii_keypress = cv::waitKey(30); // 30 fps
        if (ascii_keypress != -1)
        {
            topviewkinect::util::log_println("System terminating...");
            break;
        }

        bool frame_received = m_space.refresh_kinect_frames();
        if (frame_received)
        {
			m_space.save_android_sensor_data(server.get_sensor_data());
            m_space.save_kinect_frames();
        }
    }

	io_service.stop();
	thread1.join();
    topviewkinect::util::log_println("Done!");
    return EXIT_SUCCESS;
}

static int postprocess(const int dataset_id, const std::string& dataset_name, const bool keep_label)
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
    m_space.postprocess(dataset_name, keep_label);

    topviewkinect::util::log_println("Done!");
    return EXIT_SUCCESS;
}