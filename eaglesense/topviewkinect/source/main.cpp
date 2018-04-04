
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

static constexpr const char* DEPTH_WIN_NAME = "TopView Kinect - Depth";
static constexpr const char* INFRARED_WIN_NAME = "TopView Kinect - Infrared";
static constexpr const char* INFRARED_LOW_WIN_NAME = "TopView Kinect - Infrared (Low)";
static constexpr const char* RGB_WIN_NAME = "TopView Kinect - RGB";
static constexpr const char* CALIBRATION_WIN_NAME = "TopView Kinect - Calibration";
static constexpr const char* ANDROID_SENSOR_WIN_NAME = "Android Sensor Stream";
static constexpr const char* SPACE_WIN_NAME = "TopView - Interactive Space";
static constexpr const char* SPACE_ENLARGED_WIN_NAME = "TopView - Interactive Space (Zoom)";

static int start();
static int replay(const int dataset_id);
static int capture(const int dataset_id);
static int postprocess(const int dataset_id, const std::string& dataset_name, const bool keep_label);

// just a hack
class AndroidSensorServer {
public:
	AndroidSensorServer(boost::asio::io_service& io_service, topviewkinect::vision::TopViewSpace& eaglesense) : 
		_socket(io_service, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 8888)),
		_eaglesense(eaglesense)
	{
		try_receive();
	}

private:
	double _first_beat = -1;
	topviewkinect::vision::TopViewSpace& _eaglesense;
	boost::asio::ip::udp::socket _socket;
	boost::asio::ip::udp::endpoint _sender_endpoint;
	const static int buffer_size = 1024;
	std::array<char, buffer_size> _data;

	void try_receive() {
		_socket.async_receive_from(
			boost::asio::buffer(_data, buffer_size - 1), _sender_endpoint,
			boost::bind(&AndroidSensorServer::handle_receive, this,
				boost::asio::placeholders::error,
				boost::asio::placeholders::bytes_transferred));
	}

	void handle_receive(const boost::system::error_code& error,
		std::size_t bytes_transferred) {
		if (!error || error == boost::asio::error::message_size) {

			topviewkinect::AndroidSensorData data_current = {};

			std::string sensor_data_str(&_data[0], &_data[0] + bytes_transferred);
			std::stringstream ss(sensor_data_str);
			
			//std::cout << ss.str() << std::endl;
			data_current.addr = _sender_endpoint.address().to_string() + ":" + std::to_string(_sender_endpoint.port());

			// parsing results from an Android app
			bool time_parsed = false;
			while (ss.good())
			{
				std::string substr;
				std::getline(ss, substr, ',');
				
				if (!time_parsed)
				{
					double arrival_time = std::stod(substr);
					if (this->_first_beat < 0)
					{
						this->_first_beat = arrival_time;
					}
					
					data_current.arrival_time = (arrival_time - this->_first_beat);
					time_parsed = true;
				}
				
				float value = std::stof(substr);

				// acceleromter
				if (value == 3)
				{
					std::string accel_x_substr;
					std::getline(ss, accel_x_substr, ',');
					float accel_x = std::stof(accel_x_substr);
					data_current.accel_x = accel_x;

					std::string accel_y_substr;
					std::getline(ss, accel_y_substr, ',');
					float accel_y = std::stof(accel_y_substr);
					data_current.accel_y = accel_y;

					std::string accel_z_substr;
					std::getline(ss, accel_z_substr, ',');
					float accel_z = std::stof(accel_z_substr);
					data_current.accel_z = accel_z;
				}

				// gyro
				if (value == 4)
				{
					std::string gyro_x_substr;
					std::getline(ss, gyro_x_substr, ',');
					float gyro_x = std::stof(gyro_x_substr);
					data_current.gyro_x = gyro_x;

					std::string gyro_y_substr;
					std::getline(ss, gyro_y_substr, ',');
					float gyro_y = std::stof(gyro_y_substr);
					data_current.gyro_y = gyro_y;

					std::string gyro_z_substr;
					std::getline(ss, gyro_z_substr, ',');
					float gyro_z = std::stof(gyro_z_substr);
					data_current.gyro_z = gyro_z;
				}

				// orientation
				if (value == 81)
				{
					std::string orientation_x_substr;
					std::getline(ss, orientation_x_substr, ',');
					float orientation_x = std::stof(orientation_x_substr);
					data_current.orientation_x = orientation_x;

					std::string orientation_y_substr;
					std::getline(ss, orientation_y_substr, ',');
					float orientation_y = std::stof(orientation_y_substr);
					data_current.orientation_y = orientation_y;

					std::string orientation_z_substr;
					std::getline(ss, orientation_z_substr, ',');
					float orientation_z = std::stof(orientation_z_substr);
					data_current.orientation_z = orientation_z;
				}

				// linear_accel
				if (value == 82)
				{
					std::string linear_accel_x_substr;
					std::getline(ss, linear_accel_x_substr, ',');
					float linear_accel_x = std::stof(linear_accel_x_substr);
					data_current.linear_accel_x = linear_accel_x;

					std::string linear_accel_y_substr;
					std::getline(ss, linear_accel_y_substr, ',');
					float linear_accel_y = std::stof(linear_accel_y_substr);
					data_current.linear_accel_y = linear_accel_y;

					std::string linear_accel_z_substr;
					std::getline(ss, linear_accel_z_substr, ',');
					float linear_accel_z = std::stof(linear_accel_z_substr);
					data_current.linear_accel_z = linear_accel_z;
				}

				// gravity
				if (value == 83)
				{
					std::string gravity_x_substr;
					std::getline(ss, gravity_x_substr, ',');
					float gravity_x = std::stof(gravity_x_substr);
					data_current.gravity_x = gravity_x;

					std::string gravity_y_substr;
					std::getline(ss, gravity_y_substr, ',');
					float gravity_y = std::stof(gravity_y_substr);
					data_current.gravity_y = gravity_y;

					std::string gravity_z_substr;
					std::getline(ss, gravity_z_substr, ',');
					float gravity_z = std::stof(gravity_z_substr);
					data_current.gravity_z = gravity_z;
				}

				// rotation vect
				if (value == 84)
				{
					std::string rotation_x_substr;
					std::getline(ss, rotation_x_substr, ',');
					float rotation_x = std::stof(rotation_x_substr);
					data_current.rotation_x = rotation_x;

					std::string rotation_y_substr;
					std::getline(ss, rotation_y_substr, ',');
					float rotation_y = std::stof(rotation_y_substr);
					data_current.rotation_y = rotation_y;

					std::string rotation_z_substr;
					std::getline(ss, rotation_z_substr, ',');
					float rotation_z = std::stof(rotation_z_substr);
					data_current.rotation_z = rotation_z;
				}
			}

			_eaglesense.refresh_android_sensor_data(data_current);
			//topviewkinect::util::log_println(data_current.to_str());

			try_receive();
		}
	}

	void handle_send(std::shared_ptr<std::string> message,
		const boost::system::error_code& ec,
		std::size_t bytes_transferred) {
		try_receive();
	}
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

	// OpenCV CUDA
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

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
	topviewkinect::util::log_println("Tracking ...");

	// Create interactive space
	topviewkinect::vision::TopViewSpace m_space;
	bool space_initialized = m_space.initialize();
	if (!space_initialized)
	{
		topviewkinect::util::log_println("Failed to initialize TopViewSpace. Exiting ...");
		return EXIT_FAILURE;
	}

	topviewkinect::util::log_println("Start server ...");

	boost::asio::io_service io_service;
	AndroidSensorServer server(io_service, m_space);
	std::thread thread1([&io_service]() { io_service.run(); });

	// Create windows
	cv::namedWindow(DEPTH_WIN_NAME);
	cv::namedWindow(INFRARED_WIN_NAME);
	cv::namedWindow(ANDROID_SENSOR_WIN_NAME);
	cv::namedWindow(SPACE_WIN_NAME);
	const cv::Size enlarged_size = cv::Size(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.width * 2, topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.height * 2);
	cv::Mat visualization_frame_enlarged = cv::Mat::zeros(enlarged_size, CV_8UC3);

	// Tracking
	while (true)
	{
		cv::Mat depth_frame = m_space.get_depth_frame();
		cv::Mat infrared_frame = m_space.get_infrared_frame();
		cv::Mat android_sensor_frame = m_space.get_android_sensor_frame();
		cv::Mat visualization_frame = m_space.get_visualization_frame();
		cv::resize(visualization_frame, visualization_frame_enlarged, visualization_frame_enlarged.size(), 0, 0, cv::INTER_LINEAR);

		cv::imshow(DEPTH_WIN_NAME, depth_frame);
		cv::imshow(INFRARED_WIN_NAME, infrared_frame);
		cv::imshow(ANDROID_SENSOR_WIN_NAME, android_sensor_frame);
		cv::imshow(SPACE_WIN_NAME, visualization_frame_enlarged);

		int keypress_win_ascii = cv::waitKey(1); // any key to exit
		if (keypress_win_ascii != -1)
		{
			topviewkinect::util::log_println("EagleSense topviewkinect exiting...");
			break;
		}

		bool sensor_frame_updated = m_space.process_android_sensor_data();
		if (sensor_frame_updated)
		{
			m_space.refresh_android_sensor_frame();
		}

		bool frame_received = m_space.refresh_kinect_frames();
		if (frame_received)
		{
			m_space.process_kinect_frames();
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
	cv::namedWindow(DEPTH_WIN_NAME);
	cv::namedWindow(INFRARED_WIN_NAME);
	cv::namedWindow(SPACE_WIN_NAME);
	cv::namedWindow(SPACE_ENLARGED_WIN_NAME);
	cv::namedWindow(ANDROID_SENSOR_WIN_NAME);
	const cv::Size enlarged_size = cv::Size(topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.width * 2, topviewkinect::kinect2::CV_DEPTH_FRAME_SIZE.height * 2);
	cv::Mat visualization_frame_enlarged = cv::Mat::zeros(enlarged_size, CV_8UC3);

	m_space.offline_calibration();

	// Replay
	while (true)
	{
		cv::Mat depth_frame = m_space.get_depth_frame();
		cv::Mat infrared_frame = m_space.get_infrared_frame();
		cv::Mat rgb_frame = m_space.get_rgb_frame();
		cv::Mat visualization_frame = m_space.get_visualization_frame();
		cv::Mat android_sensor_frame = m_space.get_android_sensor_frame();
		//cv::resize(visualization_frame, visualization_frame_enlarged, visualization_frame_enlarged.size(), 0, 0, cv::INTER_LINEAR);

		// Visualize tracking
		const int frame_id = m_space.get_kinect_frame_id();
		cv::putText(depth_frame, std::to_string(frame_id), cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1, topviewkinect::color::CV_WHITE, 3);
		cv::putText(infrared_frame, std::to_string(frame_id), cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1, topviewkinect::color::CV_WHITE, 3);
		cv::putText(visualization_frame, std::to_string(frame_id), cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1, topviewkinect::color::CV_BGR_BLACK, 3);

		cv::imshow(DEPTH_WIN_NAME, depth_frame);
		cv::imshow(INFRARED_WIN_NAME, infrared_frame);
		cv::imshow(RGB_WIN_NAME, rgb_frame);
		cv::imshow(SPACE_WIN_NAME, visualization_frame);
		cv::imshow(ANDROID_SENSOR_WIN_NAME, android_sensor_frame);
		//cv::imshow(SPACE_ENLARGED_WINDOW_NAME, visualization_frame_enlarged);

		int keypress_win_ascii = cv::waitKeyEx(0);
		if (keypress_win_ascii == 2555904) // '->'
		{
			m_space.replay_next_frame();
		}
		else if (keypress_win_ascii == 2424832) // '<-'
		{
			m_space.replay_previous_frame();
		}
		else if (keypress_win_ascii == 115) // 's'
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

// Calibration
std::vector<cv::Point> calibration_points;
cv::Point new_pt;
bool calibration_changed = false;

void calibration_on_mouse(int event, int x, int y, int, void* user_data)
{
	// Action when left button is clicked
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		new_pt.x = x;
		new_pt.y = y;
		//calibration_points.push_back(cv::Point(x, y));
		calibration_changed = true;
	}

	// Action when mouse is moving
	if ((event == cv::EVENT_MOUSEMOVE))
	{
		//std::cout << "new pt" << std::endl;
		//new_pt.x = x;
		//new_pt.y = y;
		//calibration_points.push_back(cv::Point(x, y));
		//calibration_changed = true;
		//	calibration_points.push_back(cv::Point(x, y));
		//	calibration_changed = true;
	}
}

static int capture(const int dataset_id)
{
	topviewkinect::util::log_println("Capturing ...");

	bool error = false;

	// Create interactive space
	topviewkinect::vision::TopViewSpace m_space;
	bool space_initialized = m_space.initialize();
	if (!space_initialized || error)
	{
		topviewkinect::util::log_println("Failed to initialize TopViewSpace. Exiting ...");
		error = true;
	}

	// Create dataset
	bool dataset_created = m_space.create_dataset(dataset_id);
	if (!dataset_created)
	{
		topviewkinect::util::log_println("Failed to initialize dataset. Exiting ...");
		error = true;
	}

	// Connect interactive space to server
	//bool online = m_space.run_server();
	boost::asio::io_service io_service;
	AndroidSensorServer server(io_service, m_space);
	std::thread thread1([&io_service]() { io_service.run(); });

	int keypress_prev = 0;

	if (!error)
	{
		while (true)
		{
			cv::imshow(DEPTH_WIN_NAME, m_space.get_depth_frame());
			cv::imshow(INFRARED_WIN_NAME, m_space.get_infrared_frame());
			cv::imshow(INFRARED_LOW_WIN_NAME, m_space.get_low_infrared_frame());
			cv::imshow(CALIBRATION_WIN_NAME, m_space.get_crossmotion_calibration_frame());
			cv::imshow(ANDROID_SENSOR_WIN_NAME, m_space.get_android_sensor_frame());
			cv::imshow(RGB_WIN_NAME, m_space.get_rgb_frame());

			cv::setMouseCallback(CALIBRATION_WIN_NAME, calibration_on_mouse);
			if (calibration_changed)
			{
				m_space.calibrate_sensor_fusion(new_pt);
				m_space.save_calibration();
				calibration_changed = false;
			}

			bool sensor_frame_updated = m_space.refresh_android_sensor_frame();
			if (sensor_frame_updated)
			{
				m_space.save_android_sensor_data();
			}

			bool frame_received = m_space.refresh_kinect_frames();
			if (frame_received)
			{
				m_space.save_kinect_frames();
			}

			int keypress_win_ascii = cv::waitKey(30);
			if (keypress_win_ascii == 97)
			{
				if (keypress_prev != 97)
				{
					m_space.set_android_sensor_label("device-inward-start");
					std::cout << "device-inward-start" << std::endl;
					keypress_prev = 97;
				}
				else
				{
					m_space.set_android_sensor_label("rest");
					std::cout << "rest" << std::endl;
					keypress_prev = 0;
				}
			}
			if (keypress_win_ascii == 100)
			{
				if (keypress_prev != 100)
				{
					m_space.set_android_sensor_label("device-outward-start");
					std::cout << "device-outward-start" << std::endl;
					keypress_prev = 100;
				}
				else
				{
					m_space.set_android_sensor_label("rest");
					std::cout << "rest" << std::endl;
					keypress_prev = 0;
				}
			}
			if (keypress_win_ascii == 27)
			{
				topviewkinect::util::log_println("System terminating...");
				break;
			}
		}
	}

	// Capture
	//while (true)
	//{
	//	cv::imshow(DEPTH_WINDOW_NAME, m_space.get_depth_frame());
	//	cv::imshow(INFRARED_WINDOW_NAME, m_space.get_low_infrared_frame());
	//	cv::imshow(SPACE_WINDOW_NAME, m_space.get_rgb_frame());

	//	int ascii_keypress = cv::waitKey(30); // 30 fps
	//	if (ascii_keypress != -1)
	//	{
	//		topviewkinect::util::log_println("System terminating...");
	//		break;
	//	}

	//	bool frame_received = m_space.refresh_kinect_frames();
	//	const topviewkinect::AndroidSensorData sensor_data = server.get_sensor_data();
	//	if (frame_received)
	//	{
	//		m_space.save_android_sensor_data(sensor_data);
	//		m_space.save_kinect_frames();
	//	}
	//}

	// Exit server
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