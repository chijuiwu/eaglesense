
/**
EagleSense utility functions

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/locale.hpp>

namespace topviewkinect
{
    namespace util
    {
        static void split(const std::string &s, char delim, std::vector<std::string> &elems) {
            std::stringstream ss;
            ss.str(s);
            std::string item;
            while (std::getline(ss, item, delim)) {
                elems.push_back(item);
            }
        }

        static std::vector<std::string> string_split(const std::string &s, char delim) {
            std::vector<std::string> elems;
            split(s, delim, elems);
            return elems;
        }

        template <class Interface> inline void safe_release(Interface **ppT)
        {
            if (*ppT)
            {
                (*ppT)->Release();
                *ppT = NULL;
            }
        };

        //template <typename T> inline std::vector<size_t> sort_indexes(const std::vector<T>& v)
        //{
        //    std::vector<size_t> idx(v.size());
        //    std::iota(idx.begin(), idx.end(), 0);
        //    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
        //    return idx;
        //};

        static const std::string get_current_datetime()
        {
            std::ostringstream datetime_ss;
            boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
            boost::gregorian::date date = now.date();
            boost::posix_time::time_duration time = now.time_of_day();
            datetime_ss << "[" << date.year() << "-" << date.month() << "-" << date.day() << " " << time.hours() << ":" << time.minutes() << ":" << time.seconds() << "]";
            return datetime_ss.str();
        }

        static const void log_println(const std::string& message)
        {
            std::cout << get_current_datetime() << " - " << message << std::endl;
        }

        // Time counter
        //std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        //std::cout << "time: " << duration.count() << std::endl;
    }
}
