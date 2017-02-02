"""

EagleSense web server

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import argparse

import eaglesense as es


def main():
    parser = argparse.ArgumentParser(
        description="Run the EagleSense server"
    )

    parser.add_argument(
        "--host", dest="host", type=str, default="0.0.0.0",
        help="Server IP address"
    )

    parser.add_argument(
        "--port", dest="port", type=int, default=5000,
        help="Server port number"
    )

    args = parser.parse_args()

    es.server.run(args.host, args.port)


if __name__ == "__main__":
    main()
