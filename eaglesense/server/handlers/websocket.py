"""

EagleSense web application sockets

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import tornado.websocket
import tornado.escape


class TopviewKinectSkeletonWebsocket(tornado.websocket.WebSocketHandler):
    url = r"/ws/skeleton/"

    def __init__(self, application, request, **kwargs):
        super(TopviewKinectSkeletonWebsocket, self).__init__(
            application, request, **kwargs)

        application.add_visualization_websocket(self)

    def data_received(self, chunk):
        pass

    def open(self):
        self.write_message("You are connected!!")

    def on_message(self, message):
        pass


def send_skeletons(self, skeletons):
    self.write_message(skeletons)
