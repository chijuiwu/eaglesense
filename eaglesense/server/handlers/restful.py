"""

EagleSense web application RESTful connection handlers

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import logging
import tornado.web
import tornado.escape

from eaglesense import io
from eaglesense.server.handlers import base


class PostTopviewKinectSkeletons(base.BaseHandler):
    url = r"/topviewkinect/skeletons"

    def post(self):
        try:
            skeletons = tornado.escape.json_decode(self.request.body)

            logging.info("Received skeletons")

            self.application.broadcast_depth_skeletons(skeletons)

            self.finish(tornado.escape.json_encode({"response": "OK"}))

        except tornado.web.MissingArgumentError:
            self.set_status(400)
            self.finish("???")


class PostTopviewKinectLabels(base.BaseHandler):
    url = r"/topviewkinect/label/(?P<id>[\w]+)"

    def post(self, id):
        id = int(id)

        try:
            frame_labels = tornado.escape.json_decode(self.request.body)

            logging.info("Received labels")

            io.update_topviewkinect_labels(id, frame_labels)

            self.finish(tornado.escape.json_encode({"response": "OK"}))

        except tornado.web.MissingArgumentError:
            self.set_status(400)
            self.finish("Missing arguments")
