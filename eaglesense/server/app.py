"""

EagleSense web application

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import tornado.web

import os

from eaglesense import config
from eaglesense.server import handlers

STATIC_FILES = r"/static/(.*)"
DATA = r"/topviewkinect/data/([\w\W]*)"

eaglesense_web_handlers = [
    tornado.web.url(STATIC_FILES, tornado.web.StaticFileHandler),
    tornado.web.url(DATA, tornado.web.StaticFileHandler,
                    {"path": config.TOPVIEWKINECT_DATA_DNAME}),

    tornado.web.url(handlers.IndexPage.url, handlers.IndexPage),
    tornado.web.url(handlers.TrackingPage.url, handlers.TrackingPage),
    tornado.web.url(handlers.OptInOutPage.url, handlers.OptInOutPage),
    tornado.web.url(handlers.DemoPage.url, handlers.DemoPage),
    tornado.web.url(handlers.ContributorsPage.url, handlers.ContributorsPage),
    tornado.web.url(handlers.DataPage.url, handlers.DataPage),
    tornado.web.url(handlers.DatasetPage.url, handlers.DatasetPage),
    tornado.web.url(handlers.TopviewKinectFramePagination.url,
                    handlers.TopviewKinectFramePagination),

    tornado.web.url(handlers.PostTopviewKinectSkeletons.url,
                    handlers.PostTopviewKinectSkeletons),
    tornado.web.url(handlers.PostTopviewKinectLabels.url,
                    handlers.PostTopviewKinectLabels),

    tornado.web.url(handlers.TopviewKinectSkeletonWebsocket.url,
                    handlers.TopviewKinectSkeletonWebsocket)
]

eaglesense_web_settings = {
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
    "template_path": os.path.join(os.path.dirname(__file__), "templates"),
    "autoescape": None,
    "autoreload": False,
    "debug": True,
}


class EagleSense(tornado.web.Application):
    def __init__(self):
        self.handlers = eaglesense_web_handlers
        self.settings = eaglesense_web_settings
        tornado.web.Application.__init__(self, self.handlers, **self.settings)
        self.visualization_websockets = list()
        self.__shutdown = False

    def shutdown(self):
        self.__shutdown = True

    def is_shutdown(self):
        return self.__shutdown

    def add_visualization_websocket(self, socket):
        self.visualization_websockets.append(socket)

    def broadcast_depth_skeletons(self, skeletons):
        for socket in self.visualization_websockets:
            socket.send_skeletons(skeletons)
