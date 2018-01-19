"""

EagleSense web application page handlers

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import tornado.escape

import math

from eaglesense import io
from eaglesense.server.handlers import base

FRAME_PAGINATION_SIZE = 1000


class IndexPage(base.BaseHandler):
    url = r"/"

    def get(self):
        self.render("index.html")


class TrackingPage(base.BaseHandler):
    url = r"/tracking"

    def get(self):
        self.render("tracking.html")


class OptInOutPage(base.BaseHandler):
    url = r"/opt-in-out"

    def get(self):
        self.render("opt-in-out.html")


class AboutPage(base.BaseHandler):
    url = r"/about"

    def get(self):
        self.render("about.html")


class DemoPage(base.BaseHandler):
    url = r"/demo"

    def get(self):
        self.render("demo.html")


class ContributorsPage(base.BaseHandler):
    url = r"/contributors"

    def get(self):
        self.render("contributors.html")


class DataPage(base.BaseHandler):
    url = r"/data"

    def get(self):
        data_list = io.get_data()
        self.render("data.html", data_list=data_list)


class DatasetPage(base.BaseHandler):
    url = r"/dataset/(?P<id>[\d]+)"

    def get(self, id):
        dataset_json = io.load_topviewkinect_dataset(id)
        labels_df = io.load_topviewkinect_labels(id)
        labels_series = labels_df.loc[:, "activity"]
        orientations_series = labels_df.loc[:, "orientation"]
        orientations_status_series = labels_df.loc[:, "orientation_accurate"]

        num_frames = len(labels_series.index)
        num_pages = int(math.ceil(num_frames / FRAME_PAGINATION_SIZE))

        frame_labels = labels_series[:FRAME_PAGINATION_SIZE]
        frame_labels_dict = frame_labels.to_dict()

        frame_orientations = orientations_series[:FRAME_PAGINATION_SIZE]
        frame_orientations_dict = frame_orientations.to_dict()

        frame_orientations_status = orientations_status_series[:FRAME_PAGINATION_SIZE]
        frame_orientations_status_dict = frame_orientations_status.to_dict()

        self.render("dataset.html",
                    dataset=dataset_json, page_size=FRAME_PAGINATION_SIZE,
                    num_pages=num_pages, page_num=1,
                    frame_labels_dict=frame_labels_dict,
                    frame_orientations_dict=frame_orientations_dict,
                    frame_orientations_status_dict=frame_orientations_status_dict)


class TopviewKinectFramePagination(base.BaseHandler):
    url = r"/topviewkinect/(?P<id>[\d]+)/(?P<size>[\d]+)/(?P<page>[\d]+)"

    def get(self, id, size, page):
        labels_df = io.load_topviewkinect_labels(id)
        activities = labels_df.loc[:, "activity"]
        orientations = labels_df.loc[:, "orientation"]
        orientations_status = labels_df.loc[:, "orientation_accurate"]

        page_size = int(size)
        page_num = int(page)
        frame_ids_from = (page_num - 1) * page_size
        frame_ids_to = page_num * page_size
        frame_labels_dict = activities[frame_ids_from:frame_ids_to].to_dict()
        orientations_dict = orientations[frame_ids_from:frame_ids_to].to_dict()
        orientations_status_dict = orientations_status[frame_ids_from:frame_ids_to].to_dict()

        # Dictionaries must be strings to be json encoded
        frame_labels_dict = dict(
            zip(list(map(str, frame_labels_dict.keys())),
                list(map(str, frame_labels_dict.values())))
        )

        frame_orientations_dict = dict(
            zip(list(map(str, orientations_dict.keys())),
                list(map(str, orientations_dict.values())))
        )

        frame_orientations_status_dict = dict(
            zip(list(map(str, orientations_status_dict.keys())),
                list(map(str, orientations_status_dict.values())))
        )

        # Note: Handle duplicate elements in list when there are multiple skeletons

        self.finish(tornado.escape.json_encode({
            "frame_labels_dict": frame_labels_dict,
            "frame_orientations_dict": frame_orientations_dict,
            "frame_orientations_status_dict": frame_orientations_status_dict
        }))
