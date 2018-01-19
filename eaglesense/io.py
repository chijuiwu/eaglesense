"""

EagleSense I/O functions

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd

import json
import os

from eaglesense import config


def create_data_directories():
    """
    Create project directories
    """

    if not os.path.exists(config.DATA_DNAME):
        os.makedirs(config.DATA_DNAME)

    if not os.path.exists(config.TOPVIEWKINECT_DATA_DNAME):
        os.makedirs(config.TOPVIEWKINECT_DATA_DNAME)


def get_data():
    data_list = list()
    for dataset_id in next(os.walk(config.TOPVIEWKINECT_DATA_DNAME))[1]:
        dataset = load_topviewkinect_dataset(dataset_id)
        if dataset is not None:
            data_list.append(dataset)
    return sorted(data_list, key=lambda d: d["id"])


def load_topviewkinect_dataset(dataset_id):
    dataset_json = config.TOPVIEWKINECT_SUBJECT_DESCRIPTION_JSON_FNAME.format(
        id=dataset_id)
    if os.path.isfile(dataset_json):
        with open(dataset_json) as f:
            return json.load(f)
    else:
        return None


def load_topviewkinect_labels(dataset_id):
    labels_csv = config.TOPVIEWKINECT_LABELS_CSV_FNAME.format(id=dataset_id)
    labels_df = pd.read_csv(labels_csv)
    labels_df.drop_duplicates(subset="frame_id", keep=False, inplace=True)  # remove duplicates
    labels_df.set_index("frame_id", inplace=True)
    return labels_df


def update_topviewkinect_labels(dataset_id, frame_labels):
    old_labels_df = load_topviewkinect_labels(dataset_id)

    frame_ids_list = list(map(int, frame_labels.keys()))
    new_frame_labels_list = list(map(int, frame_labels.values()))
    old_labels_df.loc[frame_ids_list, "activity"] = new_frame_labels_list

    labels_csv = config.TOPVIEWKINECT_LABELS_CSV_FNAME.format(id=dataset_id)
    old_labels_df.reset_index("frame_id", inplace=True)
    old_labels_df.to_csv(labels_csv, index=False)
