"""

EagleSense project directories

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/.."

TOPVIEWKINECT_EXE = PROJECT_DIR + """\
/eaglesense/topviewkinect/x64/Release/topviewkinect.exe"""

DATA_DIR = PROJECT_DIR + """\
/data"""

TOPVIEWKINECT_DATA_DIR = PROJECT_DIR + """\
/data/topviewkinect"""

TOPVIEWKINECT_DATASET_DIR_F = PROJECT_DIR + """\
/data/topviewkinect/{id}"""

TOPVIEWKINECT_DATASET_FEATURES_CSV_F = PROJECT_DIR + """\
/data/topviewkinect/{id}/features.csv"""

TOPVIEWKINECT_DATASET_LABELS_CSV_F = PROJECT_DIR + """\
/data/topviewkinect/{id}/labels.csv"""

TOPVIEWKINECT_DATASET_DESCRIPTION_JSON_F = PROJECT_DIR + """\
/data/topviewkinect/{id}/description.json"""
