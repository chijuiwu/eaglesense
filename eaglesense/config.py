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

PROJECT_DNAME = os.path.dirname(os.path.realpath(__file__)) + "/.."

TOPVIEWKINECT_EXE = PROJECT_DNAME + """\
/eaglesense/topviewkinect/x64/Release/topviewkinect.exe"""

DATA_DNAME = PROJECT_DNAME + """\
/data"""

TOPVIEWKINECT_DATA_DNAME = PROJECT_DNAME + """\
/data/topviewkinect"""

TOPVIEWKINECT_SUBJECT_DNAME = PROJECT_DNAME + """\
/data/topviewkinect/{id}"""

TOPVIEWKINECT_SUBJECT_DESCRIPTION_JSON_FNAME = PROJECT_DNAME + """\
/data/topviewkinect/{id}/description.json"""

TOPVIEWKINECT_FEATURES_CSV_FNAME = PROJECT_DNAME + """\
/data/topviewkinect/{id}/features.csv"""

TOPVIEWKINECT_LABELS_CSV_FNAME = PROJECT_DNAME + """\
/data/topviewkinect/{id}/labels.csv"""

TOPVIEWKINECT_PARAMS_PKL_FNAME = PROJECT_DNAME + """\
/data/topviewkinect/{id}/params_{test}.pkl"""

MODEL_DNAME = PROJECT_DNAME + """\
/models"""

EAGLESENSE_MODEL_FNAME = PROJECT_DNAME + """\
/models/{model}.model"""
