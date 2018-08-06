
# EagleSense

[OVERVIEW](#overview) | [PUBLICATIONS](#publications) | [LICENSE](#license)

[INSTALL](#install) | [BUILD](#build) | [RUN](#run)

## OVERVIEW

EagleSense is a top-view tracking system that leverages a depth-infrared hybrid sensing pipeline for real-time human activity and device recognition. It provides a minimalistic RESTful API that can be used by ubicomp applications (e.g., proxemic-aware or cross-device systems) in interactive spaces.

[![EagleSense](https://github.com/cjw-charleswu/eaglesense/blob/master/screenshots/video.png)](https://youtu.be/6QLHA7hC_Kc "EagleSense")

[VIDEO](https://youtu.be/6QLHA7hC_Kc)

## PUBLICATIONS

1. [EagleSense: Tracking People and Devices in Interactive
Spaces using Real-Time Top-View Depth-Sensing](https://dl.acm.org/citation.cfm?id=3025562). In Proceedings of the 35th Annual ACM Conference on Human Factors in Computing Systems ([CHI'17](https://chi2017.acm.org/)).

## LICENSE

EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing

Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

## INSTALL

1. Install [Git](https://git-scm.com/) (and [Git Desktop](https://desktop.github.com/)).

2. Clone EagleSense.

	```bash
	cd /your_workspace/
	git clone https://github.com/cjw-charleswu/eaglesense
	```

3. Install [Visual Studio](https://www.visualstudio.com/) with Visual C++.

4. Install [Microsoft Kinect v2 SDK](https://developer.microsoft.com/en-us/windows/kinect).

5. Download [Boost C++ Libraries](http://www.boost.org/) windows binaries.

6. Download [OpenCV 3](http://opencv.org/) windows binaries.

7. Install [Python 3](https://www.python.org/) (*The [Anaconda](https://www.continuum.io/downloads) distribution will contain most of the Python requirements.*), also see `requirements.txt`.

8. Install [XGBoost](http://xgboost.readthedocs.io/en/latest/).

	Follow this [installation guide](http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/). Download XGBoost.dll and then install the Python package.

#### Project structure
```bash
eaglesense/
    +-- data/
    ¦   +-- topviewkinect/
    +-- eaglesense/
    ¦   +-- server/
    ¦   +-- topviewkinect/
	+-- models/
	+-- config.json
	+-- setup.py
	+-- server.py
	+-- topviewkinect.py
	+-- requirements.txt
	+-- README.md
```

---

## BUILD

Initialize project directories

	```bash
	cd /path/to/eaglesense/
	python setup.py
	```

### EagleSense Tracking System

1. Set the following system environment variables.
    * **BOOSTCPP**=`/path/to/boost`
    * **PYTHON3**=`/path/to/python3` (or `/path/to/anaconda`)
    * **OPENCV3**=`/path/to/opencv3`
    * **Add** `%OPENCV3%\build\x64\vc15\bin`, `%PYTHON3%`, `%PYTHON3%\Scripts` to `PATH`.

2. Open the EagleSense project `/path/to/eaglesense/eaglesense/topviewkinect/topviewkinect.sln`.

3. Set `solution configuration` to `Release` and `x64`.

4. In the solution explorer, right click the `topviewkinect` project then select `Properties`. Include the following fields as needed. Make sure the versions are correct.
    * VC++ Directories --> Include Directories
		> `$(KINECTSDK20_DIR)\inc;$(VC_IncludePath);$(WindowsSDK_IncludePath);`

    * VC++ Directories --> Library Directories
		> `$(KINECTSDK20_DIR)\lib\x64;$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64);$(NETFXKitsDir)Lib\um\x64`

    * C/C++ --> General --> Additional Include Directories
		> `include;$(OPENCV3)\build\include;$(BOOSTCPP);$(PYTHON3)\include;$(PYTHON3)\Lib\site-packages\numpy\core\include`

    * Linker --> General --> Additional Library Directories
		> `$(OPENCV3)\build\x64\vc14\lib;$(BOOSTCPP)\lib64-msvc-14.0;$(PYTHON3)\libs;$(PYTHON3)\Lib\site-packages\numpy\core\lib;%(AdditionalLibraryDirectories)`

    * Linker --> Input --> Additional Dependencies
		> `opencv_world340.lib;kinect20.lib;python36.lib;_tkinter.lib;npymath.lib;%(AdditionalDependencies)`

5. Open `Build` then `Build solution` (CTRL + SHIFT + B).

---

## RUN

### Tracking System

1. Configure the *EagleSense* top-view tracking system via `config.json` in the *EagleSense* root directory.

	```json
	{
	    "tracking": {
	        "framerate": 1,
	        "orientation_recognition": 1,
	        "interaction_recognition": 1,
	        "restful_connection": 0
	    },
	    "interaction_model": "standingtablet-xgboost.model",
	    "restful_server": {
	        "address": "localhost",
	        "port": 5000
	    },
	    "data": {
	        "depth": 1,
	        "infrared": 1,
	        "color": 0
	    }
	}
	```

2. Run

	```bash
	cd eaglesense/

	# Option 1: Python script
	python topviewkinect.py

	# Option 2: Executable
	./eaglesense/topviewkinect/x64/Release/topviewkinect.exe
	```

	Usage:

	```
	General:
	  -v [ --version ]                   	Version
	  -h [ --help ]                      	Help

	Advanced:
	  -r [ --replay ]                       Replay
	  -c [ --capture ]                      Capture
	  -p [ --postprocess ]                  Postprocess
	  -f [ --features ]                     Postprocess (Features only)
	  -d [ --dataset_id ] arg               Dataset ID (Required for advanced options)
	  -n [ --dataset_name ] arg (=Untitled) Dataset name
	```

3. JSON Data

	```json
	{
		"timestamp": 1450123752478,
		"skeletons": [
			{
				"id": 0,
				"head": {
					"x": 208,
					"y": 165,
					"z": 65,
					"orientation": 90,
				},
				"activity": "tablet",
				"activity_tracked": 1
			}
		]
	}
	```

### RESTful Server

1. Run

	```bash
	cd eaglesense/
	python server.py
	```

	Usage:

	```
	python server.py [-h] [--host HOST] [--port PORT]

	General:
	  -h [ --help ]	Help
	  --host HOST  	Server IP address
	  --port PORT  	Server port number
	```

---
