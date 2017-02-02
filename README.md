
# EagleSense

[OVERVIEW](#overview) | [PUBLICATIONS](#publications) | [LICENSE](#license) | [CONTRIBUTING](#contributing)

[INSTALL](#install) | [BUILD](#build) | [RUN](#run) | [DATASET](#dataset)

## OVERVIEW

*EagleSense* is a top-view tracking system that leverages a depth-infrared hybrid sensing pipeline for real-time human activity and device recognition. It provides a minimalistic RESTful API that can be used by ubicomp applications (e.g., proxemic-aware or cross-device systems) in interactive spaces.

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

**Click image below for video.**

[![EagleSense](https://github.com/cjw-charleswu/eaglesense/blob/master/screenshots/video.png)](https://www.youtube.com/watch?v=6QLHA7hC_Kc "EagleSense")

## PUBLICATIONS

1. EagleSense: Tracking People and Devices in Interactive
Spaces using Real-Time Top-View Depth-Sensing. **_To Appear_** In Proceedings of the 35th Annual ACM Conference on Human Factors in Computing Systems ([CHI'17](https://chi2017.acm.org/)).

## LICENSE

EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing

Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

## CURRENTLY WORKING ON

Code refactor and machine learning pipeline.

## CONTRIBUTING

Contact chi-jui.wu.15 [at] ucl.ac.uk if you are interested in working on the following for *EagleSense*:

- Machine learning (deep neural nets)
- Interaction techniques (design and studies)

Feel free to also fork and hack.

---

## INSTALL

1. Install [Git](https://git-scm.com/)
	> Optionally [Git Desktop](https://desktop.github.com/)

2. Git Bash

	```bash
	cd /your/workspace/
	git clone https://github.com/cjw-charleswu/eaglesense
	```

### Project structure

```
eaglesense/
    +-- data/
    ¦   +-- topviewkinect/
	    ¦   +-- 1/
	    	¦   +-- depth/
	    	¦   +-- low_infrared/
    +-- eaglesense/
    ¦   +-- server/
    ¦   +-- topviewkinect/
	+-- models/
	+-- config.json
	+-- setup.py
	+-- classifier.py
	+-- server.py
	+-- topviewkinect.py
	+-- README.md
```

### C++ Requirements
* [Microsoft Visual Studio 2015 (v140) with VC++](https://www.visualstudio.com/)
* [Microsoft Kinect SDK 2.0](https://developer.microsoft.com/en-us/windows/kinect)
* [OpenCV >= 3.1.0 (prebuilt Windows binaries)](http://opencv.org/)
* [C++ Boost >= 1.60.0 (prebuilt Windows binaries)](http://www.boost.org/)
* [C++ REST SDK >= 2.8 (via Visual Studio NuGet Package Manager)](https://github.com/Microsoft/cpprestsdk/)

### Python Requirements
* [Python >= 3.4](https://www.python.org/) (Python 3.4 recommended)
* [XGBoost latest](https://xgboost.readthedocs.io/en/latest/) (see below)
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [pandas](http://pandas.pydata.org/)
* [matplotlib](http://matplotlib.org/)
* [seaborn](https://stanford.edu/~mwaskom/software/seaborn/)
* [tornado](https://github.com/tornadoweb/tornado)

### XGBoost

**NOTE: This was tested on Windows x64 with Python 3.4.4.**

1. Prerequisites
	* [Python 3.4](https://www.python.org/downloads/)
		> Add `/path/to/python` and `/path/to/python/Scripts` to `PATH`.

		> Update pip: `python -m pip install --upgrade pip`.

	* [Numpy and Scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/)
		> Install `numpy+mkl` and `scipy`.

	* [MinGW; MinGW64 recommended](https://sourceforge.net/projects/mingw-w64/)
		> Choose x86_64 architecture and win32 threads.

		> Add `/path/to/mingw64/bin` to `PATH`.

2. Git Bash

	```bash
	# Download XGBOOST latest
	cd /some/root/directory/
	git clone --recursive https://github.com/dmlc/xgboost
	cd xgboost/
	git submodule init
	git submodule update

	# Build XGBOOST from source
	cp make/mingw64.mk config.mk
	mingw32-make -j4

	# Install the python package
	cd python-package/
	python setup.py install
	```

3. If the above method doesn't work, you might want to try other resources such as [Google](https://www.google.com) or [this blog post](http://dnc1994.com/2016/03/installing-xgboost-on-windows/).

### Theano and Keras (**Currently NOT required**)

---

## BUILD

1. Initialize project directories

	```bash
	cd eaglesense/
	python setup.py
	```

### Top-view tracking system

1. Set the following system environment variables.
    * **`PYTHON3`** : `/path/to/python3`
    * **`BOOSTCPP`** : `/path/to/boost`
    * **`OPENCV3`** : `/path/to/opencv3`

2. Add `/path/to/opencv/build/x64/vc14/bin` to `PATH`.

3. Open `/path/to/eaglesense/eaglesense/topviewkinect/topviewkinect.sln`.

4. Set solution configuration to `Release x64`.

5. In solution explorer, right click on the `topviewkinect` project then select `Properties`. Include the following fields as needed.
    * VC++ Directories --> Include Directories
		> `$(KINECTSDK20_DIR)\inc;$(VC_IncludePath);$(WindowsSDK_IncludePath);`

    * VC++ Directories --> Library Directories
		> `$(KINECTSDK20_DIR)\lib\x64;$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64);$(NETFXKitsDir)Lib\um\x64`

    * C/C++ --> General --> Additional Include Directories
		> `include;$(OPENCV3)\build\include;$(BOOSTCPP);$(PYTHON3)\include;$(PYTHON3)\Lib\site-packages\numpy\core\include`

    * Linker --> General --> Additional Library Directories
		> `$(OPENCV3)\build\x64\vc14\lib;$(BOOSTCPP)\lib64-msvc-14.0;$(PYTHON3)\libs;$(PYTHON3)\Lib\site-packages\numpy\core\lib;%(AdditionalLibraryDirectories)`

    * Linker --> Input --> Additional Dependencies
		> `opencv_world310.lib;kinect20.lib;python34.lib;_tkinter.lib;npymath.lib;%(AdditionalDependencies)`

6. Click `Build` then `Build solution`.

---

## RUN

### Top-view tracking system

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

### Web server

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

## DATASET

A download link will be made available soon.
