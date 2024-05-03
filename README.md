# vSLAM-py
A lightweight implementation of Visual SLAM in Python designed to run in real-time. Contains both monocular and stereo implementations. Continued from the abandoned [vSLAM](https://github.com/Gongsta/vSLAM) implementation in C++ (too tedious to write in, abandoned after writing the frontend).

Uses the following libraries (installed through `pip`):
- NumPy (for basic linear algebra)
- OpenCV (for feature detection / matching)
- g2o-python (for pose-graph optimization)
- Matplotlib (for basic plotting)


Sources of inspiration:
- [pyslam](https://github.com/luigifreda/pyslam/tree/master)
- [twitchslam](https://github.com/geohot/twitchslam/blob/master/slam.py)
- https://github.com/niconielsen32/ComputerVision


The above implementations don't have any real-time promises. My goal was to create something that can run in real-time.

## Installation
Getting started is super straightforward.

(Recommended, but not required) Create a virtual environment with Conda to not pollute your Python workspace:
```
conda create -n vslam-py python=3.8
```

Then, install the python libraries
```
pip install -r requirements.txt
```

## Usage

Run the monocular camera visual odometry:
```
python3 main_mono_camera.py
```

Run the stereo camera:
```
python3 main_stereo_camera.py
```

## Discussions

### Monocular vs. Stereo Visual SLAM

Mono and stereo visual odometry share many of the same techniques.
Only difference with Mono is that ground-truth depth can be obtained directly (unless you run deep monocular depth estimation).

However, monocular SLAM is more accessible, and everyone can just try it on their computers using a webcam. Stereo cameras aren't as standardized.
