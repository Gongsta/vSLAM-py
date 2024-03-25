# vSLAM-py
A lightweight implementation of Visual SLAM in Python designed to run in real-time.


Built on top of OpenCV for features lighting, and g2o (loop-closure).

Sources of inspiration:
- [pyslam](https://github.com/luigifreda/pyslam/tree/master)
- 


### OpenCV with CUDA support
If you want this to run fast, you should use hardware acceleration. I am using OpenCV with CUDA support. Unfortunately, OpenCV with CUDA needs to be built from source. You'll need to make sure that you have CUDA
installed.

This is NOT to be confused with [CV-CUDA](https://github.com/CVCUDA/CV-CUDA). I found CV-CUDA
to be lacking lots of the libraries I needed, such as ORB detectors.

Refer to this [gist](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7) for how to download OpenCV
with CUDA. I am using version 4.5.4 of OpenCV.

```
cd ~/Downloads
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.5.4.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.4.zip
unzip opencv.zip
unzip opencv_contrib.zip
cd opencv-4.5.4
mkdir build
cd build
```

Since we are building with CU_DNN, make sure to use the correct CUDA_ARCH_BIN. You can find the correct value for your GPU [here](https://developer.nvidia.com/cuda-gpus). For example, on a Jetson AGX Orin, the value is 8.7.

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=8.7 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D PYTHON_EXECUTABLE=/usr/bin/python \
-D OPENCV_PYTHON3_INSTALL_PATH=/usr/lib/python3/dist-packages \
-D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib-4.5.4/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF ..
```


### Older
```
VPI for hardware-accelerated image processing
```bash
sudo apt install python3.8-vpi2
```


You might have VPI3 if you are on the latest. In which case run
```bash
sudo apt install python3.9-vpi3
```




#### Setting up the zed
```
cd /usr/local/zed/
python3 get_python_api.py
```

### Design Decisions
Initially, I was desigining the entire codebase in [C++](https://github.com/Gongsta/vSLAM). For
hardware accceleration, I used the help of the VPI library. I got the entire way there,
until I had to do optimization. This was a pain to do in C++ due to lack of support. I decided
to transfer everything to Python. VPI has a Python API, but I saw that OpenCV with CUDA had
all the bindings anyways.
