import sys
import numpy as np
from PIL import Image, ImageOps
from argparse import ArgumentParser
from collections import deque

import time

import pyzed.sl as sl

import cv2
from numba import cuda
from matplotlib import pyplot as plt


OPENCV_WITH_CUDA = True
if OPENCV_WITH_CUDA:
    from cv2.cuda import cvtColor
    from cv2 import cuda_ORB as ORB
    # from opencv_vpi.orb import ORB
    from cv2 import cuda_StereoSGM as StereoSGBM
    # from opencv_vpi.disparity import StereoSGBM

else:
    from cv2 import cvtColor
    from cv2 import ORB
    from cv2 import StereoSGBM

# from scipy.optimize import least_squares

NUM_DISPARITIES = 128

def main():
    parser = ArgumentParser()
    parser.add_argument('--visualize', default=True, action='store_true', help='Show visualization')
    args = parser.parse_args()

    # --------- Init Zed Camera ---------
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 100
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    # Zed Camera Paramters
    image_size = zed.get_camera_information().camera_configuration.resolution
    cx = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cx
    cy = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cy
    fx = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
    baseline = (
        zed.get_camera_information().camera_configuration.calibration_parameters.get_camera_baseline()
    )
    print(f"fx: {fx} baseline: {baseline}")
    zed_stereo_img = sl.Mat()
    cv_stereo_img = zed_stereo_img.get_data()

    Q = np.array([[1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, -fx],
                [0, 0, -1.0 / baseline, 0]])


    # --------- Configure Detectors ---------
    # https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
    orb = ORB.create(nlevels=3, nfeatures=88*3)
    disparity_estimator = StereoSGBM.create(minDisparity=10, numDisparities=85, blockSize=11)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    sl_stereo_img = sl.Mat()
    cv_left_img_queue = deque()
    cv_depth_queue = deque()

    running = True
    while running:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            start = time.time()
            zed.retrieve_image(sl_stereo_img, sl.VIEW.SIDE_BY_SIDE)
            cv_stereo_img = sl_stereo_img.get_data()[:, :, :3] # Last channel is padded for byte alignment
            cv_img_left = cv_stereo_img[:, :image_size.width, :]
            cv_img_right = cv_stereo_img[:, image_size.width:, :]
            if OPENCV_WITH_CUDA:
                cv_img_left_gpu = cv2.cuda_GpuMat(cv_img_left)
                cv_img_right_gpu = cv2.cuda_GpuMat(cv_img_right)
                cv_img_left_gray_gpu = cv2.cuda.cvtColor(cv_img_left_gpu, cv2.COLOR_BGR2GRAY)
                cv_img_right_gray_gpu = cv2.cuda.cvtColor(cv_img_right_gpu, cv2.COLOR_BGR2GRAY)
                cv_left_img_queue.append(cv_img_left_gray_gpu)
            else:
                cv_left_img_queue.append(cv_img_left)

            if args.visualize:
                cv2.imshow("Display", cv_stereo_img)

            # ---------- ORB -----------
            if len(cv_left_img_queue) == 2:
                cv_img_t_1 = cv_left_img_queue[0]
                cv_img_t = cv_left_img_queue[-1]
                cv_left_img_queue.popleft()
                if OPENCV_WITH_CUDA:
                    cv_keypoints_t_1, descriptors_t_1 = orb.detectAndComputeAsync(cv_img_t_1, None)
                    cv_keypoints_t, descriptors_t = orb.detectAndComputeAsync(cv_img_t, None)

                    # Copy back into CPU, since Matcher is CPU-Only
                    descriptors_t_1 = descriptors_t_1.download()
                    descriptors_t = descriptors_t.download()

                else:
                    cv_keypoints_t_1, descriptors_t_1 = orb.detectAndCompute(cv_img_t_1, None)
                    cv_keypoints_t, descriptors_t = orb.detectAndCompute(cv_img_t, None)

                # CPU-Only bfmatcher
                matches = bf_matcher.match(descriptors_t_1, descriptors_t)

                if args.visualize:
                    if OPENCV_WITH_CUDA:
                        cv_keypoints_t_1 = orb.convert(cv_keypoints_t_1)
                        cv_keypoints_t = orb.convert(cv_keypoints_t)
                        cv_img_t_1 = cv_img_t_1.download()
                        cv_img_t = cv_img_t.download()
                    output_image = cv2.drawMatches(cv_img_t_1, cv_keypoints_t_1,cv_img_t,
                                    cv_keypoints_t,matches,None)
                    cv2.imshow("Tracked ORB", output_image)

            # ---------- Disparity and Depth -----------
            if OPENCV_WITH_CUDA:
                cv_img_left_gray = cv_img_left_gray_gpu.download()
                cv_img_right_gray = cv_img_right_gray_gpu.download()
            else:
                cv_img_left_gray = cv2.cvtColor(cv_img_left, cv2.COLOR_BGR2GRAY)
                cv_img_right_gray = cv2.cvtColor(cv_img_right, cv2.COLOR_BGR2GRAY)


            # cv_disparity has a dtype('int16')
            cv_disparity = disparity_estimator.compute(cv_img_left_gray, cv_img_right_gray)
            cv_disparity = cv_disparity.astype(np.float32)/16
            disparityImg = cv2.normalize(src=cv_disparity, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)
            cv2.imshow("Disparity raw", disparityImg)
            # cv2.imshow("disparity", cv_disparity)

            # disparity_map = cv2.cuda_GpuMat(cv_disparity) #cv2.CV_16SC1
            # disparity_map_f32 = cv2.cuda_GpuMat(disparity_map.size(), cv2.CV_32F)
            # disparity_map.convertTo(cv2.CV_32F, disparity_map_f32)
            # depth_map = cuda.device_array_like(np.zeros((image_size.height, image_size.width)))
            # ComputeDisparityToDepth(disparity_map_f32, depth_map, image_size.width, image_size.height, fx, baseline)
            # cuda.synchronize()
            # host_depth_map = depth_map.copy_to_host()

            # cv2.imshow("Depth Map", host_depth_map)
            # cv_depth_queue.append(host_depth_map)

            # ---------- Non-Linear Least Squares Solving -----------


            curr = time.time()
            latency = 1.0 / (curr - start)
            print(f"Running at {latency} hz")
            start = curr
            key = cv2.waitKey(1)
            if (key == 'q'):
                running = False
                break


    zed.close()


if __name__ == "__main__":
    main()
