import sys
import numpy as np
from PIL import Image, ImageOps
from argparse import ArgumentParser
from collections import deque

import cv2

import time

import vpi
import pyzed.sl as sl

from numba import cuda
from orb import ORBFeatureDetector
from disparity import DisparityEstimator
from depth import ComputeDisparityToDepth


def main():
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
    left_img_index = (0, 0, image_size.width, image_size.height)
    right_img_index = (image_size.width, 0, image_size.width, image_size.height)

    zed_stereo_img = sl.Mat()
    cv_stereo_img = zed_stereo_img.get_data()

    #   Q << 1, 0, 0, -cx,             // NOLINT
    #       0, 1, 0, -cy,              // NOLINT
    #       0, 0, 0, -fx,              // NOLINT
    #       0, 0, -1.0 / baseline, 0;  // NOLINT
    #   // VPI Params

    # --------- Configure VPI ---------
    backend = vpi.Backend.CUDA

    disparity_estimator = DisparityEstimator()
    orb_t = ORBFeatureDetector()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    sl_stereo_img = sl.Mat()
    cv_left_img_queue = deque()
    cv_depth_queue = deque()

    running = True
    while running:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            start = time.time()
            zed.retrieve_image(sl_stereo_img, sl.VIEW.SIDE_BY_SIDE)
            cv_stereo_img = sl_stereo_img.get_data() # numpy array
            cv_img_left = cv_stereo_img[:, :image_size.width, :]
            cv_img_right = cv_stereo_img[:, image_size.width:, :]
            cv_left_img_queue.append(cv_img_left)

            cv2.imshow("Display", cv_stereo_img)

            # ---------- ORB -----------
            if len(cv_left_img_queue) == 2:
                cv_img_t_1 = cv_left_img_queue[0]
                cv_img_t = cv_left_img_queue[-1]
                cv_left_img_queue.popleft()
                cv_img_out, cv_keypoints_t_1, descriptors_t_1 = orb_t(cv_img_t_1)
                cv_img_out, cv_keypoints_t, descriptors_t = orb_t(cv_img_t)

                # bfmatcher
                # matches = bf.match(descriptors_t_1[0], descriptors_t[0])

                # output_image = cv2.drawMatches(cv_img_t_1,cv_keypoints_t_1,cv_img_t,
                #                    cv_keypoints_t,matches,None,flags=2)

                # cv2.imshow("Matched ORB", output_image)
                cv2.imshow("Matched ORB", cv_img_out)

            # ---------- Disparity and Depth -----------
            cv_disparity, cv_confidence = disparity_estimator(cv_img_left, cv_img_right)
            cv2.imshow("Disparity", cv_disparity)

            disparity_map_f32 = disparity_estimator.disparityS16.convert(vpi.Format.F32, backend=vpi.Backend.CUDA)
            with disparity_map_f32.lock_cuda() as disparity_map:
                depth_map = cuda.device_array_like(np.zeros((image_size.height, image_size.width)))
                ComputeDisparityToDepth(disparity_map, depth_map, image_size.width, image_size.height, fx, baseline)
                cuda.synchronize()
                host_depth_map = depth_map.copy_to_host()

            cv2.imshow("Depth Map", host_depth_map)
            cv_depth_queue.append(host_depth_map)

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
