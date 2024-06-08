import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

import time
import cv2
import os

from frontend import VisualOdometry

USE_SIM = True

if not USE_SIM:
    import pyzed.sl as sl

from utils import download_file


def main():
    parser = ArgumentParser()
    parser.add_argument("--visualize", default=True, action="store_true", help="Show visualization")
    args = parser.parse_args()

    # --------- Init Zed Camera ---------
    if USE_SIM:
        if not os.path.isfile("sample_zed/data.npz"):
            os.makedirs("sample_zed", exist_ok=True)
            download_file(
                "https://github.com/Gongsta/Datasets/raw/main/sample_zed/camera_params.npz",
                "sample_zed/camera_params.npz",
            )
            download_file(
                "https://github.com/Gongsta/Datasets/raw/main/sample_zed/data.npz",
                "sample_zed/data.npz",
            )

        data = np.load("sample_zed/data.npz")
        calibration = np.load("sample_zed/camera_params.npz")

        stereo_images = data["stereo"]
        depth_images = data["depth"]

        K = calibration["K"]
        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        baseline = calibration["baseline"]

        image_counter = 0

    else:
        zed = sl.Camera()
        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA
        init_params.camera_fps = 100
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
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

        sl_stereo_img = sl.Mat()
        sl_depth = sl.Mat()

    vo = VisualOdometry(cx, cy, fx, baseline)

    while True:
        start = time.time()
        if USE_SIM:
            cv_stereo_img = stereo_images[image_counter]
            cv_img_left = cv_stereo_img[:, : cv_stereo_img.shape[1] // 2, :]
            cv_depth = depth_images[image_counter]

            image_counter += 1
            if image_counter >= len(stereo_images):
                break

        else:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(sl_stereo_img, sl.VIEW.SIDE_BY_SIDE)
                zed.retrieve_measure(sl_depth, sl.MEASURE.DEPTH)
                cv_stereo_img = sl_stereo_img.get_data()[
                    :, :, :3
                ]  # Last channel is padded for byte alignment
                cv_img_left = cv_stereo_img[:, : image_size.width, :]
                cv_depth = sl_depth.get_data()

            else:
                break

        vo.process_frame(cv_img_left, img_right=None, depth=cv_depth)
        print(vo.poses)
        plt.plot(vo.poses)

        key = cv2.waitKey(1)
        if key == "q":
            break
        curr = time.time()
        latency = 1.0 / (curr - start)
        print(f"Running at {latency} hz")
        start = curr


if __name__ == "__main__":
    main()
