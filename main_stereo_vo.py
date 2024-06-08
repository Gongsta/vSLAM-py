import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

import time
import cv2
import os

from visualodometry import VisualOdometry
from utils import download_file

USE_SIM = True

def main():
    parser = ArgumentParser()
    parser.add_argument("--visualize", default=True, action="store_true", help="Show visualization")
    args = parser.parse_args()

    # --------- Init Stereo Camera ---------
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
        # Open the camera
        capStereo = cv2.VideoCapture(0)

        # Stereo Camera Paramters
        image_size = 300
        cx = 100
        cy = 100
        fx = 100
        baseline = 5

    vo = VisualOdometry(cx, cy, fx, baseline)

    while True:
        start = time.time()
        if USE_SIM:
            cv_stereo_img = stereo_images[image_counter]

        else:
            ret, cv_img_left = capStereo.read()
            if ret:
                start = time.time()
                cv_img_left = cv_stereo_img[:, : cv_stereo_img.shape[1] // 2, :]
                cv_depth = depth_images[image_counter]

                image_counter += 1
                if image_counter >= len(stereo_images):
                    break

        vo.process_frame(cv_img_left, img_right=None, depth=cv_depth)
        print(vo.poses[-1])

        key = cv2.waitKey(1)
        if key == "q":
            break
        curr = time.time()
        latency = 1.0 / (curr - start)
        print(f"Running at {latency} hz")
        start = curr


if __name__ == "__main__":
    main()
