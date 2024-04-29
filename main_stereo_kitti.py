import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

from visualodometry import VisualOdometry

import os
import time
import urllib.request
import cv2
from numba import cuda
from matplotlib import pyplot as plt


def main():
    parser = ArgumentParser()
    parser.add_argument("--visualize", default=True, action="store_true", help="Show visualization")
    args = parser.parse_args()

    capLeft = cv2.VideoCapture("KITTI_sequence_1/image_l/%6d.png")
    capRight = cv2.VideoCapture("KITTI_sequence_1/image_l/%6d.png")

    gt_poses = []
    with open(os.path.join("KITTI_sequence_1", "poses.txt"), "r") as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=" ")
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            gt_poses.append(T)

    with open(os.path.join("KITTI_sequence_1", "calib.txt"), "r") as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=" ")
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]

    running = True
    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    baseline = 0.5

    vo = VisualOdometry(cx, cy, fx, baseline)
    gt_path = []
    pred_path = []
    curr_pose = None
    while capLeft.isOpened() and running:
        ret, cv_img_left = capLeft.read()
        ret, cv_img_right = capRight.read()
        if ret:
            start = time.time()
            # curr_pose = vo.process_frame(cv_img_left, cv_img_right)

            T = vo.process_frame(cv_img_left)
            if curr_pose is None:
                curr_pose = gt_poses[0]
            else:
                curr_pose = np.matmul(curr_pose, np.linalg.inv(T))
                # curr_pose = np.matmul(T, curr_pose)

            pred_path.append((curr_pose[0, 3], curr_pose[2, 3]))

            curr = time.time()
            latency = 1.0 / (curr - start)
            print(f"Running at {latency} hz")
            start = curr
            key = cv2.waitKey(10)
            if key == "q":
                running = False
                break
        else:
            running = False

    capLeft.release()
    capRight.release()
    cv2.destroyAllWindows()

    gt_path = [(gt_pose[0, 3], gt_pose[2, 3]) for gt_pose in gt_poses]
    x_t = [pred_path[i][0] for i in range(len(pred_path)) ]
    y_t = [pred_path[i][1] for i in range(len(pred_path)) ]
    x_truth = [gt_path[i][0] for i in range(len(gt_path)) ]
    y_truth = [gt_path[i][1] for i in range(len(gt_path)) ]
    plt.plot(x_t, y_t, label="estimate")
    plt.plot(x_truth, y_truth, label="gt")
    plt.legend()
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
