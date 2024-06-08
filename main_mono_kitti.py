import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import time
import cv2
from matplotlib import pyplot as plt

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from frontend import VisualOdometry

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def main():
    parser = ArgumentParser()
    parser.add_argument("--visualize", default=True, action="store_true", help="Show visualization")
    args = parser.parse_args()

    capLeft = cv2.VideoCapture("KITTI_sequence_1/image_l/%6d.png")

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

    vo = VisualOdometry(cx, cy, fx, baseline, initial_pose=gt_poses[0])
    gt_path = []
    pred_path = []
    while capLeft.isOpened() and running:
        ret, cv_img_left = capLeft.read()
        if ret:
            start = time.time()

            vo.process_frame(cv_img_left)
            pred_path.append((vo.poses[-1][0, 3], vo.poses[-1][2, 3]))

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
    cv2.destroyAllWindows()

    gt_path = [(gt_pose[0, 3], gt_pose[2, 3]) for gt_pose in gt_poses]
    x_t = [pred_path[i][0] for i in range(len(pred_path))]
    y_t = [pred_path[i][1] for i in range(len(pred_path))]
    x_truth = [gt_path[i][0] for i in range(len(gt_path))]
    y_truth = [gt_path[i][1] for i in range(len(gt_path))]
    plt.plot(x_t, y_t, label="estimate")
    plt.plot(x_truth, y_truth, label="gt")
    plt.legend()
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
