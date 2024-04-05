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
    output_path = Path("dataset/")
    image_path = output_path / "2011_09_26"
    output_path.mkdir(exist_ok=True)

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip"
    if not image_path.exists():
        print("Downloading the data.")
        zip_path = output_path / "kitti.zip"
        urllib.request.urlretrieve(data_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as fid:
            fid.extractall(output_path)
        print(f"Data extracted to {output_path}.")

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

    vo = VisualOdometry(cx, cy, fx, baseline)
    gt_path = []
    pred_path = []
    curr_pose = None
    while capLeft.isOpened() and running:
        ret, cv_img_left = capLeft.read()
        if ret:
            start = time.time()

            T = vo.process_frame(cv_img_left)
            if curr_pose is None:
                curr_pose = gt_poses[0]
            else:
                curr_pose = np.matmul(curr_pose, np.linalg.inv(T))

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
