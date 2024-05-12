import numpy as np
from argparse import ArgumentParser
import os
import time
import cv2
from frontend import VisualOdometry
from visualization import PangoVisualizer

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

def main():
    parser = ArgumentParser()
    parser.add_argument("--visualize", default=True, action="store_true", help="Show visualization")
    args = parser.parse_args()

    x_t = []
    y_t = []
    z_t = []

    # --------- Init Camera Camera ---------
    capLeft = cv2.VideoCapture(0)

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
    vis = PangoVisualizer()

    positions = []
    orientations = []
    curr_pose = None
    while capLeft.isOpened() and running:
        ret, cv_img_left = capLeft.read()
        if ret:
            start = time.time()
            T = vo.process_frame(cv_img_left)
            if curr_pose is None:
                curr_pose = np.eye(4)
            else:
                curr_pose = np.matmul(curr_pose, np.linalg.inv(T))

            positions.append(curr_pose[:3, 3])
            orientations.append(curr_pose[:3, :3])

            vis.update(positions, orientations)


            key = cv2.waitKey(1)
            if key == "q":
                running = False
                break
            curr = time.time()
            latency = 1.0 / (curr - start)
            print(f"Running at {latency} hz")
            start = curr
        else:
            running = False

    capLeft.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
