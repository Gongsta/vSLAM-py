import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

import os
import time
import cv2
from matplotlib import pyplot as plt

from scipy.spatial.transform import Rotation

from visualodometry import VisualOdometry

POSITION_PLOT = False

if POSITION_PLOT:
    plt.ion()
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection="3d")  # Change 111 to 121
    ax1.set_xlim([-10, 10])
    ax1.set_ylim([-10, 10])
    ax1.set_zlim([-10, 10])

    # # 2D plot
    ax2 = fig.add_subplot(122)  # Add this line for the 2D plot
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-10, 10])


def main():
    parser = ArgumentParser()
    parser.add_argument("--visualize", default=True, action="store_true", help="Show visualization")
    args = parser.parse_args()

    x_t = []
    y_t = []
    z_t = []
    if POSITION_PLOT:
        points = ax1.plot(x_t, y_t, z_t)[0]
        points2 = ax2.plot(x_t, y_t)[0]
        fig.canvas.draw()
        Q = ax1.quiver(0, 0, 0, 0, 0, 0, color="red")

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
    gt_path = []
    pred_path = []
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

            pred_path.append((curr_pose[0, 3], curr_pose[2, 3], curr_pose[1, 3]))

            x_t = [pred_path[i][0] for i in range(len(pred_path))]
            y_t = [pred_path[i][1] for i in range(len(pred_path))]
            z_t = [pred_path[i][2] for i in range(len(pred_path))]

            if POSITION_PLOT:
                # Update the orientation here
                r = Rotation.from_matrix(curr_pose[:3, :3])
                angles = r.as_euler("zyx", degrees=True)
                # Plot the new orientation
                Q.remove()
                Q = ax1.quiver(
                    x_t[-1], y_t[-1], z_t[-1], angles[0], angles[1], angles[2], color="red"
                )
                # Q = ax1.quiver(x_t[-1], y_t[-1], z_t[-1], curr_pose[:3, 0], curr_pose[:3, 1], curr_pose[:3, 2], color="red")

                points.set_data(x_t, y_t)
                points.set_3d_properties(z_t)  # update the z data
                points2.set_data(x_t, y_t)
                # redraw just the points
                fig.canvas.draw()

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

    # gt_path = [(gt_pose[0, 3], gt_pose[2, 3]) for gt_pose in gt_poses]
    # x_truth = [gt_path[i][0] for i in range(len(gt_path)) ]
    # y_truth = [gt_path[i][1] for i in range(len(gt_path)) ]
    # plt.plot(x_t, y_t, label="estimate")
    # plt.plot(x_truth, y_truth, label="gt")
    # plt.legend()
    # plt.axis('equal')
    # plt.show()


if __name__ == "__main__":
    main()
