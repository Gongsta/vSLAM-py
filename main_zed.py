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

import pyzed.sl as sl

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

    # Camera matrices
    # Reprojection Matrix
    Q = np.array([[1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, -fx],
                [0, 0, -1.0 / baseline, 0]])

    # Calibration Matrix
    P = np.array([[fx, 0, cx, 0],
                [0, fx, cy, 0],
                [0, 0, 1, 0]])


    vo = VisualOdometry(cx, cy, fx, baseline)
    gt_path = []
    pred_path = []
    curr_pose = None

    sl_left = sl.Mat()
    sl_depth = sl.Mat()

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            start = time.time()
            zed.retrieve_image(sl_left, sl.VIEW.SIDE_BY_SIDE)
            zed.retrive_measure(sl_depth, sl.MEASURE.DEPTH)
            # cv_stereo_img = sl_stereo_img.get_data()[:, :, :3] # Last channel is padded for byte alignment
            cv_img_left = sl_left.get_data()
            cv_depth = sl_depth.get_data()
            # cv_img_left = cv_stereo_img[:, :image_size.width, :]
            # cv_img_right = cv_stereo_img[:, image_size.width:, :]

            # T = vo.process_frame(cv_img_left, cv_img_right)
            # if curr_pose is None:
            #     curr_pose = np.eye(4)
            # else:
            #     curr_pose = np.matmul(curr_pose, np.linalg.inv(T))

            # pred_path.append((curr_pose[0, 3], curr_pose[2, 3], curr_pose[1, 3]))

            # x_t = [pred_path[i][0] for i in range(len(pred_path))]
            # y_t = [pred_path[i][1] for i in range(len(pred_path))]
            # z_t = [pred_path[i][2] for i in range(len(pred_path))]

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
            break


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
