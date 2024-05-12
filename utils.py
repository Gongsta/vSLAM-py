import numpy as np
import os
import cv2
import urllib.request
import math


def download_file(url, save_path):
    """Download a file from a URL and save it locally using urllib.request."""
    try:
        print(f"Downloading {url} to {save_path}")
        with urllib.request.urlopen(url) as response:
            content = response.read()
            with open(save_path, "wb") as file:
                file.write(content)
    except urllib.error.URLError as e:
        print(f"Failed to download {url}. Reason: {str(e)}")


def _compute_absolute_poses(self, relative_poses, include_initial=True):
    curr_pose = np.eye(4)
    absolute_poses = []

    if include_initial:
        absolute_poses.append(curr_pose)

    for T in relative_poses:
        curr_pose = T @ curr_pose
        absolute_poses.append(curr_pose)
    return absolute_poses


def _compute_relative_poses(self, absolute_poses):
    relative_poses = []
    for i in range(1, len(absolute_poses)):
        T_base = absolute_poses[i - 1]
        T_target = absolute_poses[i]
        T_base_inv = np.linalg.inv(T_base)
        T_relative = T_base_inv @ T_target
        relative_poses.append(T_relative)

    return relative_poses


def rotation_matrix_to_euler(R):
    if R[2][0] < 1:
        if R[2][0] > -1:
            theta_y = math.asin(R[2][0])
            theta_x = math.atan2(-R[2][1], R[2][2])
            theta_z = math.atan2(-R[1][0], R[0][0])
        else:  # R[2][0] = -1
            theta_y = -math.pi / 2
            theta_x = -math.atan2(R[1][2], R[1][1])
            theta_z = 0
    else:  # R[2][0] = 1
        theta_y = math.pi / 2
        theta_x = math.atan2(R[1][2], R[1][1])
        theta_z = 0

    return (theta_x, theta_y, theta_z)  # Return Euler angles in radians


def minimize(PAR, F1, F2, W1, W2, P1):
    """
    Source: https://avisingh599.github.io/vision/visual-odometry-full/

    PAR: 7x1 array of translation and rotation
    F1: 2D points in image 1
    F2: 2D points in image 2
    W1: 3D points in world frame 1
    W2: 3D points in world frame 2
    P1: Camera matrix

    """
    F = np.zeros((2 * F1.shape[0], 3))
    reproj1 = np.zeros((F1.shape[0], 3))
    reproj2 = np.zeros((F1.shape[0], 3))

    r = R.from_quat(PAR[3:]).as_matrix()
    t = PAR[:3]
    # 4x4 transformation matrix
    tran = np.eye(4)
    tran[:3, :3] = r
    tran[:3, 3] = t

    for k in range(F1.shape[0]):
        f1 = np.append(F1[k, :], 1)
        w2 = np.append(W2[k, :], 1)
        f2 = np.append(F2[k, :], 1)
        w1 = np.append(W1[k, :], 1)

        f1_repr = np.dot(P1, np.dot(tran, w2))
        f1_repr /= f1_repr[2]
        f2_repr = np.dot(P1, np.linalg.pinv(tran).dot(w1))
        f2_repr /= f2_repr[2]

        reproj1[k, :] = f1 - f1_repr
        reproj2[k, :] = f2 - f2_repr

    F = np.vstack((reproj1, reproj2))
    return F.flatten()


# Initial guess for the parameters (translation vector + rotation quaternion)
