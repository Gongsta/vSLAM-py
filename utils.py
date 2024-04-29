
import numpy as np
import os
import cv2


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
