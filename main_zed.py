import sys
import numpy as np
from PIL import Image, ImageOps
from argparse import ArgumentParser
from collections import deque
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

import time

import pyzed.sl as sl

import cv2
from numba import cuda
from matplotlib import pyplot as plt
CUDA = False
if CUDA:
    from cv2 import cuda_ORB as ORB
    # from cv2 import cuda_StereoSGM as StereoSGBM # Too slow
    from opencv_vpi.disparity import StereoSGBM

else:
    from cv2 import ORB
    from cv2 import StereoSGBM

from scipy.optimize import least_squares

NUM_DISPARITIES = 128

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


POSITION_PLOT = True

if POSITION_PLOT:
    plt.ion()
    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(121, projection='3d')  # Change 111 to 121
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    # # 2D plot
    ax2 = fig.add_subplot(122)  # Add this line for the 2D plot
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])



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
    F = np.zeros((2*F1.shape[0], 3))
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

def main():
    # Fast plotting
    x_t = []
    y_t = []
    z_t = []
    curr_pos = np.array([0,0,0,1])

    if POSITION_PLOT:
        points = ax1.plot(x_t, y_t, z_t)[0]
        points2 = ax2.plot(x_t, y_t)[0]
        fig.canvas.draw()

    PAR0 = np.array([0,0,0,1,0,0,0])
    parser = ArgumentParser()
    parser.add_argument('--visualize', default=True, action='store_true', help='Show visualization')
    args = parser.parse_args()

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
    zed_stereo_img = sl.Mat()
    cv_stereo_img = zed_stereo_img.get_data()

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


    # --------- Configure Detectors ---------
    # https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
    orb = ORB.create(nlevels=3, nfeatures=88*3)
    disparity_estimator = StereoSGBM.create(minDisparity=10, numDisparities=NUM_DISPARITIES, blockSize=5)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    sl_stereo_img = sl.Mat()
    cv_left_img_queue = deque()
    cv_disparity_queue = deque()

    running = True
    while running:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            start = time.time()
            zed.retrieve_image(sl_stereo_img, sl.VIEW.SIDE_BY_SIDE)
            cv_stereo_img = sl_stereo_img.get_data()[:, :, :3] # Last channel is padded for byte alignment
            cv_img_left = cv_stereo_img[:, :image_size.width, :]
            cv_img_right = cv_stereo_img[:, image_size.width:, :]
            if CUDA:
                cv_img_left_gpu = cv2.cuda_GpuMat(cv_img_left)
                cv_img_right_gpu = cv2.cuda_GpuMat(cv_img_right)
                cv_img_left_gray_gpu = cv2.cuda.cvtColor(cv_img_left_gpu, cv2.COLOR_BGR2GRAY)
                cv_left_img_queue.append(cv_img_left_gray_gpu)
            else:
                cv_left_img_queue.append(cv_img_left)


            # ---------- Disparity and Depth -----------
            cv_img_left_gray = cv2.cvtColor(cv_img_left, cv2.COLOR_BGR2GRAY)
            cv_img_right_gray = cv2.cvtColor(cv_img_right, cv2.COLOR_BGR2GRAY)

            # cv_disparity has a dtype('int8')
            cv_disparity = disparity_estimator.compute(cv_img_left_gray, cv_img_right_gray)
            cv_disparity_queue.append(cv_disparity)

            cv_depth_map = fx * baseline / cv_disparity
            # Clip max distance of 2.0 for visualization
            _, depth_viz = cv2.threshold(cv_depth_map, 2.0, 2.0, cv2.THRESH_TRUNC)
            depth_viz = cv2.normalize(depth_viz, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Apply TURBO colormap to turn the depth map into color, blue=close, red=far
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)

            if args.visualize:
                # Calculate middle coordinates
                mid_x = cv_depth_map.shape[1] // 2
                mid_y = cv_depth_map.shape[0] // 2

                # Select two points around the middle
                depth_points = [(mid_x - 100, mid_y), (mid_x + 100, mid_y)]

                # Get the depth values at the selected points
                depth_values = [cv_depth_map[y, x] for x, y in depth_points]

                # Annotate the depth values on the image
                for (x, y), depth in zip(depth_points, depth_values):
                    depth_str = str(round(depth, 2))  # Round to 2 decimal places
                    cv2.putText(depth_viz, depth_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow("Depth", depth_viz)

            # disparity_map = cv2.cuda_GpuMat(cv_disparity) #cv2.CV_16SC1
            # disparity_map_f32 = cv2.cuda_GpuMat(disparity_map.size(), cv2.CV_32F)
            # disparity_map.convertTo(cv2.CV_32F, disparity_map_f32)
            # depth_map = cuda.device_array_like(np.zeros((image_size.height, image_size.width)))
            # ComputeDisparityToDepth(disparity_map_f32, depth_map, image_size.width, image_size.height, fx, baseline)
            # cuda.synchronize()
            # host_depth_map = depth_map.copy_to_host()

            # cv2.imshow("Depth Map", host_depth_map)
            # cv_disparity_queue.append(host_depth_map)

            # ---------- ORB -----------
            if len(cv_left_img_queue) == 2:
                cv_img_t_1 = cv_left_img_queue[0]
                cv_img_t = cv_left_img_queue[-1]

                if CUDA:
                    cv_keypoints_t_1, descriptors_t_1 = orb.detectAndComputeAsync(cv_img_t_1, None)
                    cv_keypoints_t, descriptors_t = orb.detectAndComputeAsync(cv_img_t, None)


                    # Convert back into CPU
                    if (cv_keypoints_t_1.step == 0 or cv_keypoints_t.step == 0):
                        print("No ORB keypoints found")
                        # Pop images from queue before exiting to keep synchronized
                        if len(cv_left_img_queue) > 0:
                            cv_left_img_queue.popleft()
                            cv_disparity_queue.popleft()
                        continue
                    cv_keypoints_t_1 = orb.convert(cv_keypoints_t_1)
                    cv_keypoints_t = orb.convert(cv_keypoints_t)
                    descriptors_t_1 = descriptors_t_1.download()
                    descriptors_t = descriptors_t.download()
                else:
                    cv_keypoints_t_1, descriptors_t_1 = orb.detectAndCompute(cv_img_t_1, None)
                    cv_keypoints_t, descriptors_t = orb.detectAndCompute(cv_img_t, None)


                # CPU-Only bfmatcher
                # there's this GPU based matcher:
                # https://forums.developer.nvidia.com/t/feature-extraction-and-matching-with-cuda-opencv-python/230784
                # matcherGPU = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)

                matches = bf_matcher.match(descriptors_t_1, descriptors_t)
                matched_keypoints_t_1 = []
                matched_keypoints_t = []
                for match in matches:
                    matched_keypoints_t_1.append(list(cv_keypoints_t_1[match.queryIdx].pt))
                    matched_keypoints_t.append(list(cv_keypoints_t[match.trainIdx].pt))

                if args.visualize:
                    if CUDA:
                        cv_img_t_1 = cv_img_t_1.download()
                        cv_img_t = cv_img_t.download()
                    output_image = cv2.drawMatches(cv_img_t_1, cv_keypoints_t_1,cv_img_t,
                                    cv_keypoints_t,matches[:30], None)
                    cv2.imshow("Tracked ORB", output_image)

                # Compute 3D Points using the Depth Map
                cv_disparity_t_1 = cv_disparity_queue[0]
                cv_disparity_t = cv_disparity_queue[-1]

                keypoints_2d_t_1 = []
                keypoints_2d_t = []
                keypoints_3d_t_1 = []
                keypoints_3d_t = []

                max_disparity = np.max(cv_disparity_t)

                # TODO: Vectorize this
                for i in range(len(matched_keypoints_t_1)):
                    # From https://avisingh599.github.io/vision/visual-odometry-full/
                    keypoint_t_1 = matched_keypoints_t_1[i]
                    keypoint_t = matched_keypoints_t[i]
                    x_1 = int(keypoint_t_1[0])
                    y_1 = int(keypoint_t_1[1])
                    d_1 = cv_disparity_t_1[y_1, x_1]
                    x = int(keypoint_t[0])
                    y = int(keypoint_t[1])
                    d = cv_disparity_t[y, x]
                    if (d_1 == max_disparity or d_1 == 0 or d == max_disparity or d == 0):
                        continue

                    point = np.array([x_1, y_1, d_1, 1])
                    point_3d = np.matmul(Q, point)
                    point_3d /= point_3d[3] # Normalize
                    keypoints_2d_t_1.append(keypoint_t_1)
                    keypoints_3d_t_1.append(point_3d[:3]) # (X,Y,Z)

                    point = np.array([x, y, d, 1])
                    point_3d = np.matmul(Q, point)
                    point_3d /= point_3d[3] # Normalize
                    keypoints_2d_t.append(keypoint_t)
                    keypoints_3d_t.append(point_3d[:3]) # (X,Y,Z)

                # ---------- Non-Linear Least Squares Solving -----------
                F1 = np.array(keypoints_2d_t_1)
                F2 = np.array(keypoints_2d_t)
                W1 = np.array(keypoints_3d_t_1)
                W2 = np.array(keypoints_3d_t)

                # ---------- Essential Matrix ----------
                ransac_method = cv2.RANSAC
                kRansacThresholdNormalized = 0.0003  # metric threshold used for normalized image coordinates
                kRansacProb = 0.999

                # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
                try:
                    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
                    E, mask = cv2.findEssentialMat(F1, F2, K, cv2.RANSAC, 0.5, 3.0, None)
                    _, R, t, mask = cv2.recoverPose(E, F1, F2, focal=1, pp=(0., 0.))

                    scale_factor = 0.1  # replace with actual scale factor
                    t_scaled = t * scale_factor

                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t_scaled.flatten()
                    print(T)

                    curr_pos = np.matmul(T, curr_pos)
                    print(curr_pos)
                    x_t.append(curr_pos[0])
                    y_t.append(curr_pos[1])
                    z_t.append(curr_pos[2])

                    if POSITION_PLOT:
                        points.set_data(x_t, y_t)
                        points.set_3d_properties(z_t)  # update the z data
                        points2.set_data(x_t, y_t)
                        # redraw just the points
                        fig.canvas.draw()

                    # Cannot move faster than 0.5m
                    lower_bounds = [-0.5, -0.5, -0.5, -1.0, -1.0, -1.0, -1.0]
                    upper_bounds = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]

                    # res = least_squares(minimize, PAR0, args=(F1, F2, W1, W2, P),
                    #                      bounds=(lower_bounds, upper_bounds))
                    # print(F1.shape)
                    # res = least_squares(minimize, PAR0, args=(F1, F2, W1, W2, P),
                    #                     method="lm")
                    # print("res.x", res.x)
                    # curr_x += res.x[0]
                    # curr_y += res.x[1]
                    # curr_z += res.x[2]
                    # x_t.append(curr_x)
                    # y_t.append(curr_y)
                    # z_t.append(curr_z)

                    # points.set_data(x_t, y_t)
                    # points.set_3d_properties(z_t)  # update the z data
                    # # redraw just the points
                    # fig.canvas.draw()

                except Exception as e:
                    print("Optimization failed", e)


                cv_left_img_queue.popleft()
                cv_disparity_queue.popleft()



            curr = time.time()
            latency = 1.0 / (curr - start)
            print(f"Running at {latency} hz")
            start = curr
            key = cv2.waitKey(1)
            if (key == 'q'):
                running = False
                break


    zed.close()


if __name__ == "__main__":
    main()
