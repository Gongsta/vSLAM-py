import cv2
import numpy as np
from enum import Enum

NUM_DISPARITIES = 128

CUDA = False
if CUDA:
    from cv2 import cuda_ORB as ORB
    # from cv2 import cuda_StereoSGM as StereoSGBM  # Too slow
    from opencv_vpi.disparity import StereoSGBM

else:
    from cv2 import ORB
    from cv2 import StereoSGBM

# cv2.setRNGSeed(0)
# np.random.seed(0)

"""
There are several methods to calculate camera motion:
- 2D-2D (monocular camera): Uses two sets of 2D points (at t-1 and t) to estimate camera motion. This is solved through epipolar geometry.
- 3D-3D (depth/stereo camera): Uses two sets of 3D points (at t-1 and t) to estimate camera motion. This is solved through ICP.
- 3D-2D (depth/stereo camera): Uses 3D points (at t-1) and 2D points (at t) to estimate camera motion. This is solved through PnP.
    - PnP itself can be solved in many ways, such as DLT, P3P, etc.
"""
VOMethod = Enum("VOMethod", ["VO_2D_2D", "VO_3D_2D", "VO_3D_3D"])


class VisualOdometry:

    def __init__(self, cx, cy, fx, baseline=0) -> None:
        # --------- General Parameters ---------
        self.visualize = True

        # --------- Image Queues ---------
        self.img_left_queue = []
        self.disparity_queue = []

        # --------- Detectors ---------
        self.disparity_estimator = StereoSGBM.create(
            minDisparity=0, numDisparities=NUM_DISPARITIES, blockSize=5
        )

        # --------- States for Non-linear Optimization ---------
        self.PAR0 = np.array([0, 0, 0, 1, 0, 0, 0])
        self.curr_pos = np.array([0, 0, 0, 1])
        self.x_t = []
        self.y_t = []
        self.z_t = []

        # --------- Camera Parameters and Matrices ---------
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.baseline = baseline  # Needed for stereo camera

        self.P = np.array([[fx, 0, cx, 0], [0, fx, cy, 0], [0, 0, 1, 0]])
        self.K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
        self.Q = np.array(
            [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, -fx], [0, 0, -1.0 / baseline, 0]]
        )

        self.orb = ORB.create(3000)

        ### Feature Matcher
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """

        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(
                np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)
                / np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)
            )
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

    def process_frame_mono(self, img_left):
        # ---------- Convert to Grayscale -----------
        # img_left_gray = self._convert_grayscale(img_left)
        self.img_left_queue.append(img_left)

        if len(self.img_left_queue) >= 2:
            img_t_1 = self.img_left_queue[-2]
            img_t = self.img_left_queue[-1]

            matched_kpts_t_1, matched_kpts_t = self._detect_and_match_2d_kpts(img_t_1, img_t)
            F1, F2 = matched_kpts_t_1, matched_kpts_t

            try:
                # Epipolar Geometry
                # E, mask = cv2.findEssentialMat(F1, F2, self.K, cv2.RANSAC, 0.5, 3.0, None)
                E, mask = cv2.findEssentialMat(F1, F2, self.K, threshold=1)

                # Decompose the Essential matrix into R and t
                R, t = self.decomp_essential_mat(E, F1, F2)

                # _, R, t, mask = cv2.recoverPose(E, F1, F2, focal=1, pp=(0.0, 0.0))

                # scale_factor = 0.1  # replace with actual scale factor
                # t_scaled = t * scale_factor
                T = self._form_transf(R, np.squeeze(t))
                return T

                # T = np.eye(4)
                # T[:3, :3] = R
                # T[:3, 3] = t_scaled.flatten()

                # self.curr_pos = np.matmul(T, self.curr_pos)
                # self.x_t.append(self.curr_pos[0])
                # self.y_t.append(self.curr_pos[1])
                # self.z_t.append(self.curr_pos[2])

                # return self.curr_pos
                return T

            except Exception as e:
                print("Optimization failed", e)
                return np.eye(4)  # identity matrix

    def process_frame(self, img_left, img_right=None, depth=None, method=VOMethod.VO_3D_2D):
        """
        Three main ways to proceed:
        1. 2D-2D (Solve Epipolar geometry by computing essential matrix)
        2. 3D-2D (solve with PnP)
        3. 3D-3D (solve with ICP)
        - If no depth image is provided, the depth is computed by disparity

        if CUDA flag is true, images as by default a cv2.cuda_GpuMat. Memory is abstracted away, so you focus on
        computation.

        """

        if img_right is None and depth is None:
            return self.process_frame_mono(img_left)

        # ---------- Convert to Grayscale -----------
        img_left_gray = self._convert_grayscale(img_left)
        img_right_gray = self._convert_grayscale(img_right)

        self.img_left_queue.append(img_left_gray)

        # # ---------- Compute Disparity -----------
        disparity = self.disparity_estimator.compute(img_left_gray, img_right_gray)
        cv2.imshow("disparity", disparity / 255.0)
        # depth = self._compute_depth_from_disparity(disparity)
        # self.disparity_queue.append(disparity)

        # ---------- Compute and Match Keypoints (kpts) across frames -----------
        if len(self.img_left_queue) >= 2:
            img_t_1 = self.img_left_queue[-2]
            img_t = self.img_left_queue[-1]

            matched_kpts_t_1, matched_kpts_t = self._detect_and_match_2d_kpts(img_t_1, img_t)

        #     disparity_t_1 = self.disparity_queue[-2]
        #     disparity_t = self.disparity_queue[-1]

        #     # ---------- Non-Linear Least Squares Solving -----------
        #     F1, F2, W1, W2 = self._project_2d_kpts_to_3d(
        #         disparity_t_1, disparity_t, matched_kpts_t_1, matched_kpts_t
        #     )

        #     # ---------- Essential Matrix ----------
        #     ransac_method = cv2.RANSAC
        #     kRansacThresholdNormalized = (
        #         0.0003  # metric threshold used for normalized image coordinates
        #     )
        #     kRansacProb = 0.999

        #     # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
        #     try:
        #         E, mask = cv2.findEssentialMat(F1, F2, self.K, cv2.RANSAC, 0.5, 3.0, None)
        #         _, R, t, mask = cv2.recoverPose(E, F1, F2, focal=1, pp=(0.0, 0.0))

        #         scale_factor = 0.1  # replace with actual scale factor
        #         t_scaled = t * scale_factor

        #         T = np.eye(4)
        #         T[:3, :3] = R
        #         T[:3, 3] = t_scaled.flatten()

        #         self.curr_pos = np.matmul(T, self.curr_pos)
        #         self.x_t.append(self.curr_pos[0])
        #         self.y_t.append(self.curr_pos[1])
        #         self.z_t.append(self.curr_pos[2])

        #         return self.curr_pos

        #         # if POSITION_PLOT:
        #         #     points.set_data(x_t, y_t)
        #         #     points.set_3d_properties(z_t)  # update the z data
        #         #     points2.set_data(x_t, y_t)
        #         #     # redraw just the points
        #         #     fig.canvas.draw()

        #         # Cannot move faster than 0.5m
        #         lower_bounds = [-0.5, -0.5, -0.5, -1.0, -1.0, -1.0, -1.0]
        #         upper_bounds = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]

        #         # res = least_squares(minimize, PAR0, args=(F1, F2, W1, W2, P),
        #         #                      bounds=(lower_bounds, upper_bounds))
        #         # print(F1.shape)
        #         # res = least_squares(minimize, PAR0, args=(F1, F2, W1, W2, P),
        #         #                     method="lm")
        #         # print("res.x", res.x)
        #         # curr_x += res.x[0]
        #         # curr_y += res.x[1]
        #         # curr_z += res.x[2]
        #         # x_t.append(curr_x)
        #         # y_t.append(curr_y)
        #         # z_t.append(curr_z)

        #         # points.set_data(x_t, y_t)
        #         # points.set_3d_properties(z_t)  # update the z data
        #         # # redraw just the points
        #         # fig.canvas.draw()

        #     except Exception as e:
        #         print("Optimization failed", e)

    def _compute_orb(self, img_t):
        if CUDA:
            kpts_t, desc_t = self.orb.detectAndComputeAsync(img_t, None)

            # Convert back into CPU
            if kpts_t.step == 0:
                print("Failed to process frame, No ORB keypoints found")
                return [], []
            kpts_t = self.orb.convert(kpts_t)
            desc_t = desc_t.download()
        else:
            kpts_t, desc_t = self.orb.detectAndCompute(img_t, None)

        return kpts_t, desc_t

    def _convert_grayscale(self, img):
        if CUDA:
            if type(img) == cv2.Mat or type(img) == np.ndarray:
                img = cv2.cuda_GpuMat(img)

            img_gray = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def _detect_and_match_2d_kpts(self, img_t_1, img_t):
        # ---------- Compute ORB Keypoints and Descriptors ----------
        kpts_t_1, desc_t_1 = self._compute_orb(img_t_1)
        kpts_t, desc_t = self._compute_orb(img_t)

        # ---------- Match Descriptors Across 2 frames ----------
        matched_kpts_t_1, matched_kpts_t = self._match_2d_kpts(kpts_t_1, kpts_t, desc_t_1, desc_t)

        return matched_kpts_t_1, matched_kpts_t

    def _compute_depth_from_disparity(self, disparity):
        cv_depth_map = self.fx * self.baseline / disparity

        if self.visualize:
            # Clip max distance of 2.0 for visualization
            _, depth_viz = cv2.threshold(cv_depth_map, 2.0, 2.0, cv2.THRESH_TRUNC)
            depth_viz = cv2.normalize(depth_viz, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Apply TURBO colormap to turn the depth map into color, blue=close, red=far
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)

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
                cv2.putText(
                    depth_viz,
                    depth_str,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Depth", depth_viz)

        return cv_depth_map

    def _match_2d_kpts(self, kpts_t_1, kpts_t, desc_t_1, desc_t):
        matched_kpts_t_1 = []
        matched_kpts_t = []

        USE_BF = False
        # # CPU-Only bfmatcher
        # # TODO: there's this GPU based matcher:
        # # https://forums.developer.nvidia.com/t/feature-extraction-and-matching-with-cuda-opencv-python/230784
        # # matcherGPU = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        if USE_BF:
            matches = self.bf_matcher.match(desc_t_1, desc_t)

            for match in matches:
                matched_kpts_t_1.append(list(kpts_t_1[match.queryIdx].pt))
                matched_kpts_t.append(list(kpts_t[match.trainIdx].pt))
            good_matches = matches
        else:
            # FLANN based matcher
            matches = self.flann.knnMatch(desc_t_1, desc_t, k=2)
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
            except ValueError:
                pass

        for match in good_matches:
            matched_kpts_t_1.append(list(kpts_t_1[match.queryIdx].pt))
            matched_kpts_t.append(list(kpts_t[match.trainIdx].pt))

        if self.visualize:
            img_t_1 = self.img_left_queue[-2]
            img_t = self.img_left_queue[-1]
            if CUDA:
                img_t_1 = img_t_1.download()
                img_t = img_t.download()
            output_image = cv2.drawMatches(
                img_t_1, kpts_t_1, img_t, kpts_t, good_matches[:30], None
            )
            cv2.imshow("Tracked ORB", output_image)
        return np.array(matched_kpts_t_1), np.array(matched_kpts_t)

    def _project_2d_kpts_to_3d(self, disparity_t_1, disparity_t, matched_kpts_t_1, matched_kpts_t):
        keypoints_2d_t_1 = []
        keypoints_2d_t = []
        keypoints_3d_t_1 = []
        keypoints_3d_t = []

        max_disparity = np.max(disparity_t)

        # TODO: Vectorize this
        for i in range(len(matched_kpts_t_1)):
            # From https://avisingh599.github.io/vision/visual-odometry-full/
            kpts_t_1 = matched_kpts_t_1[i]
            kpts_t = matched_kpts_t[i]
            x_1 = int(kpts_t_1[0])
            y_1 = int(kpts_t_1[1])
            d_1 = disparity_t_1[y_1, x_1]
            x = int(kpts_t[0])
            y = int(kpts_t[1])
            d = disparity_t[y, x]
            if d_1 == max_disparity or d_1 == 0 or d == max_disparity or d == 0:
                continue

            point = np.array([x_1, y_1, d_1, 1])
            point_3d = np.matmul(self.Q, point)
            point_3d /= point_3d[3]  # Normalize
            keypoints_2d_t_1.append(kpts_t_1)
            keypoints_3d_t_1.append(point_3d[:3])  # (X,Y,Z)

            point = np.array([x, y, d, 1])
            point_3d = np.matmul(self.Q, point)
            point_3d /= point_3d[3]  # Normalize
            keypoints_2d_t.append(kpts_t)
            keypoints_3d_t.append(point_3d[:3])  # (X,Y,Z)

        return (
            np.array(keypoints_2d_t_1),
            np.array(keypoints_2d_t),
            np.array(keypoints_3d_t_1),
            np.array(keypoints_3d_t),
        )
