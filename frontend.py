import cv2
import numpy as np
from enum import Enum
import g2o

CUDA = False
if CUDA:
    from cv2 import cuda_ORB as ORB

    # from cv2 import cuda_StereoSGM as StereoSGBM  # Too slow
    from opencv_vpi.disparity import StereoSGBM
else:
    from cv2 import ORB
    from cv2 import StereoSGBM


class VOMethod(Enum):
    """
    There are several methods to calculate camera motion:
    - 2D-2D (monocular camera): Uses two sets of 2D points (at t-1 and t) to estimate camera motion. This is solved through epipolar geometry.
    - 3D-2D (depth/stereo camera): Uses 3D points (at t-1) and 2D points (at t) to estimate camera motion. This is solved through PnP.
        - PnP itself can be solved in many ways, such as DLT, P3P, etc.
    - 3D-3D (depth/stereo camera): Uses two sets of 3D points (at t-1 and t) to estimate camera motion. This is solved through ICP.

    We also need to estimate camera motion over a longer time window due to drift and high amount of noise.
    The solution is to use bundle adjustment.
    """

    VO_2D_2D = 1
    VO_3D_2D = 2
    VO_3D_3D = 3


class VisualOdometry:
    def __init__(self, cx, cy, fx, baseline=0) -> None:
        # --------- Visualization ---------
        self.visualize = True

        # --------- Queues ---------
        self.img_left_queue = []
        self.depth_queue = []

        # --------- Detectors (for Frontend) ---------
        self.disparity_estimator = StereoSGBM.create(
            minDisparity=0, numDisparities=128, blockSize=5
        )
        self.orb = ORB.create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        # --------- States (used in Bundle Adjustment in Backend) ---------
        self.BA_WINDOW = 100  # Optimize over the last 100 frames
        # TODO: Only hold frames that have useful information??
        self.landmarks_2d_prev = [None]
        self.landmarks_2d = [None]  # List of landmarks at each time frame, in camera frame
        self.landmarks_3d = [None]  # List of landmarks at each time frame, in world frame
        # Relative pose transforms at each time frame, pose is a 4x4 SE3 matrix
        self.poses = [np.eye(4)]  # length T
        self.relative_poses = []  # length T-1, since relative

        # --------- Camera Parameters and Matrices ---------
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.baseline = baseline  # Needed for stereo camera

        self.P = np.array([[fx, 0, cx, 0], [0, fx, cy, 0], [0, 0, 1, 0]])
        self.K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)
        self.Q = np.array(
            [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, -fx], [0, 0, -1.0 / baseline, 0]]
        )

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

    def process_frame(self, img_left, img_right=None, depth=None, method=VOMethod.VO_3D_2D):
        """
        Three main ways to proceed:
        1. 2D-2D (Solve Epipolar geometry by computing essential matrix)
        2. 3D-2D (solve with PnP)
        3. 3D-3D (solve with ICP) (not currently implemented)
        - If no depth image is provided, the depth is computed by disparity

        if CUDA flag is true, images as by default a cv2.cuda_GpuMat. Memory is abstracted away, so you focus on
        computation.
        """
        if img_right is None and depth is None:
            method = VOMethod.VO_2D_2D

        # ---------- Convert Image to Grayscale -----------
        img_left_gray = self._convert_grayscale(img_left)
        self.img_left_queue.append(img_left_gray)

        # ---------- Compute Depth Image (if needed) -----------
        if method != VOMethod.VO_2D_2D:
            if depth is None:
                img_right_gray = self._convert_grayscale(img_right)
                disparity = self.disparity_estimator.compute(img_left_gray, img_right_gray)
                depth = self._compute_depth_from_disparity(disparity)

            depth[depth < 0.0] = np.nan
            self.depth_queue.append(depth)

            if self.visualize:
                self._visualize_depth(depth)

        # ---------- Compute and Match Keypoints (kpts) across frames -----------
        if len(self.img_left_queue) < 2:
            return

        img_left_t_1 = self.img_left_queue[-2]
        img_left_t = self.img_left_queue[-1]

        matched_kpts_t_1, matched_kpts_t = self._detect_and_match_2d_kpts(img_left_t_1, img_left_t)
        if len(matched_kpts_t) == 0:  # TODO: Check consequences of this
            return

        # ---------- Solve for camera motion (3 Different Methods) -----------
        if method == VOMethod.VO_2D_2D:
            # 2D-2D - Solve through Epipolar Geometry
            # IDK what I did here, took from https://github.com/niconielsen32/ComputerVision/blob/master/VisualOdometry/visual_odometry.py
            F1, F2 = matched_kpts_t_1, matched_kpts_t
            try:
                # E, mask = cv2.findEssentialMat(F1, F2, self.K, cv2.RANSAC, 0.5, 3.0, None)
                E, mask = cv2.findEssentialMat(F1, F2, self.K, threshold=1)

                # Decompose the Essential matrix into R and t
                R, t = self.decomp_essential_mat(E, F1, F2)
                t_scaled = 0.1 * t

                T = self._form_transf(R, np.squeeze(t_scaled))

            except Exception as e:
                print("Optimization failed", e)
                return np.eye(4)  # identity matrix

        elif method == VOMethod.VO_3D_2D:
            # 3D-2D - Solve by minimizing the reprojection error
            depth_t_1 = self.depth_queue[-2]
            points_3d_t_1 = self._project_2d_kpts_to_3d(depth_t_1, matched_kpts_t_1)

            # Some points may have NaN values due to NaN depth values, so we need to drop them
            points_3d_t_1, matched_kpts_t, matched_kpts_t_1 = self._drop_invalid_points(
                points_3d_t_1, matched_kpts_t, matched_kpts_t_1
            )

            T = self._minimize_reprojection_error(matched_kpts_t, points_3d_t_1)
            self.relative_poses.append(T)
            self.poses.append(self.relative_poses[-1] @ self.poses[-1])

        else:
            # 3D-3D method
            raise NotImplementedError("3D-3D method not implemented")

        if method == VOMethod.VO_2D_2D:
            return T

        # ---------- Store poses and landmarks that will be used by backend -----------
        depth_t = self.depth_queue[-1]
        points_3d_t = self._project_2d_kpts_to_3d(depth_t, matched_kpts_t)
        points_3d_t, matched_kpts_t, matched_kpts_t_1 = self._drop_invalid_points(
            points_3d_t, matched_kpts_t, matched_kpts_t_1
        )
        self.landmarks_2d_prev.append(matched_kpts_t_1)
        self.landmarks_2d.append(matched_kpts_t)

        # Convert 3D points from camera to world frame
        p_k = np.hstack([points_3d_t, np.ones((points_3d_t.shape[0], 1))])  # Homogenous
        w_T_k = np.linalg.inv(self.poses[-1])
        p_w = w_T_k @ p_k.T
        world_points_3d_t = p_w[:3].T  # Discard the homogeneous coordinate
        self.landmarks_3d.append(world_points_3d_t)

        return T

    def _minimize_reprojection_error(self, points_2d, points_3d):
        """
        Refer to SLAM textbook for formulation. Solved with g2o.
        """
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())  # TODO: Try PCG solver
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        cam = g2o.CameraParameters(self.fx, (self.cx, self.cy), 0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        # Add camera pose
        pose = g2o.SE3Quat()
        vertex_pose = g2o.VertexSE3Expmap()
        vertex_pose.set_id(0)
        vertex_pose.set_estimate(pose)
        optimizer.add_vertex(vertex_pose)

        for i, point_2d in enumerate(points_2d):
            point_3d = points_3d[i]

            vp = g2o.VertexPointXYZ()
            vp.set_id(i + 1)
            vp.set_marginalized(True)
            vp.set_estimate(point_3d)
            optimizer.add_vertex(vp)

            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, vp)
            edge.set_vertex(1, optimizer.vertex(0))
            edge.set_measurement(point_2d)
            edge.set_information(np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())
            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)

        optimizer.initialize_optimization()
        optimizer.optimize(10)

        T = vertex_pose.estimate().to_homogeneous_matrix()
        return T

    def _drop_invalid_points(self, points_3d, points_2d, points_2d_prev=None):
        """
        Drop points that have NaN values in either 2D or 3D coordinates. Assumes that the points are aligned,
        i.e. points_3d[i] corresponds to points_2d[i].

        Parameters
        ----------
        points_3d (ndarray): 3D points
        points_2d (ndarray): 2D points
        (optional) points_2d_prev (ndarray): 2D points from the previous frame

        Returns
        -------
        valid_points_2d (ndarray): 2D points without NaN values
        valid_points_3d (ndarray): 3D points without NaN values
        """
        assert len(points_2d) == len(points_3d)

        invalid_points = np.logical_or(
            np.isnan(points_3d).any(axis=1), np.isnan(points_2d).any(axis=1)
        )

        valid_points_2d = points_2d[~invalid_points]
        valid_points_3d = points_3d[~invalid_points]
        if points_2d_prev is not None:
            valid_points_2d_prev = points_2d_prev[~invalid_points]
            return valid_points_3d, valid_points_2d, valid_points_2d_prev

        return valid_points_3d, valid_points_2d

    def _compute_orb(self, img_t):
        """
        Compute ORB keypoints and descriptors for a given image.

        Parameters
        ----------
        img_t (ndarray): Image at time t

        Returns
        -------
        kpts_t (tuple): Keypoints at time t
        desc_t (ndarray): Descriptors at time t
        """
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
        """
        Detect and match 2D keypoints across two consecutive frames.

        Parameters
        ----------
        img_t_1 (ndarray): Image at time t-1
        img_t (ndarray): Image at time t

        Returns
        -------
        matched_kpts_t_1 (ndarray): Matched keypoints at time t-1
        matched_kpts_t (ndarray): Matched keypoints at time t
        """
        # ---------- Compute ORB Keypoints and Descriptors ----------
        kpts_t_1, desc_t_1 = self._compute_orb(img_t_1)
        kpts_t, desc_t = self._compute_orb(img_t)

        # ---------- Match Descriptors Across 2 frames ----------
        matched_kpts_t_1, matched_kpts_t = self._match_2d_kpts(kpts_t_1, kpts_t, desc_t_1, desc_t)

        return matched_kpts_t_1, matched_kpts_t

    def _compute_depth_from_disparity(self, disparity):
        cv_depth_map = self.fx * self.baseline / disparity
        return cv_depth_map

    def _visualize_depth(self, cv_depth_map):
        # Replace NaN values with 5.0m
        depth_viz = np.nan_to_num(cv_depth_map, nan=5.0)
        # Clip to (0.0m, 5.0m) for visualization
        depth_viz = np.clip(depth_viz, 0.0, 5.0)
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

    def _match_2d_kpts(self, kpts_t_1, kpts_t, desc_t_1, desc_t):
        """
        Match 2D keypoints across two frames using descriptors.

        Parameters
        ----------
        kpts_t_1 (list): Keypoints at time t-1
        kpts_t (list): Keypoints at time t
        desc_t_1 (ndarray): Descriptors at time t-1
        desc_t (ndarray): Descriptors at time t

        Returns
        -------
        matched_kpts_t_1 (ndarray): Matched keypoints at time t-1
        matched_kpts_t (ndarray): Matched keypoints at time t
        """
        matched_kpts_t_1 = []
        matched_kpts_t = []

        # FLANN based matcher
        try:
            matches = self.flann.knnMatch(desc_t_1, desc_t, k=2)
        except:
            matches = []

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

    def _project_2d_kpts_to_3d(self, depth, matched_kpts):
        """
        Vectorized method to project 2D keypoints to 3D points using depth information.
        3D points may contain NaN values if the depth is NaN.

        To convert from 2D to 3D, we use the following equation:

        sp = KP

        where
        - p = [u v 1] is the 2D point as measured by the camera in homogeneous coordinates
        - K is the intrinsic matrix
        - P = [x y z] is the 3D point in camera frame
        - s is some scalar (equal to z in the 3D point P = [x y z] in camera frame)

        Therefore, we can solve for P by multiplying the inverse of K with the 2D keypoint:
        P = K^{-1} s p

        Parameters
        ----------
        depth (ndarray): Depth image
        matched_kpts (ndarray): 2D keypoints, assumed to be an array of shape (N, 2)

        Returns
        -------
        points_3d (ndarray): 3D points in camera frame, shape (N, 3)
        """
        # Extracting the depth values of 2D keypoints
        depths = depth[matched_kpts[:, 1].astype(int), matched_kpts[:, 0].astype(int)]

        # Recover depth from homogeneous system by multiplying keypoints by their respective depths (solve s p)
        scaled_2d_points = matched_kpts * depths[:, np.newaxis]  # Shape (N, 2)
        scaled_2d_points = np.hstack((scaled_2d_points, depths.reshape(-1, 1)))  # Shape (N, 3)

        # K^{-1} z p
        points_3d = self.K_inv @ scaled_2d_points.T  # (3,3) @ (3,N) = (3, N)
        points_3d = points_3d.T  # (N, 3)

        return points_3d  # Transpose to get shape (N, 3)
