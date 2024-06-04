import cv2
import numpy as np
from enum import Enum
import g2o
from scipy.spatial.transform import Rotation

CUDA = False
if CUDA:
    from cv2 import cuda_ORB as ORB

    # from cv2 import cuda_StereoSGM as StereoSGBM  # Too slow
    from opencv_vpi.disparity import StereoSGBM
else:
    from cv2 import ORB
    from cv2 import StereoSGBM

np.random.seed(0)


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
    def __init__(
        self, cx, cy, fx, baseline=1, initial_pose=np.eye(4), visualize=False, save_path=True
    ) -> None:
        # --------- Visualization ---------
        self.visualize = visualize
        self.save_path = save_path

        if self.save_path:
            with open("poses.txt", "w") as f:
                f.write("# timestamp x y z qx qy qz qw\n")

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

        # Trying this out for better performance
        self.GLOBAL_landmarks_2d = []
        self.GLOBAL_landmarks_3d = []

        # Relative pose transforms at each time frame, pose is a 4x4 SE3 matrix
        self.poses = [initial_pose]  # length T
        self.relative_poses = []  # length T-1, since relative

        # TODO: Need an API redesign, since I want to track features over a window, instead of only consecutive time frames. Also, I want to visualize the pose in real-time

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

        # --------- keyframe TODO ---------
        self.keyframe_queue = []
        self.keyframe_poses = [np.eye(4)]

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

    def process_frame(
        self, img_left, img_right=None, depth=None, method=VOMethod.VO_3D_2D, timestamp=None
    ):
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
            # Inspired from https://github.com/niconielsen32/ComputerVision/blob/master/VisualOdometry/visual_odometry.py
            # and https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
            F1, F2 = matched_kpts_t_1, matched_kpts_t
            try:
                E, mask = cv2.findEssentialMat(F1, F2, self.K)

                # # We select only inlier points
                if mask is not None:
                    F1 = F1[mask.ravel() == 1]
                    F2 = F2[mask.ravel() == 1]
                # Decompose the Essential matrix into R and t
                _, R, t, _ = cv2.recoverPose(E, F1, F2, cameraMatrix=self.K)
                t_scaled = 0.02 * t

                T = self._form_transf(R, np.squeeze(t_scaled))
                # old_T_new
                T = np.linalg.inv(T)
                self.relative_poses.append(T)
                # w_T_k_new = w_T_k_old * k_old_T_k_new
                self.poses.append(self.poses[-1] @ self.relative_poses[-1])

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
            self.poses.append(self.poses[-1] @ self.relative_poses[-1])
            # Keyframe solution
            # # w_T_p = w_T_keyframe * keyframe_T_p
            # self.poses.append(self.keyframe_poses[-1] @ self.relative_poses[-1])
            # if len(self.poses) % 20 == 0:
            #     self.keyframe_poses.append(self.poses[-1])

        else:
            # 3D-3D method
            raise NotImplementedError("3D-3D method not implemented")

        if self.save_path:
            with open("poses.txt", "a") as f:
                pose = self.poses[-1]
                position = pose[0:3, 3]
                quaternion = Rotation.from_matrix(pose[0:3, 0:3]).as_quat()
                pose_list = list(position) + list(quaternion)
                f.write(f"{timestamp} {' '.join(map(str, pose_list))}\n")

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

        # # ---------- Local Mapping -----------
        # if len(self.GLOBAL_landmarks_3d) == 0:  # Create first set of 3d landmarks if not existent
        #     self.GLOBAL_landmarks_3d = list(world_points_3d_t)
        #     self.GLOBAL_landmarks_2d.append(
        #         {tuple(kpt): i for i, kpt in enumerate(matched_kpts_t_1)}
        #     )  # time 0
        #     self.GLOBAL_landmarks_2d.append(
        #         {tuple(kpt): i for i, kpt in enumerate(matched_kpts_t)}
        #     )  # time 1

        # else:  # Track new points from previous, and update if new landmarks are detected
        #     landmark_2d_t_1 = self.GLOBAL_landmarks_2d[-1]
        #     dic = {}
        #     for i, kpt in enumerate(matched_kpts_t_1):
        #         kpt = tuple(kpt)
        #         if kpt in landmark_2d_t_1:  # O(1) lookup
        #             idx = landmark_2d_t_1[kpt]
        #         else:
        #             # Add new 3D landmark. Note that these new 3d landmarks might only be measured once
        #             idx = len(self.GLOBAL_landmarks_3d)
        #             self.GLOBAL_landmarks_3d.append(world_points_3d_t[i])

        #         dic[tuple(matched_kpts_t[i])] = idx

        #     self.GLOBAL_landmarks_2d.append(dic)

        # # Run local mapping
        # if len(self.poses) % 30 == 0:
        #     print("local mapping")
        #     # do it only for the last 10 keyframes
        #     # self.poses[-10:], self.landmarks_3d = self.solve_better(
        #     #     self.poses[-10:], self.landmarks_2d[-10:], self.landmarks_3d
        #     # )
        #     self.poses, self.GLOBAL_landmarks_3d = self.solve_better(
        #         self.poses, self.GLOBAL_landmarks_2d, self.GLOBAL_landmarks_3d
        #     )

        return T

    def solve_better(self, poses, landmarks_2d, landmarks_3d, num_iterations=10):
        """
        Optimize poses and points in the local window. Visual odometry suffers from lots of drift, so we need to do local bundle adjusment.

        each landmark is tracked form t and t-1.

        Parameters
        ----------
        landmarks_2d_prev: Previous 2d landmarks
        landmarks_2d: array of 2d landmarks at time t
        landmarks_3d: 3d landmarks
        num_iterations (int): The maximum number of iterations to run the optimization for.
        """

        # ----------- Bundle Adjustment -----------
        print(
            f"Running Bundle Adjustment with {len(poses)} poses and {len(landmarks_3d)} landmarks"
        )

        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())  # TODO: Try PCG solver
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        # Add Camera
        cam = g2o.CameraParameters(self.fx, (self.cx, self.cy), 0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        # Add camera poses
        for i, curr_pose in enumerate(poses):
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(i)
            v_se3.set_estimate(g2o.SE3Quat(curr_pose[:3, :3], curr_pose[:3, 3]))
            if i < 2:
                v_se3.set_fixed(True)
            optimizer.add_vertex(v_se3)

        # Add 3d landmarks
        landmarks_3d_used = [False] * len(landmarks_3d)
        for i, landmark_3d in enumerate(landmarks_3d):
            # landmark_3d ~ np.array([x, y, z])
            vp = g2o.VertexPointXYZ()
            vp.set_id(len(poses) + i)
            vp.set_marginalized(True)
            vp.set_estimate(landmark_3d)
            optimizer.add_vertex(vp)

        # Add measurements
        for i, landmarks_2d_t in enumerate(landmarks_2d):
            for landmark_2d, landmark_3d_i in landmarks_2d_t.items():
                # Edge constraint
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, optimizer.vertex(len(poses) + landmark_3d_i))
                edge.set_vertex(1, optimizer.vertex(i))
                landmarks_3d_used[landmark_3d_i] = True
                edge.set_measurement(landmark_2d)
                edge.set_information(np.identity(2))
                edge.set_robust_kernel(g2o.RobustKernelHuber())
                edge.set_parameter_id(0, 0)
                optimizer.add_edge(edge)

        print("num vertices:", len(optimizer.vertices()))
        print("num edges:", len(optimizer.edges()))
        optimizer.initialize_optimization()
        optimizer.optimize(num_iterations)

        # ----------- Update State -----------
        # Camera poses
        poses = [optimizer.vertex(i).estimate().to_homogeneous_matrix() for i in range(len(poses))]

        # Landmarks positions have also shifted, so we need to update the state for all landmarks that are not fixed
        for i in range(len(landmarks_3d)):
            if landmarks_3d_used[i]:
                landmarks_3d[i] = optimizer.vertex(len(poses) + i).estimate()

        # if self.visualize:
        #     positions = [T[:3, 3] for T in poses]
        #     orientations = [T[:3, :3] for T in poses]
        #     self.vis.update(positions, orientations, landmarks_3d[-1])
        # Optimized parameters
        return poses, landmarks_3d

    def _minimize_reprojection_error(self, points_2d, points_3d):
        """
        Refer to SLAM textbook for formulation. Solved with g2o.
        """
        assert len(points_2d) == len(points_3d)
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
        # old_T_new
        T = np.linalg.inv(T)
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
        kpts_t (tuple of cv2.Keypoint): Keypoints at time t
        descs_t (ndarray): Descriptors at time t
        """
        if CUDA:
            kpts_t, descs_t = self.orb.detectAndComputeAsync(img_t, None)

            # Convert back into CPU
            if kpts_t.step == 0:
                print("Failed to process frame, No ORB keypoints found")
                return [], []
            kpts_t = self.orb.convert(kpts_t)
            descs_t = descs_t.download()
        else:
            kpts_t, descs_t = self.orb.detectAndCompute(img_t, None)

        return kpts_t, descs_t

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
        kpts_t_1, descs_t_1 = self._compute_orb(img_t_1)
        kpts_t, descs_t = self._compute_orb(img_t)

        # ---------- Match Descriptors Across 2 frames ----------
        matched_kpts_t_1, matched_kpts_t = self._match_2d_kpts(kpts_t_1, kpts_t, descs_t_1, descs_t)

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
        cv2.imwrite(f"depth/{len(self.poses)}.png", depth_viz)

    def _get_matches(
        self,
        kpts_t_1,
        kpts_t,
        descs_t_1,
        descs_t,
        ratio_threshold=0.7,
        distance_threshold=50.0,
        use_homography=True,
    ):
        # FLANN based matcher
        try:
            matches = self.flann.knnMatch(descs_t_1, descs_t, k=2)
        except:
            matches = []

        good_matches = []
        try:
            for m, n in matches:
                if m.distance < ratio_threshold * n.distance and m.distance < distance_threshold:
                    good_matches.append(m)
        except ValueError:
            pass

        print("length of matches BEFORE filtering", len(good_matches))
        if len(good_matches) < 4 or not use_homography:
            return good_matches
        # Find the homography matrix using RANSAC, needs at least 4 points
        if type(kpts_t_1[0]) == cv2.KeyPoint:
            src_pts = np.float32([kpts_t_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpts_t[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        else:
            src_pts = np.float32([kpts_t_1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpts_t[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # Use the mask to select inlier matches
        good_matches = [m for m, msk in zip(good_matches, mask) if msk[0] == 1]
        print("length of matches AFTER filtering", len(good_matches))
        return good_matches

    def _match_2d_kpts(self, kpts_t_1, kpts_t, descs_t_1, descs_t):
        """
        Match 2D keypoints across two frames using descriptors.

        Parameters
        ----------
        kpts_t_1 (list of cv2.KeyPoint): Keypoints at time t-1
        kpts_t (list of cv2.KeyPoint): Keypoints at time t
        descs_t_1 (ndarray): Descriptors at time t-1
        descs_t (ndarray): Descriptors at time t

        Returns
        -------
        matched_kpts_t_1 (ndarray of (y,x) kpts): Matched keypoints at time t-1
        matched_kpts_t (ndarray of (y,x) kpts): Matched keypoints at time t
        """
        matched_kpts_t_1 = []
        matched_kpts_t = []

        good_matches = self._get_matches(kpts_t_1, kpts_t, descs_t_1, descs_t)

        for match in good_matches:
            matched_kpts_t_1.append(kpts_t_1[match.queryIdx].pt)
            matched_kpts_t.append(kpts_t[match.trainIdx].pt)

        if self.visualize:
            img_t_1 = self.img_left_queue[-2]
            img_t = self.img_left_queue[-1]
            if CUDA:
                img_t_1 = img_t_1.download()
                img_t = img_t.download()
            output_image = cv2.drawMatches(
                img_t_1, kpts_t_1, img_t, kpts_t, good_matches[:100], None
            )
            cv2.imshow("Tracked ORB", output_image)
            cv2.imwrite(f"orb/{len(self.poses)}.png", output_image)
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
