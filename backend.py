import g2o
import numpy as np
from visualization import PangoVisualizer


class BundleAdjustment:
    def __init__(self, cx, cy, fx) -> None:

        # --------- Visualization ---------
        self.visualize = True
        self.vis = PangoVisualizer()

        # --------- Camera Parameters and Matrices ---------
        self.cx = cx
        self.cy = cy
        self.fx = fx

    def solve(self, poses, landmarks_2d_prev, landmarks_2d, landmarks_3d, num_iterations=30):
        """
        Optimize poses and points in the local window. Visual odometry suffers from lots of drift, so we need to do local bundle adjusment.

        each landmark is tracked form t and t-1.

        Parameters
        ----------
        landmarks_2d_prev: Previous 2d landmarks
        landmarks_2d: 2d landmarks
        landmarks_3d: 3d landmarks
        num_iterations (int): The maximum number of iterations to run the optimization for.
        """

        # ----------- Bundle Adjustment -----------
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())  # TODO: Try PCG solver
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        # Add Camera
        cam = g2o.CameraParameters(self.fx, (self.cx, self.cy), 0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        # Add camera poses
        point_id = len(poses)

        for i, curr_pose in enumerate(poses):
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(i)
            v_se3.set_estimate(g2o.SE3Quat(curr_pose[:3, :3], curr_pose[:3, 3]))
            if i < 2:
                v_se3.set_fixed(True)
            optimizer.add_vertex(v_se3)

            if i == 0:  # At first frame, we don't have the previous landmark
                continue

            points_2d_prev = landmarks_2d_prev[i]
            points_2d = landmarks_2d[i]
            points_3d = landmarks_3d[i]

            for point_2d_prev, point_2d, point_3d in zip(points_2d_prev, points_2d, points_3d):
                point_id += 1
                # Landmark 3D point
                vp = g2o.VertexPointXYZ()
                vp.set_id(point_id)
                vp.set_marginalized(True)
                vp.set_estimate(point_3d)
                optimizer.add_vertex(vp)

                # Edge constraint for current frame
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, vp)
                edge.set_vertex(1, optimizer.vertex(i))
                edge.set_measurement(point_2d)
                edge.set_information(np.identity(2))
                edge.set_robust_kernel(g2o.RobustKernelHuber())
                edge.set_parameter_id(0, 0)
                optimizer.add_edge(edge)

                if i > 0:  # Edge constraint for previous frame
                    edge = g2o.EdgeProjectXYZ2UV()
                    edge.set_vertex(0, vp)
                    edge.set_vertex(1, optimizer.vertex(i - 1))
                    edge.set_measurement(point_2d_prev)
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
        poses = [
            optimizer.vertex(i).estimate().to_homogeneous_matrix() for i in range(len(poses))
        ]

        # Landmarks positions have also shifted, so we need to update the state for all landmarks
        point_id = len(poses)
        for i in range(1, len(poses)):
            new_points_3d = []
            for j in range(len(landmarks_3d[i])):
                point_id += 1
                new_points_3d.append(optimizer.vertex(point_id).estimate())

            landmarks_3d[i] = np.array(new_points_3d)

        if self.visualize:
            positions = [T[:3, 3] for T in poses]
            orientations = [T[:3, :3] for T in poses]
            self.vis.update(positions, orientations, landmarks_3d[-1])
        # Optimized parameters
        return poses, landmarks_3d
