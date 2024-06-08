import g2o
import numpy as np

np.random.seed(0)

class BundleAdjustment:
    def __init__(self, cx, cy, fx) -> None:
        # --------- Camera Parameters and Matrices ---------
        self.cx = cx
        self.cy = cy
        self.fx = fx

        self.verbose = True

    def optimize(self, keyframes, map_points, num_iterations=30, inplace=True):
        """
        Parameters
        ----------
        num_iterations (int): The maximum number of iterations to run the optimization for.
        """

        # ----------- Bundle Adjustment -----------
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        # Add Camera
        cam = g2o.CameraParameters(self.fx, (self.cx, self.cy), 0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        # Add camera poses
        # --------- Add Camera Poses ---------
        for i, keyframe in enumerate(keyframes):
            curr_pose = keyframe.pose
            curr_pose = np.linalg.inv(curr_pose)
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(i)
            v_se3.set_estimate(g2o.SE3Quat(curr_pose[:3, :3], curr_pose[:3, 3]))
            if i == 0:
                v_se3.set_fixed(True)
            optimizer.add_vertex(v_se3)

            if i == 0:  # At first frame, we don't have the previous landmark
                continue

        # --------- Add 3D Landmark position ---------
        is_valid = [True] * len(map_points)
        for i, map_point in enumerate(map_points):
            if len(map_point.observed_keyframe_ids) < 3:  # ignore unreliable points
                is_valid[i] = False
                continue
            else:
                measurements = []
                for keyframe_id in map_point.observed_keyframe_ids:
                    keyframe = keyframes[keyframe_id]
                    measurements.append(keyframe.points_3d[keyframe.map_point_ids.index(i)])
            point_3d = map_point.position
            # Landmark 3D point
            vp = g2o.VertexPointXYZ()
            vp.set_id(len(keyframes) + i)
            vp.set_marginalized(True)
            vp.set_estimate(point_3d)
            optimizer.add_vertex(vp)

        # --------- Add Edges (observations) ---------
        for i, keyframe in enumerate(keyframes):
            for j, map_point_id in enumerate(keyframe.map_point_ids):
                if map_point_id is None or not is_valid[map_point_id]:
                    continue
                # Edge constraint for current frame
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, optimizer.vertex(len(keyframes) + map_point_id))
                edge.set_vertex(1, optimizer.vertex(i))
                edge.set_measurement(keyframe.kpts[j])
                edge.set_information(np.identity(2))
                edge.set_robust_kernel(g2o.RobustKernelHuber())
                edge.set_parameter_id(0, 0)
                optimizer.add_edge(edge)

        # # --------- Add edge to constrain camera jumps ---------
        # for i in range(1, len(keyframes)):
        #     prev_pose = optimizer.vertex(i - 1).estimate().to_homogeneous_matrix()
        #     curr_pose = optimizer.vertex(i).estimate().to_homogeneous_matrix()

        #     # Calculate the relative transformation
        #     relative_transform = np.linalg.inv(prev_pose) @ curr_pose
        #     # relative_transform = np.linalg.inv(relative_transform)
        #     edge = g2o.EdgeSE3Expmap()
        #     edge.set_vertex(0, optimizer.vertex(i - 1))
        #     edge.set_vertex(1, optimizer.vertex(i))
        #     edge.set_measurement(g2o.SE3Quat(relative_transform[:3, :3], relative_transform[:3, 3]))
        #     lambda_reg = 1
        #     edge.set_information(lambda_reg * np.identity(6))
        #     optimizer.add_edge(edge)

        print("num vertices:", len(optimizer.vertices()))
        print("num edges:", len(optimizer.edges()))
        optimizer.initialize_optimization()
        # optimizer.set_verbose(self.verbose)
        optimizer.optimize(num_iterations)

        # ----------- Update State -----------
        print("Done optimization")
        # Camera poses
        for i in range(len(keyframes)):
            if self.verbose:
                print("Before BA: ", keyframes[i].pose)
                print(
                    "After BA: ", np.linalg.inv(optimizer.vertex(i).estimate().to_homogeneous_matrix())
                )
            keyframes[i].pose = np.linalg.inv(
                optimizer.vertex(i).estimate().to_homogeneous_matrix()
            )

        for i in range(len(map_points)):
            if is_valid[i]:
                # if self.verbose:
                # print(f"Before: {map_points[i].position}")
                # print(f"After: {optimizer.vertex(len(keyframes) + i).estimate()}")

                map_points[i].position = optimizer.vertex(len(keyframes) + i).estimate()

        # Optimized parameters
        return keyframes, map_points
