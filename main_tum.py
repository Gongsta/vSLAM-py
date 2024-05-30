import os
import multiprocessing as mp
import numpy as np
from scipy.spatial.transform import Rotation as R

np.random.seed(0)


import cv2


from utils import download_file, extract
from eval.associate import read_file_list, associate


def main():
    mp.set_start_method(
        "spawn", force=True
    )  # Required to get Zed and Pangolin working in different processes

    # --------- Download Dataset ---------
    dataset_name = "rgbd_dataset_freiburg1_xyz" # for debugging
    # dataset_name = "rgbd_dataset_freiburg1_rpy" # for debugging
    # dataset_name = "rgbd_dataset_freiburg1_desk"
    # dataset_name = "rgbd_dataset_freiburg1_room"
    # dataset_name = "rgbd_dataset_freiburg3_walking_static"
    # dataset_name = "rgbd_dataset_freiburg3_long_office_household" # big dataset
    if not os.path.exists(dataset_name):
        path_to_zip_file = f"{dataset_name}.zip"
        download_file(
            f"https://cvg.cit.tum.de/rgbd/dataset/freiburg{dataset_name[21]}/{dataset_name}.tgz",
            path_to_zip_file,
        )
        extract(path_to_zip_file)

    # --------- Load Dataset Images ---------
    # Taken from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    factor = 5000  # for the 16-bit PNG files

    depth_image_paths = read_file_list(f"{dataset_name}/depth.txt")
    rgb_image_paths = read_file_list(f"{dataset_name}/rgb.txt")
    gt_paths = read_file_list(f"{dataset_name}/groundtruth.txt")

    print("Associating depth and rgb images")
    matches = associate(depth_image_paths, rgb_image_paths, 0.0, 0.02)
    gt_matches = associate(depth_image_paths, gt_paths, 0.0, 0.02)

    depth_images = []
    rgb_images = []
    timestamps = []
    gt_poses = []

    gt_dict = {}
    with open(f"{dataset_name}/groundtruth.txt") as file:
        for line in file:
            if line[0] == "#":
                continue
            pose = np.eye(4)
            pose_list = list(map(float, line.split()))
            pose[:3, 3] = pose_list[1:4]
            pose[:3, :3] = R.from_quat(pose_list[4:]).as_matrix()
            gt_dict[float(pose_list[0])] = pose

    for depth_timestamp, rgb_timestamp in matches.items():
        if depth_timestamp not in gt_matches:
            continue

        depth_image = (
            cv2.imread(
                dataset_name + "/" + depth_image_paths[depth_timestamp][0],
                cv2.IMREAD_UNCHANGED,
            )
            / factor
        )
        # A pixel value of 0 means missing value/no data.
        depth_image[depth_image == 0] = np.NaN
        depth_images.append(depth_image)

        rgb_image = cv2.imread(dataset_name + "/" + rgb_image_paths[rgb_timestamp][0])
        rgb_images.append(rgb_image)

        timestamps.append(depth_timestamp)
        gt_poses.append(gt_dict[gt_matches[depth_timestamp]])

    initial_pose = gt_poses[0]

    # --------- Queues for sharing data across Processes ---------
    # image grabber -> frontend
    cv_img_queue = mp.Queue()
    # frontend -> renderer
    renderer_queue = mp.Queue()

    # frontend -> loop_closure
    descriptors_queue = mp.Queue()
    # frontend -> visualizer
    vis_queue = mp.Queue()

    # Create a Manager object to manage shared state
    manager = mp.Manager()
    shared_data = manager.dict()
    shared_data["new_keyframe"] = []
    shared_data["keyframes"] = []
    shared_data["map_points"] = []
    shared_data["lock"] = manager.Lock()

    # Events
    new_keyframe_event = mp.Event()
    map_done_optimization_event = mp.Event()


    # --------- Processes ---------
    from main_zed_slam import (
        grab_rgbd_images_sim,
        process_frontend,
        process_tracking,
        process_backend,
        render,
        visualize_path,
    )

    image_grabber = mp.Process(
        target=grab_rgbd_images_sim, args=(rgb_images, depth_images, timestamps, cv_img_queue)
    )

    renderer_proc = mp.Process(target=render, args=(renderer_queue,))

    frontend_proc = mp.Process(
        target=process_frontend,
        args=(
            cv_img_queue,
            vis_queue,
            renderer_queue,
            descriptors_queue,
            cx,
            cy,
            fx,
            1,  # baseline, ignored
            initial_pose,
        ),
    )


    tracking_proc = mp.Process(
        target=process_tracking,
        args=(
            cv_img_queue,
            new_keyframe_event,
            map_done_optimization_event,
            shared_data,
            vis_queue,
            renderer_queue,
            descriptors_queue,
            cx,
            cy,
            fx,
            1,
            initial_pose,
        ),
    )

    backend_proc = mp.Process(
        target=process_backend,
        args=(new_keyframe_event, map_done_optimization_event, shared_data, cx, cy, fx),
    )


    path_visualizer_proc = mp.Process(target=visualize_path, args=(vis_queue, gt_poses))

    image_grabber.start()
    # frontend_proc.start()
    tracking_proc.start()

    FAST = False
    if not FAST:
        path_visualizer_proc.start()
        renderer_proc.start()
    # backend_proc.start()

    image_grabber.join()
    # frontend_proc.join()
    tracking_proc.join()
    if not FAST:
        path_visualizer_proc.join()
        renderer_proc.join()
        # loop_closure_proc.join()
    # backend_proc.join()
    print("reached here")


if __name__ == "__main__":
    main()
