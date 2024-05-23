import numpy as np
from argparse import ArgumentParser

import multiprocessing as mp

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

import cv2
import os

# Using some local imports to prevent interactions amongst libraries, specifically pangolin and pyzed

USE_SIM = True

from utils import download_file


def main():
    mp.set_start_method(
        "spawn", force=True
    )  # Required to get Zed and Pangolin working in different processes
    parser = ArgumentParser()
    parser.add_argument("--visualize", default=True, action="store_true", help="Show visualization")
    args = parser.parse_args()

    # --------- Init Zed Camera ---------
    if not os.path.isfile("sample_zed/data.npz"):
        os.makedirs("sample_zed", exist_ok=True)
        download_file(
            "https://github.com/Gongsta/Datasets/raw/main/sample_zed/camera_params.npz",
            "sample_zed/camera_params.npz",
        )
        download_file(
            "https://github.com/Gongsta/Datasets/raw/main/sample_zed/data.npz",
            "sample_zed/data.npz",
        )

    data = np.load("sample_zed/data.npz")
    calibration = np.load("sample_zed/camera_params.npz")

    stereo_images = data["stereo"]
    depth_images = data["depth"]
    timestamps = [i for i in range(len(stereo_images))]

    K = calibration["K"]
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    baseline = calibration["baseline"]
    # else:
    # Importing zed at the top of the file causes issues with pangolin

    # zed = sl.Camera()
    # # Set configuration parameters
    # init_params = sl.InitParameters()
    # init_params.camera_resolution = sl.RESOLUTION.VGA
    # # Open the camera
    # err = zed.open(init_params)
    # if err != sl.ERROR_CODE.SUCCESS:
    #     print(repr(err))
    #     zed.close()
    #     exit(1)

    # # Zed Camera Paramters
    # cx = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cx
    # cy = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cy
    # fx = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
    # baseline = (
    #     zed.get_camera_information().camera_configuration.calibration_parameters.get_camera_baseline()
    # )

    # zed.close()

    # --------- Queues for sharing data across Processes ---------
    # image grabber -> frontend
    cv_img_queue = mp.Queue()
    # frontend -> renderer
    renderer_queue = mp.Queue()
    frontend_backend_queue = mp.Queue()
    backend_frontend_queue = mp.Queue()

    # frontend -> loop_closure
    descriptors_queue = mp.Queue()
    # loop_closure -> backend
    loop_closure_queue = mp.Queue()
    # frontend -> visualizer
    vis_queue = mp.Queue()

    # --------- Processes ---------
    if USE_SIM:
        image_grabber = mp.Process(
            target=grab_stereo_images_sim,
            args=(stereo_images, depth_images, timestamps, cv_img_queue),
        )
    else:
        image_grabber = mp.Process(
            target=grab_images_realtime,
            args=(cv_img_queue,),
        )

    renderer_proc = mp.Process(target=render, args=(renderer_queue,))
    loop_closure_proc = mp.Process(
        target=loop_closure, args=(descriptors_queue, loop_closure_queue)
    )

    frontend_proc = mp.Process(
        target=process_frontend,
        args=(
            cv_img_queue,
            frontend_backend_queue,
            backend_frontend_queue,
            vis_queue,
            renderer_queue,
            descriptors_queue,
            cx,
            cy,
            fx,
            baseline,
        ),
    )
    backend_proc = mp.Process(
        target=process_backend, args=(frontend_backend_queue, backend_frontend_queue, cx, cy, fx)
    )

    path_visualizer_proc = mp.Process(target=visualize_path, args=(vis_queue,))

    image_grabber.start()
    frontend_proc.start()
    path_visualizer_proc.start()
    renderer_proc.start()
    loop_closure_proc.start()
    # backend_proc.start()

    image_grabber.join()
    frontend_proc.join()
    path_visualizer_proc.join()
    renderer_proc.join()
    loop_closure_proc.join()
    # backend_proc.join()


def render(cv_img_queue):
    from render import Renderer

    renderer = None

    while True:
        image, pose = cv_img_queue.get()
        width, height = image.shape[1], image.shape[0]
        if renderer is None:
            renderer = Renderer(width=width, height=height)
        renderer.update(pose, image)


def grab_rgbd_images_sim(rgb_images, depth_images, timestamps, cv_img_queue):
    # --------- Grag Images ---------
    image_counter = 0
    while True:
        cv_img_left = rgb_images[image_counter]
        cv_depth = depth_images[image_counter]
        timestamp = timestamps[image_counter]

        image_counter += 1
        if image_counter >= len(rgb_images):
            break

        cv_img_queue.put((cv_img_left, cv_depth, timestamp))


def grab_stereo_images_sim(stereo_images, depth_images, timestamps, cv_img_queue):
    # --------- Grag Images ---------
    image_counter = 0
    while True:
        cv_stereo_img = stereo_images[image_counter]
        timestamp = timestamps[image_counter]
        cv_img_left = cv_stereo_img[:, : cv_stereo_img.shape[1] // 2, :]
        cv_depth = depth_images[image_counter]

        image_counter += 1
        if image_counter >= len(stereo_images):
            break

        cv_img_queue.put((cv_img_left, cv_depth, timestamp))


def grab_images_realtime(cv_img_queue):
    import pyzed.sl as sl  # local import

    # Sharing a zed object between different process is iffy, so we'll fix it to a single isolated process
    # https://community.stereolabs.com/t/python-multiprocessing-bug-fix/4310/6
    zed = sl.Camera()
    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 100
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    # Zed Camera Paramters
    image_size = zed.get_camera_information().camera_configuration.resolution
    sl_stereo_img = sl.Mat()
    sl_depth = sl.Mat()
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)
            zed.retrieve_image(sl_stereo_img, sl.VIEW.SIDE_BY_SIDE)
            zed.retrieve_measure(sl_depth, sl.MEASURE.DEPTH)
            cv_stereo_img = sl_stereo_img.get_data()[
                :, :, :3
            ]  # Last channel is padded for byte alignment
            cv_img_left = cv_stereo_img[:, : image_size.width, :]
            # cv_img_right = cv_stereo_img[:, image_size.width:, :]
            cv_depth = sl_depth.get_data()

        else:
            break

        cv_img_queue.put((cv_img_left, cv_depth, timestamp))


def visualize_path(vis_queue, gt_poses=None):
    from visualization import PangoVisualizer  # local import

    vis = PangoVisualizer(title="Path Visualizer")
    while True:
        poses, landmarks = vis_queue.get()
        vis.update(poses, landmarks[-1], gt_poses[: len(poses)])


def process_frontend(
    cv_img_queue,
    frontend_backend_queue,
    backend_frontend_queue,
    vis_queue,
    renderer_queue,
    descriptors_queue,
    cx,
    cy,
    fx,
    baseline=1,  # not used if not using stereo images
    initial_pose=np.eye(4),
):

    from frontend import VisualOdometry, VOMethod

    vo = VisualOdometry(cx, cy, fx, baseline, initial_pose)
    counter = 0
    while True:
        cv_img_left, cv_depth, timestamp = cv_img_queue.get()
        T = vo.process_frame(
            cv_img_left,
            img_right=None,
            depth=cv_depth,
            timestamp=timestamp,
            method=VOMethod.VO_3D_2D,
        )
        counter += 1

        renderer_queue.put((cv_img_left, vo.poses[-1]))
        vis_queue.put((vo.poses.copy(), vo.landmarks_3d.copy()))

        if counter % 50 == 0:  # Run backend every 50 frames
            frontend_backend_queue.put(
                (
                    vo.poses.copy(),
                    vo.landmarks_2d.copy(),
                    vo.landmarks_3d.copy(),
                )
            )

        # backend might be using an older version of the poses and landmarks, since backend is non-blocking
        # BAD: Using the empty() function is unreliable
        # if not backend_frontend_queue.empty():
        #     poses, landmarks_3d = backend_frontend_queue.get()
        #     vo.poses = poses
        #     vo.landmarks_3d = landmarks_3d

        kpts_t, desc_t = vo._compute_orb(cv_img_left)
        descriptors_queue.put(desc_t)

        key = cv2.waitKey(1)
        if key == "q":
            break


def mock_process_frontend(
    cv_img_queue,
    gt_poses,
    vis_queue,
    renderer_queue,
):
    counter = 0
    while True:
        cv_img_left, cv_depth, timestamp = cv_img_queue.get()
        renderer_queue.put((cv_img_left, gt_poses[counter]))
        vis_queue.put((gt_poses[: counter + 1].copy(), [None]))
        counter += 1


def process_backend(frontend_backend_queue, backend_frontend_queue, cx, cy, fx):
    from backend import BundleAdjustment  # local import

    backend = BundleAdjustment(cx, cy, fx)
    while True:
        poses, landmarks_2d, landmarks_3d = frontend_backend_queue.get()
        print("backend", len(poses))
        # poses, landmarks_3d = backend.solve(poses, landmarks_2d, landmarks_3d)
        # backend_frontend_queue.put((poses, landmarks_3d))

    # curr = time.time()
    # latency = 1.0 / (curr - start)
    # print(f"Running at {latency} hz")
    # start = curr


def loop_closure(descriptors_queue, loop_closure_queue):
    from loop_closure import LoopClosure  # local import

    lc = LoopClosure()
    while True:
        descriptors = descriptors_queue.get()
        scores, max_idx = lc.query(descriptors)
        if scores is not None:
            print("Loop Closure Detected")
            loop_closure_queue.put((scores, max_idx))


if __name__ == "__main__":
    main()
