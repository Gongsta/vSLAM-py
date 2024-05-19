import numpy as np
from argparse import ArgumentParser

import multiprocessing as mp

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

import cv2
import os

# Using some local imports to prevent interactions amongst libraries, specifically pangolin and pyzed

USE_SIM = False

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
    cv_img_queue = mp.Queue()
    frontend_backend_queue = mp.Queue()
    backend_frontend_queue = mp.Queue()
    vis_queue = mp.Queue()

    # --------- Processes ---------
    if USE_SIM:
        image_grabber = mp.Process(
            target=grab_images_sim, args=(stereo_images, depth_images, cv_img_queue)
        )
    else:
        image_grabber = mp.Process(
            target=grab_images_realtime,
            args=(cv_img_queue,),
        )

    frontend_proc = mp.Process(
        target=process_frontend,
        args=(
            cv_img_queue,
            frontend_backend_queue,
            backend_frontend_queue,
            vis_queue,
            cx,
            cy,
            fx,
            baseline,
        ),
    )
    backend_proc = mp.Process(
        target=process_backend, args=(frontend_backend_queue, backend_frontend_queue, cx, cy, fx)
    )

    visualizer_proc = mp.Process(target=visualize, args=(vis_queue,))

    image_grabber.start()
    frontend_proc.start()
    visualizer_proc.start()
    # backend_proc.start()

    image_grabber.join()
    frontend_proc.join()
    visualizer_proc.join()
    # backend_proc.join()


def grab_images_sim(stereo_images, depth_images, cv_img_queue):
    # --------- Grag Images ---------
    image_counter = 0
    while True:
        cv_stereo_img = stereo_images[image_counter]
        cv_img_left = cv_stereo_img[:, : cv_stereo_img.shape[1] // 2, :]
        cv_depth = depth_images[image_counter]

        image_counter += 1
        if image_counter >= len(stereo_images):
            break

        cv_img_queue.put((cv_img_left, cv_depth))


def grab_images_realtime(cv_img_queue):
    import pyzed.sl as sl # local import

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

        cv_img_queue.put((cv_img_left, cv_depth))


def visualize(vis_queue):
    from visualization import PangoVisualizer # local import

    vis = PangoVisualizer(title="Frontend Visualizer")
    while True:
        poses, landmarks = vis_queue.get()
        positions = [T[:3, 3] for T in poses]
        orientations = [T[:3, :3] for T in poses]
        vis.update(positions, orientations, landmarks[-1])
        # vis.update(positions, orientations, landmarks)


def process_frontend(
    cv_img_queue, frontend_backend_queue, backend_frontend_queue, vis_queue, cx, cy, fx, baseline
):

    from frontend import VisualOdometry
    vo = VisualOdometry(cx, cy, fx, baseline)
    counter = 0
    while True:
        cv_img_left, cv_depth = cv_img_queue.get()
        T = vo.process_frame(cv_img_left, img_right=None, depth=cv_depth)
        counter += 1

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

        key = cv2.waitKey(1)
        if key == "q":
            break


def process_backend(frontend_backend_queue, backend_frontend_queue, cx, cy, fx):
    from backend import BundleAdjustment # local import

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


if __name__ == "__main__":
    main()
