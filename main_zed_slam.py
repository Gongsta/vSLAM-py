import numpy as np
from argparse import ArgumentParser

import multiprocessing as mp

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

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
    cv_img_queue = mp.Queue(maxsize=5)
    # frontend -> renderer
    renderer_queue = mp.Queue(maxsize=1)

    # frontend -> loop_closure
    descriptors_queue = mp.Queue()
    # frontend -> visualizer
    vis_queue = mp.Queue(maxsize=1)

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
            baseline,
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
            baseline,
        ),
    )

    backend_proc = mp.Process(
        target=process_backend,
        args=(new_keyframe_event, map_done_optimization_event, shared_data, cx, cy, fx),
    )

    path_visualizer_proc = mp.Process(target=visualize_path, args=(vis_queue,))

    image_grabber.start()
    frontend_proc.start()
    path_visualizer_proc.start()
    renderer_proc.start()
    # backend_proc.start()

    image_grabber.join()
    frontend_proc.join()
    path_visualizer_proc.join()
    renderer_proc.join()
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
        import cv2
        cv2.imwrite(f"raw/{image_counter}.png", cv_img_left)


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
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
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
        # vis.update(poses, landmarks, gt_poses[: len(poses)])
        vis.update(poses, landmarks, gt_poses)


def process_frontend(
    cv_img_queue,
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

        if counter > 1:
            renderer_queue.put((cv_img_left, vo.poses[-1]))
            vis_queue.put((vo.poses.copy(), vo.landmarks_3d[-1].copy()))

        import cv2
        key = cv2.waitKey(1)
        if key == "q":
            break


def process_tracking(
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
    baseline=1,  # not used if not using stereo images
    initial_pose=np.eye(4),
):

    from tracking import Tracking

    tracker = Tracking(cx, cy, fx, baseline, initial_pose)
    counter = 0
    while True:
        cv_img_left, cv_depth, timestamp = cv_img_queue.get()
        tracker.track(
            cv_img_left,
            cv_depth,
            timestamp,
        )
        counter += 1

        if tracker.new_keyframe_event:  # multiprocessing enabled
            with shared_data["lock"]:
                shared_data["new_keyframe"] = tracker.new_keyframe
            new_keyframe_event.set()  # notify backend
            tracker.new_keyframe_event = False

        # Check if map optimization is done
        if map_done_optimization_event.is_set():
            # Access updated map
            with shared_data["lock"]:
                tracker.synchronize(shared_data["keyframes"], shared_data["map_points"])
            map_done_optimization_event.clear()

        renderer_queue.put((cv_img_left, tracker.frames[-1].pose))
        map_points = [pt.position for pt in tracker.map_points]
        poses = [frame.pose for frame in tracker.frames]
        # poses = [frame.pose for frame in tracker.keyframes]
        vis_queue.put((poses, map_points))

        # if counter % 50 == 0:  # Run backend every 50 frames
        #     frontend_backend_queue.put(
        #         (
        #             vo.poses.copy(),
        #             vo.landmarks_2d.copy(),
        #             vo.landmarks_3d.copy(),
        #         )
        #     )

        # backend might be using an older version of the poses and landmarks, since backend is non-blocking
        # BAD: Using the empty() function is unreliable
        # if not backend_frontend_queue.empty():
        #     poses, landmarks_3d = backend_frontend_queue.get()
        #     vo.poses = poses
        #     vo.landmarks_3d = landmarks_3d

        # kpts_t, descs_t = vo._compute_orb(cv_img_left)
        # descriptors_queue.put(descs_t)

        import cv2
        key = cv2.waitKey(1)
        if key == "q":
            break


def process_backend(new_keyframe_event, map_done_optimization_event, shared_data, cx, cy, fx):
    from tracking import Map

    map = Map(new_keyframe_event, map_done_optimization_event, shared_data, cx, cy, fx)
    while True:
        if new_keyframe_event.is_set():
            with shared_data["lock"]:
                new_keyframe = shared_data["new_keyframe"]
                shared_data["new_keyframe"] = []

            new_keyframe_event.clear()
            map.add_keyframe(new_keyframe)

            # --------- Bundle Adjustment ---------
            map.optimize()
            with shared_data["lock"]:
                shared_data["keyframes"] = map.keyframes  # optimized keyframes
                shared_data["map_points"] = map.map_points  # optimized landmarks
            map_done_optimization_event.set()

    # curr = time.time()
    # latency = 1.0 / (curr - start)
    # print(f"Running at {latency} hz")
    # start = curr


if __name__ == "__main__":
    main()
