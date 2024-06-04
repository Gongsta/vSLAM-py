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

    # --------- Queues for sharing data across Processes ---------
    cv_img_queue = mp.Queue(maxsize=1)
    renderer_queue = mp.Queue(maxsize=1)
    hand_poses_queue = mp.Queue(maxsize=1)
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
    image_grabber = mp.Process(
        target=grab_images_realtime,
        args=(cv_img_queue, vis_queue, renderer_queue),
    )
    hand_tracker = mp.Process(
        target=hand_tracker,
        args=(cv_img_queue, hand_poses_queue)

    )

    renderer_proc = mp.Process(target=render, args=(renderer_queue, cx, cy, fx))

    path_visualizer_proc = mp.Process(target=visualize_path, args=(vis_queue,))

    image_grabber.start()
    path_visualizer_proc.start()
    renderer_proc.start()
    hand_tracker.start()

    image_grabber.join()
    path_visualizer_proc.join()
    renderer_proc.join()


def hand_tracker(hand_input_queue, hand_poses_queue):
    import cv2
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    import numpy as np
    import time
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python#live-stream_1
    # https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


    def draw_landmarks_on_image(rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image


    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode


    import os
    import urllib.request

    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    if not os.path.exists("hand_landmarker.task"):
        urllib.request.urlretrieve(url, "hand_landmarker.task")
        print(f"model downloaded.")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )

    detector = vision.HandLandmarker.create_from_options(options)

    while True:
        cv_img = hand_input_queue.get()
        start = time.time()
        cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB, cv_img)  # Input is BGR, but mediapipe expects RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_img)
        detection_result = detector.detect(image)

        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow("wow", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)
        if key == "q":
            running = False
            break
        curr = time.time()
        latency = 1.0 / (curr - start)
        print(f"Running at {latency} hz")
        start = curr



def grab_images_realtime(cv_img_queue, vis_queue, renderer_queue):
    import pyzed.sl as sl  # local import

    # Sharing a zed object between different process is iffy, so we'll fix it to a single isolated process
    # https://community.stereolabs.com/t/python-multiprocessing-bug-fix/4310/6
    zed = sl.Camera()
    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    # init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.coordinate_units = sl.UNIT.METER

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    # Enable positional tracking with default parameters
    py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable positional tracking : " + repr(err) + ". Exit program.")
        zed.close()
        exit()

    # Zed Camera Paramters
    image_size = zed.get_camera_information().camera_configuration.resolution
    sl_stereo_img = sl.Mat()
    sl_depth = sl.Mat()
    zed_pose = sl.Pose()
    poses = []

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns

            zed.retrieve_image(sl_stereo_img, sl.VIEW.SIDE_BY_SIDE)
            zed.retrieve_measure(sl_depth, sl.MEASURE.DEPTH)
            zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
            cv_stereo_img = sl_stereo_img.get_data()[
                :, :, :3
            ]  # Last channel is padded for byte alignment
            cv_img_left = cv_stereo_img[:, : image_size.width, :]
            cv_depth = sl_depth.get_data()

            rotation_matrix = zed_pose.get_rotation_matrix().r
            translation = zed_pose.get_translation().get()
            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = translation
            poses.append(T)
            print(poses[-1])

        else:
            break

        cv_img_queue.put(cv_img_left)
        renderer_queue.put((cv_img_left, poses[-1]))
        vis_queue.put((poses.copy(), None))


def visualize_path(vis_queue, gt_poses=None):
    from visualization import PangoVisualizer  # local import

    vis = PangoVisualizer(title="Path Visualizer")
    while True:
        poses, landmarks = vis_queue.get()
        # vis.update(poses, landmarks, gt_poses[: len(poses)])
        vis.update(poses, landmarks, gt_poses)


def render(cv_img_queue, cx, cy, fx):
    from render import Renderer

    renderer = None

    while True:
        image, pose = cv_img_queue.get()
        width, height = image.shape[1], image.shape[0]
        if renderer is None:
            renderer = Renderer(width=width, height=height)
        renderer.update(pose, image)


if __name__ == "__main__":
    main()
