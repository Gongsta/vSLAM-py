# Record a dataset with the Zed Camera
import numpy as np
import time
import cv2

import pyzed.sl as sl

def main():
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    # Zed Camera Paramters
    cx = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cx
    cy = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cy
    fx = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
    baseline = (
        zed.get_camera_information().camera_configuration.calibration_parameters.get_camera_baseline()
    )

    print(f"fx: {fx} baseline: {baseline}")
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    # Save the camera parameters
    np.savez("sample_zed/camera_params.npz", K=K, baseline=baseline)

    sl_stereo = sl.Mat()
    sl_depth = sl.Mat()

    cv_depth_array = []
    cv_stereo_array = []
    i = 0
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            print(i)
            zed.retrieve_image(sl_stereo, sl.VIEW.SIDE_BY_SIDE)
            zed.retrieve_measure(sl_depth, sl.MEASURE.DEPTH)
            # cv_stereo_img = sl_stereo_img.get_data()[:, :, :3] # Last channel is padded for byte alignment
            cv_stereo = sl_stereo.get_data().copy()
            cv_depth = sl_depth.get_data().copy()
            cv_stereo_array.append(cv_stereo)
            cv_depth_array.append(cv_depth)
            cv2.imwrite(f"sample_zed/stereo_{i}.jpg", cv_stereo)
            i += 1

        if (i >= 200):
            break

    np.savez_compressed("sample_zed/data.npz", stereo=cv_stereo_array, depth=cv_depth_array)

if __name__ == "__main__":
    main()
