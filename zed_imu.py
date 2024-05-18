from visualization import PangoVisualizer
from scipy.spatial.transform import Rotation as R
import time
import pyzed.sl as sl


zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 100
init_params.depth_mode = sl.DEPTH_MODE.NONE
init_params.coordinate_units = sl.UNIT.METER

# Basic class to handle the timestamp of the different sensors to know if it is a new sensors_data or an old one
class TimestampHandler:
    def __init__(self):
        self.t_imu = None
        self.time_elapsed = None

    # check if the new timestamp is higher than the reference one, and if yes, save the current as reference
    def is_new(self, sensor):
        if isinstance(sensor, sl.IMUData):
            if (self.t_imu is None):
                new_ = True
                self.time_elapsed = 0
            else:
                new_ = sensor.timestamp.get_microseconds() > self.t_imu.get_microseconds()
                if new_:
                    self.time_elapsed = sensor.timestamp.get_microseconds() - self.t_imu.get_microseconds()

            self.t_imu = sensor.timestamp
            return new_


# vis = PangoVisualizer()

import numpy as np
np.set_printoptions(precision=3)




def imu_thread():
    # Open the camera
    err = zed.open(init_params)
    ts_handler = TimestampHandler()
    sensors_data = sl.SensorsData()
    pose = np.eye(4)
    positions = []
    orientations = []
    while zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
        # Check if the data has been updated since the last time
        # IMU is the sensor with the highest rate
        new_imu_data = sensors_data.get_imu_data()
        if ts_handler.is_new(new_imu_data):
            dt = ts_handler.time_elapsed / 1e6
            quaternion = new_imu_data.get_pose().get_orientation().get()
            r = R.from_quat(quaternion).as_matrix()
            linear_acceleration = np.array(new_imu_data.get_linear_acceleration())
            print(linear_acceleration)
            t = linear_acceleration * (dt ** 2)

            T = np.eye(4)
            T[:3, :3] = r
            T[:3, 3] = t
            pose = T @ pose
            positions.append(pose[:3, 3])
            orientations.append(pose[:3, :3])
            # print(positions[-1])


imu_thread()
