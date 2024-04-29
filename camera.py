import numpy as np
import os
import cv2

# Can be a real camera, or a camera with a pre-fed feed
class Camera:
    def __init__(self) -> None:
        pass
        # # Camera matrices
        # # Reprojection Matrix
        # self.Q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, -fx], [0, 0, -1.0 / baseline, 0]])
        # self.P = np.array([[fx, 0, cx, 0], [0, fx, cy, 0], [0, 0, 1, 0]])

        # # Calibration Matrix
    def read(self):
        # TO BE IMPLEMENTED
        # different implementations
        # ret, frame = cap.read()
        pass

class MonoDatasetCamera(Camera):
    """Camera that plays from a dataset"""
    def __init__(self, filepath):
        super().__init__()
        self.images = self._load_images()

    def _load_images(self, filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P


    def grab(self):
        return
