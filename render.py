import numpy as np
import pypangolin as pango
from OpenGL.GL import *
from scipy.spatial.transform import Rotation as R


class Renderer:
    """Renders the display that one would see in the VR Headset."""

    def __init__(self, title="VR Display", width=672, height=376) -> None:
        self.debug = True
        self.win = pango.CreateWindowAndBind(title, width, height)
        self.width = width
        self.height = height

        glEnable(GL_DEPTH_TEST)

        self.fx = 420
        self.pm = pango.ProjectionMatrix(
            width, height, self.fx, self.fx, width / 2, height / 2, 0.1, 1000
        )  # width, height, fx, fy, cx, cy, near clip, far clip
        self.mv = pango.ModelViewLookAt(
            0.0, 0.0, 0.0, 1, 0, 0, pango.AxisZ
        )  # Using Canonical Frame (Z-up)
        self.s_cam = pango.OpenGlRenderState(self.pm, self.mv)

        self.handler = pango.Handler3D(self.s_cam)
        self.d_cam = (
            pango.CreateDisplay()
            .SetBounds(
                pango.Attach(0),
                pango.Attach(1),
                pango.Attach(0),
                pango.Attach(1),
                -width / height,
            )
            .SetHandler(self.handler)
        )

        self.texture = pango.GlTexture(width, height, GL_RGB, False, 0, GL_RGB, GL_UNSIGNED_BYTE)

        apple_img = cv2.imread("apple.jpg")
        self.apple_texture = pango.GlTexture(
            apple_img.shape[1], apple_img.shape[0], GL_RGB, False, 0, GL_RGB, GL_UNSIGNED_BYTE
        )
        self.apple_texture.Upload(apple_img, GL_BGR, GL_UNSIGNED_BYTE)

        self.pose = np.eye(4)  # w_T_k

    def update(self, pose, image):
        # ------- Update Camera Position and Orientation -------
        self.move_camera(pose)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.d_cam.Activate(self.s_cam)

        # ------- Render Background -------
        self.render_background_video(image)

        # ------- Render 3D Objects -------
        apple_img = cv2.imread("apple.jpg")
        self.render_apple(apple_img.shape[1], apple_img.shape[0], 0.1)

        pango.FinishFrame()

    def move_camera(self, pose):
        self.pose = pose
        # Pango uses axis-angle representation
        position = pose[:3, 3]
        t = (self.pose @ np.array([1, 0, 0, 1])).T
        print(t)
        self.mv = pango.ModelViewLookAt(
            position[0], position[1], position[2], t[0], t[1], t[2], pango.AxisZ
        )
        self.s_cam.SetModelViewMatrix(self.mv)

    def render_background_video(self, image):
        # Draw the texture on a large quad in the background
        self.texture.Upload(image[:, :, :3].copy(), GL_BGR, GL_UNSIGNED_BYTE)

        glEnable(GL_TEXTURE_2D)
        self.texture.Bind()
        glBegin(GL_QUADS)
        # Mapping from 2D to 3D
        z = 10.0  # fixed depth to 10 meters away. Note the coordinates are in canonical frame.
        # p_w = w_T_k * p_k

        # First vertex
        p_o = np.array([[-self.width / 2 * z / self.fx, self.height / 2 * z / self.fx, z, 1]]).T
        # coordinates in optical frame, from camera perspective

        canonical_T_optical = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        p_c = canonical_T_optical @ p_o
        p_w = self.pose @ p_c  # world frame coordinates
        glTexCoord2f(0.0, 0.0)
        glVertex3f(p_w[0], p_w[1], p_w[2])

        # Second vertex
        p_o = np.array([[-self.width / 2 * z / self.fx, -self.height / 2 * z / self.fx, z, 1]]).T
        p_c = canonical_T_optical @ p_o
        p_w = self.pose @ p_c
        glTexCoord2f(0.0, 1.0)
        glVertex3f(p_w[0], p_w[1], p_w[2])

        # Third vertex
        p_o = np.array([[self.width / 2 * z / self.fx, -self.height / 2 * z / self.fx, z, 1]]).T
        p_c = canonical_T_optical @ p_o
        p_w = self.pose @ p_c
        glTexCoord2f(1.0, 1.0)
        glVertex3f(p_w[0], p_w[1], p_w[2])

        # Fourth vertex
        p_o = np.array([[self.width / 2 * z / self.fx, self.height / 2 * z / self.fx, z, 1]]).T
        p_c = canonical_T_optical @ p_o
        p_w = self.pose @ p_c
        glTexCoord2f(1.0, 0.0)
        glVertex3f(p_w[0], p_w[1], p_w[2])

        glEnd()
        glDisable(GL_TEXTURE_2D)

    def render_apple(self, width, height, scale):
        """
        TODO: Figure this out
        """
        # Draw the texture on a large quad in the background
        glEnable(GL_TEXTURE_2D)
        self.apple_texture.Bind()
        glBegin(GL_QUADS)
        # Mapping from 2D to 3D
        z = 1.0
        x = width / 2 * z / self.fx * scale
        y = height / 2 * z / self.fx * scale
        glTexCoord2f(0.0, 0.0)
        glVertex3f(z, -x, y)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(z, -x, -y)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(z, x, -y)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(z, x, y)
        glEnd()
        glDisable(GL_TEXTURE_2D)


import os
from utils import download_file
import cv2

if __name__ == "__main__":
    # Open file
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

    renderer = Renderer()
    counter = 0
    pose = np.eye(4)
    pose[:3, 3] = np.array([0.01, 0, 0])
    while True:
        cv_stereo_img = stereo_images[counter % len(stereo_images)]
        counter += 1
        cv_img_left = cv_stereo_img[:, : cv_stereo_img.shape[1] // 2, :]
        renderer.update(pose, cv_img_left)
        pose[:3, 3] += np.array([0.01, 0.01, 0])
