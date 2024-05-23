from PIL import Image
import numpy as np
import pypangolin as pango
from OpenGL.GL import *
from scipy.spatial.transform import Rotation as R

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

apple_img = Image.open("apple.jpg")
apple_img.load()
apple_img = np.asarray(apple_img, dtype="uint8")

canonical_T_optical = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
optical_T_canonical = np.linalg.inv(canonical_T_optical)

# Pango has a different optical frame when rendering the s_cam: z is facing us,  y up, x right
# trust me, I sanity checked this.... it took so long
canonical_T_pango_optical = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
pango_optical_T_canonical = np.linalg.inv(canonical_T_pango_optical)


class Renderer:
    """Renders the display that one would see in the VR Headset."""

    def __init__(self, title="VR Display", width=672, height=376, save_render=False) -> None:
        self.debug = False
        self.win = pango.CreateWindowAndBind(title, width, height)
        self.width = width
        self.height = height
        self.save_render = save_render
        if self.save_render:
            if not os.path.isdir("renders"):
                os.makedirs("renders")

        self.counter = 0  # keep track of the number of renders

        glEnable(GL_DEPTH_TEST)

        self.fx = 420
        self.fy = 420
        self.cx = width / 2
        self.cy = height / 2
        self.pm = pango.ProjectionMatrix(
            width, height, self.fx, self.fy, self.cx, self.cy, 0.1, 1000
        )  # width, height, fx, fy, cx, cy, near clip, far clip
        self.mv = pango.ModelViewLookAt(
            0.0, 0.0, 0.0, 1, 0, 0, pango.AxisZ
        )  # Using Canonical Frame (Z-up)
        # The camera uses a -z forward, y up coordinate system...
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

        self.apple_texture = pango.GlTexture(
            apple_img.shape[1], apple_img.shape[0], GL_RGB, False, 0, GL_RGB, GL_UNSIGNED_BYTE
        )
        self.apple_texture.Upload(apple_img, GL_BGR, GL_UNSIGNED_BYTE)

        self.pose = np.eye(4)  # w_T_k, world to camera pose, given canonical to canonical

    def update(self, pose, image):
        # ------- Update Camera Position and Orientation -------
        self.pose = pose
        if not self.debug:
            self.s_cam.SetModelViewMatrix(
                pango.OpenGlMatrix(np.linalg.inv(self.pose @ canonical_T_pango_optical))
            )

        # world canonical frame to camera optical frame. I am 100% sure about this
        w_T_k = np.array(self.s_cam.GetModelViewMatrix().Inverse().Matrix())
        # canonical to canonical
        w_T_k = w_T_k @ pango_optical_T_canonical

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.d_cam.Activate(self.s_cam)

        # ------- Render Background -------
        self.render_background_video(image)

        # ------- Render 3D Objects -------
        self.render_apple(apple_img.shape[1], apple_img.shape[0], 0.5)

        if self.debug:
            print("pose cam\n", w_T_k)
            print("target pose\n", pose)
            pango.glDrawAxis(1)
            self.display_camera_position()

        pango.FinishFrame()

        if self.save_render:
            self.d_cam.SaveOnRender(f"renders/render{self.counter}.jpg")

        self.counter += 1

    def render_background_video(self, image):
        self.texture.Upload(image[:, :, :3].copy(), GL_BGR, GL_UNSIGNED_BYTE)
        glEnable(GL_TEXTURE_2D)
        self.texture.Bind()

        # Calculate the position and size of the quad relative to the camera
        z = 10.0  # Fixed depth
        half_width = (self.width / 2) * z / self.fx
        half_height = (self.height / 2) * z / self.fy
        cx_offset = (self.cx - self.width / 2) * z / self.fx
        cy_offset = (self.cy - self.height / 2) * z / self.fy
        # Vertices in camera space
        vertices_camera = np.array(
            [
                [-half_width + cx_offset, half_height + cy_offset, z, 1.0],
                [-half_width + cx_offset, -half_height + cy_offset, z, 1.0],
                [half_width + cx_offset, -half_height + cy_offset, z, 1.0],
                [half_width + cx_offset, half_height + cy_offset, z, 1.0],
            ]
        ).T

        vertices_camera_canonical = canonical_T_optical @ vertices_camera

        # Transform vertices to world space
        vertices_world = self.pose @ vertices_camera_canonical
        # print("world vertices\n", vertices_world)

        glBegin(GL_QUADS)
        # Draw the texture on a quad in the background relative to the camera
        glTexCoord2f(0.0, 0.0)
        glVertex3f(vertices_world[0, 0], vertices_world[1, 0], vertices_world[2, 0])
        glTexCoord2f(0.0, 1.0)
        glVertex3f(vertices_world[0, 1], vertices_world[1, 1], vertices_world[2, 1])
        glTexCoord2f(1.0, 1.0)
        glVertex3f(vertices_world[0, 2], vertices_world[1, 2], vertices_world[2, 2])
        glTexCoord2f(1.0, 0.0)
        glVertex3f(vertices_world[0, 3], vertices_world[1, 3], vertices_world[2, 3])
        glEnd()

        glDisable(GL_TEXTURE_2D)

    def render_apple(self, width, height, scale, base_x=4.0, base_y=4.0, base_z=1.0):
        """
        TODO: Figure this out
        """
        # Draw the texture on a large quad in the background
        glEnable(GL_TEXTURE_2D)
        self.apple_texture.Bind()
        glBegin(GL_QUADS)
        # Mapping from 2D to 3D
        z = base_z
        x = width / 2 * z / self.fx * scale
        y = height / 2 * z / self.fx * scale
        if self.debug:
            print(f"x: {base_x - x} - {base_x + x}, y: {base_y - y} - {base_y + y}, z: {z}")
        glTexCoord2f(0.0, 0.0)
        glVertex3f(z, base_x - x, base_y + y)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(z, base_x - x, base_y - y)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(z, base_x + x, base_y - y)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(z, base_x + x, base_y + y)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def display_camera_position(self):
        """ """
        translation = self.pose[:3, 3]
        rotvec = R.from_matrix(self.pose[:3, :3]).as_rotvec(degrees=True)
        angle = np.linalg.norm(rotvec)
        # Apply the translation
        glPushMatrix()
        glTranslatef(translation[0], translation[1], translation[2])
        glRotatef(angle, rotvec[0], rotvec[1], rotvec[2])  # expects axis-angle representation
        pango.glDrawColouredCube()
        glPopMatrix()


import os
from utils import download_file

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
    pose[:3, 3] = np.array([0.0, 1.0, 0])
    pose[:3, :3] = R.from_euler("xyz", [3.14, 0, 0]).as_matrix()

    while True:
        cv_stereo_img = stereo_images[counter % len(stereo_images)]
        counter += 1
        cv_img_left = cv_stereo_img[:, : cv_stereo_img.shape[1] // 2, :]
        renderer.update(pose.copy(), cv_img_left)
        pose[:3, 3] += np.random.rand(3)
        pose[:3, :3] = R.from_euler("xyz", np.random.rand(3)).as_matrix()
