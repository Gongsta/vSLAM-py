import numpy as np
import pypangolin as pango
from OpenGL.GL import *
import OpenGL.GLUT as glut
from scipy.spatial.transform import Rotation as R


class PangoVisualizer:
    def __init__(self, title="Trajectory Viewer", width=1280, height=720) -> None:

        self.debug = True
        self.win = pango.CreateWindowAndBind(title, width, height)
        glEnable(GL_DEPTH_TEST)

        self.pm = pango.ProjectionMatrix(
            width, height, 420, 420, width / 2, height / 2, 0.1, 1000
        )  # width, height, fx, fy, cx, cy, near clip, far clip
        # self.mv = pango.ModelViewLookAt(2.0, 2.0, 2.0, 0, 0, 0, pango.AxisZ)
        # self.mv = pango.ModelViewLookAt(0.0, 0.0, 3.5, 0, 0, 0, pango.AxisX)
        self.mv = pango.ModelViewLookAt(1.0, 1.0, 2.0, 1.0, 1.0, 0.0, pango.AxisX)
        # top down
        # self.mv = pango.ModelViewLookAt(0.0, 0.0, 3.0, 0.0, 0.0, 0.0, pango.AxisY)
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

    def _display_grid(self, grid_size, step):
        """Draws a 3D grid on the XY, YZ, and XZ planes."""
        glColor3f(0.5, 0.5, 0.5)  # Set grid color
        indices = np.linspace(0, grid_size, num=int(grid_size / step + 1))
        for i in indices:
            glBegin(GL_LINES)
            # X-Plane
            p1 = np.array([i, 0, 0])
            p2 = np.array([i, grid_size, 0])
            glVertex3d(p1[0], p1[1], p1[2])
            glVertex3d(p2[0], p2[1], p2[2])

            p1 = np.array([0, i, 0])
            p2 = np.array([grid_size, i, 0])
            glVertex3d(p1[0], p1[1], p1[2])
            glVertex3d(p2[0], p2[1], p2[2])

            # Y-Plane
            p1 = np.array([0, 0, i])
            p2 = np.array([0, grid_size, i])
            glVertex3d(p1[0], p1[1], p1[2])
            glVertex3d(p2[0], p2[1], p2[2])

            p1 = np.array([0, i, 0])
            p2 = np.array([0, i, grid_size])
            glVertex3d(p1[0], p1[1], p1[2])
            glVertex3d(p2[0], p2[1], p2[2])

            # Z-Plane
            p1 = np.array([i, 0, 0])
            p2 = np.array([i, 0, grid_size])
            glVertex3d(p1[0], p1[1], p1[2])
            glVertex3d(p2[0], p2[1], p2[2])

            p1 = np.array([0, 0, i])
            p2 = np.array([grid_size, 0, i])
            glVertex3d(p1[0], p1[1], p1[2])
            glVertex3d(p2[0], p2[1], p2[2])

            glEnd()

    def update(self, poses, landmarks=None, gt_poses=None):
        """


        Parameters
        ----------
        positions (list of 3x1 ndarray): Positions expressed as a 3x1 vector, in world frame
        orientations (list of 3x3 ndarray): Orientations expressed as a 3x3 rotation matrix, in world frame

        """
        positions = [T[:3, 3] for T in poses]
        orientations = [T[:3, :3] for T in poses]
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.d_cam.Activate(self.s_cam)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glLineWidth(2)

        # Draw Trajectory
        for i in range(len(positions) - 1):
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_LINES)
            p1 = positions[i]
            p2 = positions[i + 1]
            glVertex3d(p1[0], p1[1], p1[2])
            glVertex3d(p2[0], p2[1], p2[2])
            glEnd()

        self.draw_orientation_axis(poses[-1])

        if self.debug:
            pango.glDrawAxis(1)
            # 2x2m grid with 0.1m step
            self._display_grid(2, 0.1)

        if landmarks is not None:
            glPointSize(5)
            glColor3f(0.0, 1.0, 0.0)
            pango.glDrawPoints(landmarks)

        if gt_poses is not None:
            positions = [T[:3, 3] for T in gt_poses]
            # Draw Trajectory
            for i in range(len(positions) - 1):
                glColor3f(0.0, 1.0, 0.0)
                glBegin(GL_LINES)
                p1 = positions[i]
                p2 = positions[i + 1]
                glVertex3d(p1[0], p1[1], p1[2])
                glVertex3d(p2[0], p2[1], p2[2])
                glEnd()
            # Draw orientaiton axis for latest position
            self.draw_orientation_axis(gt_poses[-1])

        pango.FinishFrame()

    def draw_orientation_axis(self, pose):
        # Draw orientation axis for latest position
        Ow = pose[:3, 3]
        p = np.array([0.1, 0, 0, 1])
        Xw = pose @ p
        p = np.array([0, 0.1, 0, 1])
        Yw = pose @ p
        p = np.array([0, 0, 0.1, 1])
        Zw = pose @ p
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3d(Ow[0], Ow[1], Ow[2])
        glVertex3d(Xw[0], Xw[1], Xw[2])
        glColor3f(0.0, 1.0, 0.0)
        glVertex3d(Ow[0], Ow[1], Ow[2])
        glVertex3d(Yw[0], Yw[1], Yw[2])
        glColor3f(0.0, 0.0, 1.0)
        glVertex3d(Ow[0], Ow[1], Ow[2])
        glVertex3d(Zw[0], Zw[1], Zw[2])
        glEnd()

    def draw_sphere(self, position):
        glPushMatrix()
        glTranslate(position[0], position[1], position[2])  # Translate to the position
        glut.glutSolidSphere(0.05, 20, 20)  # Draw a sphere with radius 0.1
        glPopMatrix()


if __name__ == "__main__":
    # Open file
    f = open("helpers/trajectory.txt", "r")

    viz = PangoVisualizer()
    positions = []
    orientations = []
    for line in f:
        line = line.split(" ")
        t = np.array([float(line[1]), float(line[2]), float(line[3])])
        quat = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
        positions.append(t)
        orientations.append(R.from_quat(quat).as_matrix())
        poses = [np.eye(4) for _ in range(len(positions))]
        for i in range(len(positions)):
            poses[i][:3, 3] = positions[i]
            poses[i][:3, :3] = orientations[i]
        viz.update(poses)

    f.close()
    # Keep drawing
    while True:
        viz.update(positions, orientations)
