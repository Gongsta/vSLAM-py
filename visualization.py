import numpy as np
import pypangolin as pango
from OpenGL.GL import *
from scipy.spatial.transform import Rotation as R


class PangoVisualizer:
    def __init__(self, width=1280, height=720) -> None:
        self.debug = True
        self.win = pango.CreateWindowAndBind("Trajectory Viewer", width, height)
        glEnable(GL_DEPTH_TEST)

        self.pm = pango.ProjectionMatrix(
            width, height, 420, 420, width / 2, height / 2, 0.1, 1000
        )  # width, height, fx, fy, cx, cy, near clip, far clip
        self.mv = pango.ModelViewLookAt(3.0, 3.0, 3.0, 0, 0, 0, pango.AxisZ)
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

    def update(self, positions, orientations, landmarks=None):
        assert len(positions) == len(orientations)
        # while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.d_cam.Activate(self.s_cam)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glLineWidth(2)

        # Draw Orientation Axes
        for i in range(len(positions) - 1):
            t = positions[i]
            R = orientations[i]
            Ow = t
            pose = np.eye(4)
            pose[:3, :3] = orientations[i]
            pose[:3, 3] = Ow
            T = np.eye(4)
            T[:3, 3] = np.array([0.1, 0, 0])
            Xw = (T @ pose)[:3, 3]
            T[:3, 3] = np.array([0, 0.1, 0])
            Yw = (T @ pose)[:3, 3]
            T[:3, 3] = np.array([0, 0, 0.1])
            Zw = (T @ pose)[:3, 3]
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

        # Draw Trajectory
        for i in range(len(positions) - 1):
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_LINES)
            p1 = positions[i]
            p2 = positions[i + 1]
            glVertex3d(p1[0], p1[1], p1[2])
            glVertex3d(p2[0], p2[1], p2[2])
            glEnd()

        if self.debug:
            pango.glDrawAxis(1)
            self._display_grid(2, 0.1)

        if landmarks is not None:
            glPointSize(5)
            glColor3f(0.0, 1.0, 0.0)
            pango.glDrawPoints(landmarks)

        pango.FinishFrame()


if __name__ == "__main__":
    # Open file
    f = open("trajectory.txt", "r")

    viz = PangoVisualizer()
    positions = []
    orientations = []
    for line in f:
        line = line.split(" ")
        t = np.array([float(line[1]), float(line[2]), float(line[3])])
        quat = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
        positions.append(t)
        orientations.append(R.from_quat(quat).as_matrix())
        viz.draw(positions, orientations)

    f.close()
    # Keep drawing
    while True:
        viz.draw(positions, orientations)
