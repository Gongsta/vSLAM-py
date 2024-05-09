from matplotlib import pyplot as plt
from utils import rotation_matrix_to_euler
import numpy as np

class Visualizer3D:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 5))

        self.ax1 = self.fig.add_subplot(121, projection="3d")  # Change 111 to 121
        self.ax1.set_xlim([-2, 2])
        self.ax1.set_ylim([-2, 2])
        self.ax1.set_zlim([-2, 2])
        self.points = self.ax1.plot([], [], [])[0]

        # Plot Cube
        self.ax2 = self.fig.add_subplot(122, projection="3d")  # Add this line for the 2D plot
        self.cube_plot, = self.ax2.plot([], [], [], 'go')  # Dummy plot for updating positions
        self.cube_vertices = np.array([[-1, -1, -1],
                                [1, -1, -1],
                                [1, 1, -1],
                                [-1, 1, -1],
                                [-1, -1, 1],
                                [1, -1, 1],
                                [1, 1, 1],
                                [-1, 1, 1]])

        # Define cube edges for 3D visualization
        self.cube_edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


        self.fig.canvas.draw()

    def draw_cube(self, ax, vertices):
        ax.cla()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        for edge in self.cube_edges:
            ax.plot(*vertices[edge].T, color='b')

    def update(self, positions, orientations):
        x_t, y_t, z_t = zip(*positions)
        self.points.set_data(x_t, y_t)
        self.points.set_3d_properties(z_t)

        new_vertices = self.cube_vertices.copy()
        for R in orientations:
            new_vertices = [R @ vertex for vertex in new_vertices]
        new_vertices = np.array(new_vertices)
        self.draw_cube(self.ax2, new_vertices)

        # redraw just the points
        self.fig.canvas.draw()


if __name__ == "__main__":
    vis = Visualizer3D()
    positions = []
    orientations = []
    for i in range(100):
        pos = np.random.rand(3)
        positions.append(pos)
        orientation = np.array([[0.9998477, -0.0000000,  0.0174524], [0.0003046,  0.9998477, -0.0174497,
], [-0.0174497,  0.0174524,  0.9996954 ]])
        orientations.append(orientation)
        print(pos)
        vis.update(positions, orientations)
        plt.pause(0.001)
