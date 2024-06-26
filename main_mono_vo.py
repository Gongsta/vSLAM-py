import numpy as np
import os
import time
import cv2

from vo import VisualOdometry
from visualization import PangoVisualizer

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def main():
    # --------- Init Camera Camera ---------
    capLeft = cv2.VideoCapture(0)

    with open(os.path.join("KITTI_sequence_1", "calib.txt"), "r") as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=" ")
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]

    running = True
    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    baseline = 0.5

    vo = VisualOdometry(cx, cy, fx, baseline)
    vis = PangoVisualizer()

    while capLeft.isOpened() and running:
        ret, cv_img_left = capLeft.read()
        if ret:
            start = time.time()
            vo.process_frame(cv_img_left)
            vis.update(vo.poses)

            key = cv2.waitKey(1)
            if key == "q":
                running = False
                break
            curr = time.time()
            latency = 1.0 / (curr - start)
            print(f"Running at {latency} hz")
            start = curr
        else:
            running = False

    capLeft.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
