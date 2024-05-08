import cv2
import numpy as np

data = np.load("sample_zed/data.npz")

images = data["stereo"]
depth_images = data["depth"]

print(len(images))
for i in range(len(images)):
    print(i)
    cv2.imshow("image", images[i])
    cv2.imshow("depth", depth_images[i])
    q = cv2.waitKey(33)
    if q == ord("q"):
        break
cv2.destroyAllWindows()
