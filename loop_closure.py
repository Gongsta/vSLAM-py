import cv2
import numpy as np
import os
from utils import download_file, extract
from eval.associate import read_file_list

if __name__ == "__main__":
    import dbow

    # --------- Download Dataset ---------
    dataset_name = "rgbd_dataset_freiburg1_desk"
    # dataset_name = "rgbd_dataset_freiburg3_long_office_household" # big dataset
    if not os.path.exists(dataset_name):
        path_to_zip_file = f"{dataset_name}.zip"
        download_file(
            f"https://cvg.cit.tum.de/rgbd/dataset/freiburg{dataset_name[21]}/{dataset_name}.tgz",
            path_to_zip_file,
        )
        extract(path_to_zip_file)

    # --------- Load Dataset Images ---------
    # Taken from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    factor = 5000  # for the 16-bit PNG files

    rgb_path_list = read_file_list(f"{dataset_name}/rgb.txt")
    rgb_images = [
        cv2.imread(f"{dataset_name}/{path}") for path in rgb_path_list.values()
    ]

    n_clusters = 10
    depth = 2

    # --------- Creating the Vocabulary and Database (one-time setup) ---------
    # https://github.com/goktug97/PyDBoW
    images = rgb_images
    print("Creating Vocabulary")
    vocabulary = dbow.Vocabulary(images, n_clusters, depth)

    orb = cv2.ORB_create()
    print("Creating Bag of Binary Words from Images")
    bows = []
    for image in images:
        kps, descs = orb.detectAndCompute(image, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        bows.append(vocabulary.descs_to_bow(descs))

    print("Creating Database with pretrained vocabulary")
    db = dbow.Database(vocabulary)
    db.save("pretrained_tum_db.pickle")
    # vocabulary.save("vocabulary.pickle")

    # --------- Loading the pickled  Database ---------
    print("Loading Database")
    loaded_db = dbow.Database.load("pretrained_tum_db.pickle")
    # loaded_vocabulary = vocabulary.load("vocabulary.pickle")
    # Dynamically add images
    for image in images:
        kps, descs = orb.detectAndCompute(image, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        db.add(descs)

    print("Querying Database")
    for image in images:
        kps, descs = orb.detectAndCompute(image, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        scores = db.query(descs)
        match_bow = db[np.argmax(scores)]
        match_desc = db.descriptors[np.argmax(scores)]
