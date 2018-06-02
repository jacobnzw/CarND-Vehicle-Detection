import numpy as np
import sklearn as skl
import skimage as skm
import matplotlib.pyplot as plt
import cv2
import os
import glob


class VehicleDetector:
    """
    Vehicle detection pipeline.
    """

    def __init__(self):
        pass


# TODO: data pre-processing, train/validation/test split
def load_data():
    IMGDIR_VEHICLE_GTI_LEFT = os.path.join('vehicles', 'GTI_Left')
    IMGDIR_VEHICLE_GTI_RIGHT = os.path.join('vehicles', 'GTI_Right')
    IMGDIR_VEHICLE_GTI_MIDCLOSE = os.path.join('vehicles', 'GTI_MiddleClose')
    IMGDIR_VEHICLE_GTI_FAR = os.path.join('vehicles', 'GTI_Far')
    IMGDIR_VEHICLE_KITTI = os.path.join('vehicles', 'KITTI_extracted')
    IMGDIR_NONVEHICLE_GTI = os.path.join('non-vehicles', 'GTI')
    pass


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True):
    """
    Histogram of oriented gradients.

    Parameters
    ----------
    img : numpy.array
    orient : int
        Number of orientation bins.
    pix_per_cell : (int, int)
        Size of a cell in pixels.
    cell_per_block : (int, int)
        Number of cells in each block.
    vis : bool
        Also return image of the HOG.
    feature_vec : bool
        Return data as a feature vector.

    Returns
    -------
    HOG features.

    """

    return_list = skm.feature.hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm='L2-Hys', transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)

    # name returns explicitly
    hog_features = return_list[0]
    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features


# TODO: feature extraction, save features to speed up testing
def extract_features(data, labels):
    scaler = skl.preprocessing.StandardScaler()
    scaler.fit(data)

    return None


if __name__ == '__main__':
    pass
