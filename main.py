import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
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


LABEL_CAR = 1
LABEL_NONCAR = 0


def prep_data_files():
    # make lists of file names for cars and non-cars
    car_left = glob.glob(os.path.join('vehicles', 'GTI_Left', '*.png'))
    car_right = glob.glob(os.path.join('vehicles', 'GTI_Right', '*.png'))
    car_close = glob.glob(os.path.join('vehicles', 'GTI_MiddleClose', '*.png'))
    car_far = glob.glob(os.path.join('vehicles', 'GTI_Far', '*.png'))
    car_kitti = glob.glob(os.path.join('vehicles', 'KITTI_extracted', '*.png'))
    # randomly choose equal number of images from left, right, middle-close and far directories
    num_carsmp = 400
    np.random.shuffle(car_left)
    np.random.shuffle(car_right)
    np.random.shuffle(car_close)
    np.random.shuffle(car_far)
    np.random.shuffle(car_kitti)
    # combine into one car list
    car_list = car_left[:num_carsmp] + car_right[:num_carsmp] + car_close[:num_carsmp] + \
               car_far[:num_carsmp] + car_kitti[:num_carsmp]

    # list of noncars same length as list of cars
    noncar_files = glob.glob(os.path.join('non-vehicles', 'GTI', '*.png'))
    np.random.shuffle(noncar_files)
    noncar_list = noncar_files[:len(car_list)]

    return car_list, noncar_list


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Histogram of oriented gradients.

    Parameters
    ----------
    img : numpy.array
    orient : int
        Number of orientation bins.
    pix_per_cell : int
        Size of a cell in pixels.
    cell_per_block : int
        Number of cells in each block.
    vis : bool
        Also return image of the HOG.
    feature_vec : bool
        Return data as a feature vector.

    Returns
    -------
    HOG features.

    """

    result = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                 cells_per_block=(cell_per_block, cell_per_block),
                 block_norm='L2-Hys', transform_sqrt=False,
                 visualise=vis, feature_vector=feature_vec)

    # name returns explicitly
    if vis:
        return result[0], result[1]
    else:
        return result


def get_raw_pixel_features(img, color_space='BGR', size=(32, 32)):
    """
    Feature vector of raw pixel values in given color space and resizing.

    Parameters
    ----------
    img :
        BGR image
    color_space : str {'BGR', 'HSV', 'HLS', 'LUV', 'YUV', 'YCrCb'}
        Which color space convert the image to
    size : (int, int)
        Size of the resized image

    Returns
    -------

    """

    # Convert image to new color space (if specified)
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(img)

    features = cv2.resize(feature_image, size).ravel()
    return features


# TODO: more features: color histograms
def pca_demo(car_filenames, noncar_filenames, var_cutoff=0.95):
    """
    Attempt dimensionality reduction of raw pixel data.

    Parameters
    ----------
    img
    var_cutoff

    Returns
    -------

    """
    features = []
    labels = []
    for file in car_filenames:
        img = cv2.imread(file)
        feat = get_raw_pixel_features(img, color_space='BGR')
        features.append(feat)
        labels.append(LABEL_CAR)
    for file in noncar_filenames:
        img = cv2.imread(file)
        feat = get_raw_pixel_features(img, color_space='BGR')
        features.append(feat)
        labels.append(LABEL_NONCAR)

    # shuffle features and labels
    features, labels = shuffle(np.array(features), np.array(labels))
    features, labels = features[:len(features)//2, :], labels[:len(labels)//2]

    from sklearn.decomposition import PCA
    dimred = PCA()
    dimred.fit(features)
    # print(np.cumsum(dimred.explained_variance_ratio_[:100]))
    num_comp = np.argwhere(np.cumsum(dimred.explained_variance_ratio_) > var_cutoff).min()
    print('{:d} components needed to preserve {:.2f}% of variance'.format(num_comp, var_cutoff*100))
    # plt.plot(dimred.explained_variance_)
    plt.plot(dimred.explained_variance_ratio_[:100])
    plt.show()



def extract_features(car_filenames, noncar_filenames):
    features = []
    labels = []
    for file in car_filenames:
        img = cv2.imread(file)
        feat_hog = get_hog_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 8, 2)
        features.append(feat_hog)
        labels.append(LABEL_CAR)

    for file in noncar_filenames:
        img = cv2.imread(file)
        feat_hog = get_hog_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 8, 2)
        features.append(feat_hog)
        labels.append(LABEL_NONCAR)

    # shuffle features and labels
    features, labels = shuffle(np.array(features), np.array(labels))

    # feature normalization
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    return features, labels


if __name__ == '__main__':
    car_list, noncar_list = prep_data_files()
    # X, y = extract_features(car_list, noncar_list)

    # # save extracted features and labels
    # np.savez('data_2k', X, y, features=X, labels=y)

    # # load features and labels
    # data = np.load('data_2k_hog.npz')
    # X, y = data['features'], data['labels']
    #
    # from sklearn.model_selection import train_test_split, GridSearchCV
    # from sklearn.svm import LinearSVC, SVC
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    #
    # print('Fitting LinearSVC...')
    # clf = LinearSVC()
    # clf.fit(X_train, y_train)
    # print('Mean accuracy {}'.format(clf.score(X_test, y_test)))
    #
    # # TODO: finish preliminary version of the whole pipeline
    #
    # print('Fitting LinearSVC grid search...')
    # from sklearn.metrics import accuracy_score, make_scorer
    # clf = GridSearchCV(clf, {'C': np.logspace(-3, 3, 10)}, scoring=make_scorer(accuracy_score), n_jobs=3)
    # clf.fit(X_train, y_train)
    # print('Mean accuracy {:.2f} (Best params: {:.4e})'.format(clf.score(X_test, y_test), clf.best_params_['C']))

    pca_demo(car_list, noncar_list)
