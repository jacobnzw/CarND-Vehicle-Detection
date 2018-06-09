import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
import os
import glob
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib


class VehicleDetector:
    """
    Vehicle detection pipeline.
    """

    LABEL_CAR = 1
    LABEL_NONCAR = 0
    IMGDIR_TEST = 'test_images'
    IMG_SHAPE = (720, 1280, 3)
    BASE_WIN_SHAPE = (64, 64)

    def __init__(self):
        self.boxes = []  # list of bounding boxes pre-computed
        self.classifier = joblib.load('clf_svm_linear.pkl')  # handle for storing a classifier
        self.heat_maps = []  # list of heat maps from several previous frames

        # standard feature scaler
        self.scaler = StandardScaler()

        # pre-compute windows
        # contains lists of windows, one window list for each depth level
        self.windows = self._slide_window(y_start_stop=[400, 500],
                                          x_start_stop=[200, None], xy_overlap=(0.5, 0.5))

    def _slide_window(self, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        """
        Records corners of a window in each step as it slides across the image (from top left to bottom right).

        Parameters
        ----------
        img : ndarray
        x_start_stop : [int, int] or [None, None]
            Starting and stopping position of a window sliding in x (horizontal) direction.
        y_start_stop : [int, int] or [None, None]
            Starting and stopping position of a window sliding in y (vertical) direction.
        xy_window : (int, int)
            Window width and height
        xy_overlap : (int, int)
            Window overlap in x (horizontal) and y (vertical) directions.

        Notes
        -----
        Taken from Udacity's sliding window implementation. Minor modifications added.

        Returns
        -------
            List of tuples ((startx, starty), (endx, endy)), where (startx, starty) is top left corner and (endx, endy) is
            bottom right window corner.
        """
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = self.IMG_SHAPE[1]
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = self.IMG_SHAPE[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def _prep_data_files(self):
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

    def _extract_features(self, car_filenames, noncar_filenames):
        features = []
        labels = []
        for file in car_filenames:
            img = cv2.imread(file)
            feat_hog = self._get_hog_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 8, 2)
            features.append(feat_hog)
            labels.append(LABEL_CAR)

        for file in noncar_filenames:
            img = cv2.imread(file)
            feat_hog = self._get_hog_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 8, 2)
            features.append(feat_hog)
            labels.append(LABEL_NONCAR)

        # shuffle features and labels
        features, labels = shuffle(np.array(features), np.array(labels))

        # feature normalization
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

        return features, labels

    def _get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
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

    def _get_raw_pixel_features(self, img, color_space='BGR', size=(32, 32)):
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

    def _get_color_histogram_features(self, img_bgr, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the RGB channels separately
        bhist = np.histogram(img_bgr[:, :, 0], bins=nbins, range=bins_range)
        ghist = np.histogram(img_bgr[:, :, 1], bins=nbins, range=bins_range)
        rhist = np.histogram(img_bgr[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        return np.concatenate((bhist[0], ghist[0], rhist[0]))

    def _draw_boxes(self, img_bgr, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img_bgr)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def _process_frame(self, img_bgr):
        # vehicle detection pipeline

        # get image crops for all depths
        # extract features for all image crops
        crop_features = []
        for win in self.windows:
            crop = img_bgr[win[0][1]:win[1][1], win[0][0]:win[1][0]]
            crop_features.append(self._get_hog_features(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 9, 8, 2))
        crop_features = np.array(crop_features)

        # feature normalization
        crop_features = self.scaler.fit_transform(crop_features)

        # feed image crops to the classifier
        y_pred = self.classifier.predict(crop_features)
        # pick out boxes predicted as containing a car
        car_boxes = [self.windows[i] for i in np.argwhere(y_pred == LABEL_CAR).squeeze()]

        # TODO: remove false positives
        # create heat map for the current frame
        # integrate with heat maps form past frames
        # threshold away "cool" detections

        # draw bounding boxes around detected cars
        img_out = self._draw_boxes(img_bgr, car_boxes)

        return img_out

    def process_image(self, infile, outfile=None):
        """
        Process a single image. Saves image if `outfile` is provided.

        Parameters
        ----------
        infile : str
            Input file name

        outfile : str or None
            Output file name

        Returns
        -------

        """

        img = cv2.imread(infile)
        img_out = self._process_frame(img)

        if outfile is not None:
            cv2.imwrite(outfile, img_out)

        return img_out

    def process_video(self, infile, outfile=None, start_time=0, end_time=None):
        """
        Process a video file. Saves output to `outfile` if provided.

        Parameters
        ----------
        infile : str
            Input file name.

        outfile : str or None
            Output file name.

        start_time : int
        end_time : int or None
            Both arguments specify which segment of video file to process. Values are in seconds.

        Returns
        -------

        """
        clip = VideoFileClip(infile).subclip(start_time, end_time)
        out_clip = clip.fl_image(self._process_frame)
        if outfile is not None:
            out_clip.write_videofile(outfile, audio=False)
        return out_clip


LABEL_CAR = 1
LABEL_NONCAR = 0
IMGDIR_TEST = 'test_images'


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

    # feature normalization
    features = StandardScaler().fit_transform(features)

    from sklearn.decomposition import PCA
    dimred = PCA(n_components=features.shape[0])
    dimred.fit(features)
    # print(np.cumsum(dimred.explained_variance_ratio_[:100]))
    num_comp = np.argwhere(np.cumsum(dimred.explained_variance_ratio_) > var_cutoff).min()
    print('{:d} components needed to preserve {:.2f}% of variance'.format(num_comp, var_cutoff*100))
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(dimred.explained_variance_ratio_[:100])
    ax[1].plot(np.cumsum(dimred.explained_variance_ratio_))
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


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Records corners of a window in each step as it slides across the image (from top left to bottom right).

    Parameters
    ----------
    img : ndarray
    x_start_stop : [int, int] or [None, None]
        Starting and stopping position of a window sliding in x (horizontal) direction.
    y_start_stop : [int, int] or [None, None]
        Starting and stopping position of a window sliding in y (vertical) direction.
    xy_window : (int, int)
        Window width and height
    xy_overlap : (int, int)
        Window overlap in x (horizontal) and y (vertical) directions.

    Notes
    -----
    Taken from Udacity's sliding window implementation. Minor modifications added.

    Returns
    -------
        List of tuples ((startx, starty), (endx, endy)), where (startx, starty) is top left corner and (endx, endy) is
        bottom right window corner.
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


if __name__ == '__main__':
    # # TODO: finish preliminary version of the whole pipeline
    # # car_list, noncar_list = prep_data_files()
    # # X, y = extract_features(car_list, noncar_list)
    #
    # # # save extracted features and labels
    # # np.savez('data_2k', X, y, features=X, labels=y)
    #
    # # load features and labels
    # data = np.load('data_2k_hog.npz')
    # X, y = data['features'], data['labels']
    # #
    # from sklearn.model_selection import train_test_split, GridSearchCV
    # from sklearn.svm import LinearSVC, SVC
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    #
    # print('Fitting LinearSVC...')
    # clf = LinearSVC()
    # clf.set_params(C=0.004)
    # clf.fit(X_train, y_train)
    # print('Mean accuracy {}'.format(clf.score(X_test, y_test)))
    #
    # joblib.dump(clf, 'clf_svm_linear.pkl')
    #
    # # print('Fitting LinearSVC grid search...')
    # # from sklearn.metrics import accuracy_score, make_scorer
    # # clf = GridSearchCV(clf, {'C': np.logspace(-3, 3, 10)}, scoring=make_scorer(accuracy_score), n_jobs=3)
    # # clf.fit(X_train, y_train)
    # # print('Mean accuracy {:.2f} (Best params: {:.4e})'.format(clf.score(X_test, y_test), clf.best_params_['C']))
    #
    # print('Processing test image...')
    # test_files = glob.glob(os.path.join(IMGDIR_TEST, '*.jpg'))
    # img = cv2.cvtColor(cv2.imread(test_files[1]), cv2.COLOR_BGR2GRAY)
    # windows = slide_window(img, y_start_stop=[400, 500], x_start_stop=[600, None], xy_overlap=(0.5, 0.5))
    # test_features = []
    # for win in windows:
    #     crop = img[win[0][1]:win[1][1], win[0][0]:win[1][0]]
    #     test_features.append(get_hog_features(crop, 9, 8, 2))
    # test_features = np.array(test_features)
    # # feature scaling
    # scaler = StandardScaler()
    # test_features_scaled = scaler.fit_transform(test_features)
    #
    # y_pred = clf.predict(test_features_scaled)
    # # which windows were predicted as cars
    # car_win = [windows[i] for i in np.argwhere(y_pred == LABEL_CAR).squeeze()]
    # img_boxes = draw_boxes(img, car_win, thick=3)
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(img_boxes, cmap='gray')
    # plt.show()
    #
    # # pca_demo(car_list, noncar_list)


    vd = VehicleDetector()
    # out = vd.process_image(os.path.join(IMGDIR_TEST, 'test1.jpg'))
    # plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    # plt.show()
    vd.process_video('project_video.mp4', outfile='project_video_processed.mp4')
