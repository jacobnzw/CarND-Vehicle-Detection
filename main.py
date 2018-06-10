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
from collections import deque
from scipy.ndimage.measurements import label
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC


class VehicleDetector:
    """
    Vehicle detection pipeline.
    """

    LABEL_CAR = 1
    LABEL_NONCAR = 0
    IMGDIR_TEST = 'test_images'
    IMG_SHAPE = (720, 1280, 3)
    BASE_WIN_SHAPE = (64, 64)
    HEATMAP_BUFFER_LEN = 5  # combine heat-maps from HEATMAP_BUFFER_LEN past frames
    HEATMAP_THRESHOLD = 3

    def __init__(self):
        self.boxes = []  # list of bounding boxes pre-computed
        self.classifier = None  # joblib.load('clf_svm_rbf.pkl')  # handle for storing a classifier
        self.hm_buffer = deque(maxlen=self.HEATMAP_BUFFER_LEN)  # list of heat maps from several previous frames
        self.hm_weights = np.ones((self.HEATMAP_BUFFER_LEN, )) / self.HEATMAP_BUFFER_LEN
        # decay for weighted averaging of past heat-maps
        # self.hm_weights = np.array([(1-1/self.HEATMAP_BUFFER_LEN)**i for i in range(self.HEATMAP_BUFFER_LEN)])
        # self.hm_weights /= self.hm_weights.sum()

        # standard feature scaler
        self.scaler = StandardScaler()

        # pre-compute windows
        # list of windows from all depth levels
        # TODO: define ROIs for several depths
        self.windows = self._slide_window(y_start_stop=[400, 650],
                                          x_start_stop=[200, None], xy_overlap=(0.5, 0.5))

        # pre-allocate blank image for speed
        self.img_blank = np.zeros(self.IMG_SHAPE[:2])

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
        xy_overlap : (float, float)
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

    def _extract_features(self, img_bgr):
        features = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # histogram of oriented gradients
        features.append(self._get_hog_features(gray, 9, 8, 2))
        # color histograms
        features.append(self._get_color_histogram_features(img_bgr))
        # raw pixel values
        # features.append(self._get_raw_pixel_features(img_bgr, color_space='HLS', size=(32, 32)))

        return np.concatenate(features)

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

    def _draw_boxes(self, img_bgr, bboxes, color=(0, 0, 255), thick=2):
        # Make a copy of the image
        imcopy = np.copy(img_bgr)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def _reduce_false_positives(self, car_boxes):
        hm = self.img_blank.copy()
        for box in car_boxes:
            # add heat to all pixels inside each bbox
            hm[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        self.hm_buffer.append(hm)
        if len(self.hm_buffer) == self.HEATMAP_BUFFER_LEN:
            # integrate heat-maps from past frames when the buffer fills up
            hm = np.uint8(np.average(np.array(self.hm_buffer), axis=0, weights=self.hm_weights))

        # threshold away "cool" detections
        hm[hm <= self.HEATMAP_THRESHOLD] = 0
        # identify connected components
        img_labeled, num_objects = label(hm)

        # compute bbox coordinates of the found objects
        car_boxes = []
        for car_number in range(num_objects):
            nonzero = (img_labeled == car_number+1).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            car_boxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

        return car_boxes

    def _process_frame(self, img_bgr):
        # vehicle detection pipeline

        # get image crops for all depths
        # extract features for all image crops
        X_test = []
        for win in self.windows:
            crop = img_bgr[win[0][1]:win[1][1], win[0][0]:win[1][0]]
            crop = cv2.resize(crop, self.BASE_WIN_SHAPE)
            X_test.append(self._extract_features(crop))
        X_test = np.array(X_test)

        # feature normalization
        X_test = self.scaler.fit_transform(X_test)

        # feed image crops to the classifier
        y_pred = self.classifier.predict(X_test)
        # pick out boxes predicted as containing a car
        car_boxes = [self.windows[i] for i in np.argwhere(y_pred == self.LABEL_CAR).squeeze()]
        # noncar_boxes = [self.windows[i] for i in np.argwhere(y_pred == self.LABEL_NONCAR).squeeze()]

        # reduce false positives
        # car_boxes = self._reduce_false_positives(car_boxes)

        # draw bounding boxes around detected cars
        img_out = self._draw_boxes(img_bgr, car_boxes, thick=5)
        # img_out = self._draw_boxes(img_out, noncar_boxes, color=(0, 255, 0), thick=1)

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

    def build_features(self, outfile=None):
        # TODO: hard negative mining, take image with no cars, create new data from patches predicted as cars
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
                   car_far[:num_carsmp] + car_kitti

        # list of noncars same length as list of cars
        noncar_files = glob.glob(os.path.join('non-vehicles', '*', '*.png'))
        np.random.shuffle(noncar_files)
        noncar_list = noncar_files[:len(car_list)]

        # extract feature vectors from each image and create labels
        features = []
        labels = []
        for file in car_list:
            img = cv2.imread(file)
            feat_hog = self._extract_features(img)
            features.append(feat_hog)
            labels.append(self.LABEL_CAR)
        for file in noncar_list:
            img = cv2.imread(file)
            feat_hog = self._extract_features(img)
            features.append(feat_hog)
            labels.append(self.LABEL_NONCAR)

        # shuffle features and labels
        features, labels = shuffle(np.array(features), np.array(labels))
        # feature normalization
        features = StandardScaler().fit_transform(features, labels)
        # save extracted features and labels, if outfile provided
        if outfile is not None:
            np.savez(outfile, features, labels, features=features, labels=labels)
        return features, labels

    def train_classifier(self, data_file=None, dump_file=None, diag=False):
        if data_file is not None:
            # train using features/labels from data_file
            print('Loading features from {}'.format(data_file))
            data = np.load(data_file)
            X, y = data['features'], data['labels']
        else:
            # train using the standard data from build_features()
            print('Building features ...')
            X, y = self.build_features()

        print('Fitting classifier ...')
        if diag:  # do we wish to report performance for tunning?
            self.classifier = LinearSVC()
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
            self.classifier.fit(X_train, y_train)
            yp_train = self.classifier.predict(X_train)
            yp_test = self.classifier.predict(X_test)
            print('Test accuracy {:.2f}, recall {:.2f}, precision {:.2f}'.format(
                accuracy_score(y_test, yp_test), recall_score(y_test, yp_test), precision_score(y_test, yp_test)))
            print('Train accuracy {:.2f}, recall {:.2f}, precision {:.2f}'.format(
                accuracy_score(y_train, yp_train),
                recall_score(y_train, yp_train), precision_score(y_train, yp_train)))

        else:
            self.classifier = LinearSVC()
            self.classifier.fit(X, y)

        if dump_file is not None:
            # dump classifier using dump_file name
            joblib.dump(self.classifier, dump_file)
            print('Classifier save to {}'.format(dump_file))
        else:
            # dump classifier using default name
            joblib.dump(self.classifier, 'clf_default')
            print('Classifier saved to clf_default.pkl')


# LABEL_CAR = 1
# LABEL_NONCAR = 0
# IMGDIR_TEST = 'test_images'
#
#
# def build_features(outfile=None):
#     # make lists of file names for cars and non-cars
#     car_left = glob.glob(os.path.join('vehicles', 'GTI_Left', '*.png'))
#     car_right = glob.glob(os.path.join('vehicles', 'GTI_Right', '*.png'))
#     car_close = glob.glob(os.path.join('vehicles', 'GTI_MiddleClose', '*.png'))
#     car_far = glob.glob(os.path.join('vehicles', 'GTI_Far', '*.png'))
#     car_kitti = glob.glob(os.path.join('vehicles', 'KITTI_extracted', '*.png'))
#     # randomly choose equal number of images from left, right, middle-close and far directories
#     num_carsmp = 400
#     np.random.shuffle(car_left)
#     np.random.shuffle(car_right)
#     np.random.shuffle(car_close)
#     np.random.shuffle(car_far)
#     np.random.shuffle(car_kitti)
#     # combine into one car list
#     car_list = car_left[:num_carsmp] + car_right[:num_carsmp] + car_close[:num_carsmp] + \
#                car_far[:num_carsmp] + car_kitti
#
#     # list of noncars same length as list of cars
#     noncar_files = glob.glob(os.path.join('non-vehicles', '*', '*.png'))
#     np.random.shuffle(noncar_files)
#     noncar_list = noncar_files[:len(car_list)]
#
#     # extract feature vectors from each image and create labels
#     features = []
#     labels = []
#     for file in car_list:
#         img = cv2.imread(file)
#         feat_hog = extract_features(img)
#         features.append(feat_hog)
#         labels.append(LABEL_CAR)
#     for file in noncar_list:
#         img = cv2.imread(file)
#         feat_hog = extract_features(img)
#         features.append(feat_hog)
#         labels.append(LABEL_NONCAR)
#
#     # shuffle features and labels
#     features, labels = shuffle(np.array(features), np.array(labels))
#     # feature normalization
#     features = StandardScaler().fit_transform(features, labels)
#     # save extracted features and labels, if outfile provided
#     if outfile is not None:
#         np.savez(outfile, features, labels, features=features, labels=labels)
#     return features, labels
#
#
# def extract_features(img_bgr):
#     features = []
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     # histogram of oriented gradients
#     features.append(get_hog_features(gray, 9, 8, 2))
#     # color histograms
#     features.append(get_color_histogram_features(img_bgr))
#     # raw pixel values
#     # features.append(get_raw_pixel_features(img_bgr, color_space='HLS', size=(32, 32)))
#
#     return np.concatenate(features)
#
#
# def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
#     """
#     Histogram of oriented gradients.
#
#     Parameters
#     ----------
#     img : numpy.array
#     orient : int
#         Number of orientation bins.
#     pix_per_cell : int
#         Size of a cell in pixels.
#     cell_per_block : int
#         Number of cells in each block.
#     vis : bool
#         Also return image of the HOG.
#     feature_vec : bool
#         Return data as a feature vector.
#
#     Returns
#     -------
#     HOG features.
#
#     """
#
#     result = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
#                  cells_per_block=(cell_per_block, cell_per_block),
#                  block_norm='L2-Hys', transform_sqrt=True,
#                  visualise=vis, feature_vector=feature_vec)
#     if vis:
#         return result[0], result[1]
#     else:
#         return result
#
#
# def get_color_histogram_features(img_bgr, nbins=32, bins_range=(0, 256)):
#     # Compute the histogram of the RGB channels separately
#     bhist = np.histogram(img_bgr[:, :, 0], bins=nbins, range=bins_range)
#     ghist = np.histogram(img_bgr[:, :, 1], bins=nbins, range=bins_range)
#     rhist = np.histogram(img_bgr[:, :, 2], bins=nbins, range=bins_range)
#     # Concatenate the histograms into a single feature vector
#     return np.concatenate((bhist[0], ghist[0], rhist[0]))
#
#
# def get_raw_pixel_features(img, color_space='BGR', size=(32, 32)):
#     """
#     Feature vector of raw pixel values in given color space and resizing.
#
#     Parameters
#     ----------
#     img :
#         BGR image
#     color_space : str {'BGR', 'HSV', 'HLS', 'LUV', 'YUV', 'YCrCb'}
#         Which color space convert the image to
#     size : (int, int)
#         Size of the resized image
#
#     Returns
#     -------
#
#     """
#
#     # Convert image to new color space (if specified)
#     if color_space != 'BGR':
#         if color_space == 'HSV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         elif color_space == 'LUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
#         elif color_space == 'HLS':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#         elif color_space == 'YUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#         elif color_space == 'YCrCb':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     else:
#         feature_image = np.copy(img)
#
#     features = cv2.resize(feature_image, size).ravel()
#     return features
#
#
# def pca_demo(car_filenames, noncar_filenames, var_cutoff=0.95):
#     """
#     Attempt dimensionality reduction of raw pixel data.
#
#     Parameters
#     ----------
#     img
#     var_cutoff
#
#     Returns
#     -------
#
#     """
#     features = []
#     labels = []
#     for file in car_filenames:
#         img = cv2.imread(file)
#         feat = get_raw_pixel_features(img, color_space='BGR')
#         features.append(feat)
#         labels.append(LABEL_CAR)
#     for file in noncar_filenames:
#         img = cv2.imread(file)
#         feat = get_raw_pixel_features(img, color_space='BGR')
#         features.append(feat)
#         labels.append(LABEL_NONCAR)
#
#     # shuffle features and labels
#     features, labels = shuffle(np.array(features), np.array(labels))
#     features, labels = features[:len(features)//2, :], labels[:len(labels)//2]
#
#     # feature normalization
#     features = StandardScaler().fit_transform(features)
#
#     from sklearn.decomposition import PCA
#     dimred = PCA(n_components=features.shape[0])
#     dimred.fit(features)
#     # print(np.cumsum(dimred.explained_variance_ratio_[:100]))
#     num_comp = np.argwhere(np.cumsum(dimred.explained_variance_ratio_) > var_cutoff).min()
#     print('{:d} components needed to preserve {:.2f}% of variance'.format(num_comp, var_cutoff*100))
#     fig, ax = plt.subplots(2, 1)
#     ax[0].plot(dimred.explained_variance_ratio_[:100])
#     ax[1].plot(np.cumsum(dimred.explained_variance_ratio_))
#     plt.show()
#
#
# def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
#                  xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
#     """
#     Records corners of a window in each step as it slides across the image (from top left to bottom right).
#
#     Parameters
#     ----------
#     img : ndarray
#     x_start_stop : [int, int] or [None, None]
#         Starting and stopping position of a window sliding in x (horizontal) direction.
#     y_start_stop : [int, int] or [None, None]
#         Starting and stopping position of a window sliding in y (vertical) direction.
#     xy_window : (int, int)
#         Window width and height
#     xy_overlap : (float, float)
#         Window overlap in x (horizontal) and y (vertical) directions.
#
#     Notes
#     -----
#     Taken from Udacity's sliding window implementation. Minor modifications added.
#
#     Returns
#     -------
#         List of tuples ((startx, starty), (endx, endy)), where (startx, starty) is top left corner and (endx, endy) is
#         bottom right window corner.
#     """
#     # If x and/or y start/stop positions not defined, set to image size
#     if x_start_stop[0] is None:
#         x_start_stop[0] = 0
#     if x_start_stop[1] is None:
#         x_start_stop[1] = img.shape[1]
#     if y_start_stop[0] is None:
#         y_start_stop[0] = 0
#     if y_start_stop[1] is None:
#         y_start_stop[1] = img.shape[0]
#     # Compute the span of the region to be searched
#     xspan = x_start_stop[1] - x_start_stop[0]
#     yspan = y_start_stop[1] - y_start_stop[0]
#     # Compute the number of pixels per step in x/y
#     nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
#     ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
#     # Compute the number of windows in x/y
#     nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
#     ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
#     nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
#     ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
#     # Initialize a list to append window positions to
#     window_list = []
#     # Loop through finding x and y window positions
#     # Note: you could vectorize this step, but in practice
#     # you'll be considering windows one by one with your
#     # classifier, so looping makes sense
#     for ys in range(ny_windows):
#         for xs in range(nx_windows):
#             # Calculate window position
#             startx = xs*nx_pix_per_step + x_start_stop[0]
#             endx = startx + xy_window[0]
#             starty = ys*ny_pix_per_step + y_start_stop[0]
#             endy = starty + xy_window[1]
#             # Append window position to list
#             window_list.append(((startx, starty), (endx, endy)))
#     # Return the list of windows
#     return window_list
#
#
# def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
#     # Make a copy of the image
#     imcopy = np.copy(img)
#     # Iterate through the bounding boxes
#     for bbox in bboxes:
#         # Draw a rectangle given bbox coordinates
#         cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
#     # Return the image copy with boxes drawn
#     return imcopy


if __name__ == '__main__':
    data_file = 'data_hog-ch.npz'

    # print('Building features ...')
    # X, y = build_features(data_file)

    # # load features and labels
    # data = np.load(data_file)
    # X, y = data['features'], data['labels']
    # #
    # from sklearn.model_selection import train_test_split, GridSearchCV
    # from sklearn.svm import LinearSVC, SVC
    # from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    #
    # clf = LinearSVC()
    # clf.set_params(C=1e-3)
    # print('Fitting {} ...'.format(clf.__class__.__name__))
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # y_pred_train = clf.predict(X_train)
    # print('Test accuracy {:.2f}, recall {:.2f}, precision {:.2f}'.format(
    #     accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred)))
    # print('Train accuracy {:.2f}, recall {:.2f}, precision {:.2f}'.format(
    #     accuracy_score(y_train, y_pred_train),
    #     recall_score(y_train, y_pred_train), precision_score(y_train, y_pred_train)))
    #
    # # print('Fitting {} grid search...'.format(clf.__class__.__name__))
    # # clf = GridSearchCV(clf, {'C': np.logspace(-3, 3, 5)}, scoring=make_scorer(precision_score), n_jobs=4)
    # # clf.fit(X_train, y_train)
    # # print('Mean accuracy {:.2f} (Best params: {}'.format(clf.score(X_test, y_test), clf.best_params_))
    #
    # # # save classifier
    # # clf_filename = 'clf_{:s}_{:d}.pkl'.format(clf.__class__.__name__, X.shape[0])
    # # # joblib.dump(clf, clf_filename)
    # # clf = joblib.load(clf_filename)

    # print('Processing test image...')
    # test_files = glob.glob(os.path.join(IMGDIR_TEST, '*.jpg'))
    # img = cv2.imread(test_files[4])
    # windows = slide_window(img, y_start_stop=[400, 650], x_start_stop=[600, None], xy_overlap=(0.5, 0.5))
    # test_features = []
    # for win in windows:
    #     crop = img[win[0][1]:win[1][1], win[0][0]:win[1][0]]
    #     test_features.append(extract_features(crop))
    # test_features = np.array(test_features)
    #
    # # feature scaling
    # test_features_scaled = StandardScaler().fit_transform(test_features)
    #
    # y_pred = clf.predict(test_features_scaled)
    # # which windows were predicted as cars
    # car_win = [windows[i] for i in np.argwhere(y_pred == LABEL_CAR).squeeze()]
    # img_boxes = draw_boxes(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), car_win, thick=2)
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(img_boxes)
    # plt.show()

    vd = VehicleDetector()
    vd.train_classifier(data_file, diag=True)
    out = vd.process_image(os.path.join(vd.IMGDIR_TEST, 'test1.jpg'))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.show()
    # vd.process_video('project_video.mp4', outfile='project_video_processed.mp4', start_time=20, end_time=35)
