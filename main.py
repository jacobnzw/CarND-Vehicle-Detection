import numpy as np
import sklearn.utils
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import glob
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib
from collections import deque
from scipy.ndimage.measurements import label
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
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
    HEATMAP_BUFFER_LEN = 10  # combine heat-maps from HEATMAP_BUFFER_LEN past frames
    HEATMAP_THRESHOLD = 8
    # ROI_SPECS = (
    #     ((0, 380), (1280, 650), (128, 128), (0.9, 0.25)),
    #     ((0, 380), (1280, 522), (96, 96), (0.9, 0.25)),
    #     ((0, 380), (1280, 458), (64, 64), (0.9, 0.25)),
    # )
    ROI_SPECS = (
        ((200, 400), (1280, 656), (128, 128), (0.75, 0.75)),
        ((200, 400), (1280, 556), (64, 64), (0.75, 0.75)),
    )

    def __init__(self):
        self.boxes = []  # list of bounding boxes pre-computed
        self.classifier = LinearSVC()  # handle for storing a classifier
        self.hm_buffer = deque(maxlen=self.HEATMAP_BUFFER_LEN)  # list of heat maps from several previous frames
        # self.hm_weights = np.ones((self.HEATMAP_BUFFER_LEN, )) / self.HEATMAP_BUFFER_LEN
        # decay for weighted averaging of past heat-maps
        self.hm_weights = np.array([(1-2/(self.HEATMAP_BUFFER_LEN))**i for i in range(self.HEATMAP_BUFFER_LEN)])
        # self.hm_weights /= self.hm_weights.sum()

        # standard feature scaler
        self.scaler = StandardScaler()

        # pre-compute windows
        # list of windows from all depth levels
        self.windows = []
        # self.windows.extend(self._slide_window(y_start_stop=[400, 656], x_start_stop=[None, None],
        #                                        xy_window=(96, 96), xy_overlap=(0.75, 0.75)))
        for rs in self.ROI_SPECS:
            y_range = [rs[0][1], rs[1][1]]
            x_range = [rs[0][0], rs[1][0]]
            self.windows.extend(self._slide_window(y_start_stop=y_range, x_start_stop=x_range,
                                                   xy_window=rs[2], xy_overlap=rs[3]))

        # pre-allocate blank image for speed
        self.img_blank = np.zeros(self.IMG_SHAPE[:2], dtype=np.float32)

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

    def _find_cars(self, img, ystart, ystop, scale):
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        spatial_size = (32, 32)
        hist_bins = 32

        on_windows = []
        win_confid = []
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above, hold the number of hog cells
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step: there are 8 cells, and move 2 cells per step, 75% overlap
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self._get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = self._get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = self._get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
                # Get color features
                spatial_features = self._get_raw_pixel_features(subimg, size=spatial_size)
                hist_features = self._get_color_histogram_features(subimg, nbins=hist_bins)

                # put together a feature vector, normalize and predict
                X = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                test_features = self.scaler.transform(X)
                test_prediction = self.classifier.predict(test_features)

                if test_prediction == 1:  # does the window contain a car?
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    on_windows.append(((xbox_left, ytop_draw + ystart),
                                       (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                    win_confid.append(self.classifier.decision_function(test_features))

        return on_windows, win_confid

    def _extract_features(self, img_rgb, hog=True, color_hist=True, raw_pix=True):
        features = []
        img_con = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        if raw_pix:
            # raw pixel values
            features.extend(self._get_raw_pixel_features(img_con, size=(32, 32)))
        if color_hist:
            # color histograms
            features.extend(self._get_color_histogram_features(img_con))
        if hog:
            # histogram of oriented gradients
            features.extend(self._get_hog_features(img_con[..., 0], 9, 8, 2))
            features.extend(self._get_hog_features(img_con[..., 1], 9, 8, 2))
            features.extend(self._get_hog_features(img_con[..., 2], 9, 8, 2))
        return features

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
                     transform_sqrt=True, visualise=vis, feature_vector=feature_vec)

        # name returns explicitly
        if vis:
            return result[0], result[1]
        else:
            return result

    def _get_raw_pixel_features(self, img, size=(32, 32)):
        """
        Feature vector of raw pixel values in given color space and resizing.

        Parameters
        ----------
        img :
            image
        size : (int, int)
            Size of the resized image

        Returns
        -------

        """
        return np.concatenate([cv2.resize(img[..., i], size).ravel() for i in range(img.shape[2])])

    def _get_color_histogram_features(self, img, nbins=32):
        return np.concatenate([np.histogram(img[..., i], bins=nbins)[0] for i in range(img.shape[2])])

    def _draw_boxes(self, img_bgr, bboxes, color=(0, 0, 255), thick=2):
        # Make a copy of the image
        imcopy = np.copy(img_bgr)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def _draw_heatmap(self, img_out, heat_map, scale=0.2):
        hm_dim = [int(self.IMG_SHAPE[0] * scale), int(self.IMG_SHAPE[1] * scale)]
        # resize heat-map to fit into frame corner
        heat_map = cv2.resize(heat_map, (hm_dim[1], hm_dim[0]))
        # run heat-map through colormap to make it RGB
        colormap = plt.get_cmap('hot')
        heat_map_rgb = colormap(heat_map/heat_map.max())[..., :3]
        img_out[20:20 + hm_dim[0], 20:20 + hm_dim[1]] = heat_map_rgb * 255
        return img_out

    def _reduce_false_positives(self, car_boxes, box_confid):
        hm = self.img_blank.copy()
        for box in car_boxes:
            # add heat to all pixels inside each bbox
            hm[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap = hm.copy()

        self.hm_buffer.append(hm)
        if len(self.hm_buffer) == self.HEATMAP_BUFFER_LEN:
            # integrate heat-maps from past frames when the buffer fills up
            # hm = np.uint8(np.average(np.array(self.hm_buffer), axis=0, weights=self.hm_weights))
            hm = np.sum(self.hm_buffer, axis=0)

            # threshold away "cool" detections
            threshold = hm.max() - 5.0*self.HEATMAP_BUFFER_LEN
            hm[hm <= 18] = 0
        # identify connected components
        img_labeled, num_objects = label(hm)

        # compute bbox coordinates of the found objects
        car_boxes = []
        for car_number in range(num_objects):
            nonzero = (img_labeled == car_number+1).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            car_boxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

        return car_boxes, np.sum(self.hm_buffer, axis=0)

    def _process_frame(self, img_rgb):
        # vehicle detection pipeline

        # # get image crops for all depths
        # # extract features for all image crops
        # X_test = []
        # for win in self.windows:
        #     crop = img_bgr[win[0][1]:win[1][1], win[0][0]:win[1][0]]
        #     crop = cv2.resize(crop, self.BASE_WIN_SHAPE)
        #     X_test.append(self._extract_features(crop))
        #     # X_test[i, :] = self._extract_features(crop)
        # X_test = np.array(X_test)
        #
        # # feature normalization
        # X_test = self.scaler.transform(X_test)
        #
        # # feed image crops to the classifier
        # y_pred = self.classifier.predict(X_test)
        # # pick out boxes predicted as containing a car
        # car_boxes = [self.windows[i] for i in np.argwhere(y_pred == self.LABEL_CAR)[:, 0]]

        # alternative: using HOG subsampling

        car_boxes = []
        box_confidences = []
        # y_start, y_stop, scale configurations
        configs = [[400, 556, 1.5], [400, 656, 2.0]]
        for config in configs:
            # boxes, c = self._find_cars(img_rgb, config[0], config[1], config[2])
            boxes, c = self._find_cars(img_rgb, *config)
            car_boxes.extend(boxes)
            box_confidences.extend(c)
        # car_boxes.extend(self._find_cars(img_rgb, 400, 656, 2.5))
        # car_boxes = self._find_cars(img_rgb, 400, 528, 1.5)
        # car_boxes.extend(self._find_cars(img_rgb, 400, 556, 2.0))
        # car_boxes.extend(self._find_cars(img_rgb, 450, 656, 2.5))

        # reduce false positives
        car_boxes, heat_map = self._reduce_false_positives(car_boxes, box_confidences)

        # draw bounding boxes around detected cars
        img_out = self._draw_boxes(img_rgb, car_boxes, thick=3)

        # draw heat-map into the frame's corner
        img_out = self._draw_heatmap(img_out, heat_map)

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

    def build_features(self, outfile=None, hog=True, color_hist=True, raw_pix=True):
        # get file names for cars and non-cars
        car_list = glob.glob(os.path.join('vehicles', '*', '*.png'))
        noncar_list = glob.glob(os.path.join('non-vehicles', '*', '*.png'))
        np.random.shuffle(car_list)
        np.random.shuffle(noncar_list)
        # take the same amount of cars and non-cars
        half_len = min(len(car_list), len(noncar_list))
        car_list, noncar_list = car_list[:half_len], noncar_list[:half_len]

        # extract feature vectors from each image and create labels
        print('Feature extraction ...')
        features = []
        labels = np.concatenate((np.ones(len(car_list)), np.zeros(len(noncar_list))))
        for file in car_list:
            img = mpimg.imread(file)
            feat_vec = self._extract_features(img, hog, color_hist, raw_pix)
            features.append(feat_vec)
        for file in noncar_list:
            img = mpimg.imread(file)
            feat_vec = self._extract_features(img, hog, color_hist, raw_pix)
            features.append(feat_vec)

        # shuffle features and labels
        features, labels = sklearn.utils.shuffle(np.array(features), np.array(labels))
        # feature normalization
        self.scaler.fit(features)
        features = self.scaler.transform(features)

        # save extracted features and labels, if outfile provided
        if outfile is not None:
            joblib.dump({'features': features, 'labels': labels, 'scaler': self.scaler}, outfile)
            print('Features saved in {}'.format(outfile))
        return features, labels

    def train_classifier(self, data_file=None, dump_file=None, diag=False):
        if data_file is not None:
            # train using features/labels from data_file
            print('Loading features from {}'.format(data_file))
            data = joblib.load(data_file)
            X, y = data['features'], data['labels']
            self.scaler = data['scaler']
        else:
            # train using the standard data from build_features()
            print('Building features ...')
            X, y = self.build_features()

        print('Fitting classifier ...')
        if diag:  # do we wish to report performance for tunning?
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
            self.classifier.fit(X_train, y_train)
            yp_train = self.classifier.predict(X_train)
            yp_test = self.classifier.predict(X_test)
            print('Test accuracy {:.2f}, recall {:.2f}, precision {:.2f}'.format(
                accuracy_score(y_test, yp_test), recall_score(y_test, yp_test), precision_score(y_test, yp_test)))
            print('Train accuracy {:.2f}, recall {:.2f}, precision {:.2f}'.format(
                accuracy_score(y_train, yp_train),
                recall_score(y_train, yp_train), precision_score(y_train, yp_train)))
        else:
            self.classifier.fit(X, y)

        if dump_file is not None:
            # dump classifier using dump_file name
            joblib.dump(self.classifier, dump_file)
            print('Classifier saved to {}'.format(dump_file))
        else:
            # dump classifier using default name
            joblib.dump(self.classifier, 'clf_default')
            print('Classifier saved to clf_default')

    def set_classifier(self, clf_file, data_file):
        self.classifier = joblib.load(clf_file)
        self.scaler = joblib.load(data_file)['scaler']


if __name__ == '__main__':
    data_file = 'data_hog-all-ch-ycc.pkl'
    clf_file = 'linsvc_hog-all-ch-ycc.pkl'
    vd = VehicleDetector()
    # vd.build_features(data_file)
    # vd.train_classifier(data_file, dump_file=clf_file, diag=True)
    vd.set_classifier(clf_file, data_file)

    # TODO: with classifier save also feature parameters
    # test_files = glob.glob(os.path.join(vd.IMGDIR_TEST, '*.jpg'))
    # fig, ax = plt.subplots(1, len(test_files))
    # for i in range(len(test_files)):
    #     out = vd.process_image(test_files[i])
    #     ax[i].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    # plt.show()

    vd.process_video('project_video.mp4', outfile='project_video_processed.mp4', start_time=41, end_time=None)
    # vd.process_video('test_video.mp4', outfile='test_video_processed.mp4')

    # NOTE: feature ordering really has an effect!
