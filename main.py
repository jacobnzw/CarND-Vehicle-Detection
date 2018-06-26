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
from sklearn.svm import LinearSVC
import json

# TODO: call function with library prefixes, e.g: sklearn.metrics.accuracy_score() rather than accuracy_score()
# TODO: add in the code for lane finding
# TODO: trim the unnecessary methods
# TODO: optimization (profile, cv2.HOGDescriptor, cython)


class VehicleDetector:
    """
    Vehicle detection pipeline.
    """

    LABEL_CAR = 1
    LABEL_NONCAR = 0
    IMGDIR_TEST = 'test_images'
    IMG_SHAPE = (720, 1280, 3)
    BASE_WIN_SIZE = (64, 64)
    HEATMAP_BUFFER_LEN = 10  # combine heat-maps from HEATMAP_BUFFER_LEN past frames
    HEATMAP_THRESHOLD = 8

    def __init__(self):
        self.classifier = LinearSVC()

        # heat-map buffer
        self.hm_buffer = deque(maxlen=self.HEATMAP_BUFFER_LEN)
        # self.hm_weights = np.ones((self.HEATMAP_BUFFER_LEN, )) / self.HEATMAP_BUFFER_LEN
        # decay for weighted averaging of past heat-maps
        self.hm_weights = np.array([(1-2/(self.HEATMAP_BUFFER_LEN))**i for i in range(self.HEATMAP_BUFFER_LEN)])
        # self.hm_weights /= self.hm_weights.sum()

        # standard feature scaler
        self.scaler = StandardScaler()

        # y_start, y_stop, xstart, xstop, window_scale configurations
        # window_scale is multiplier of the base
        self.regions_of_interest = [[400, 496, 0, 1280, 1.0],
                                    [400, 556, 0, 1280, 1.5],
                                    [400, 656, 0, 1280, 2.0]]

        # feature specs
        self.feat_specs = {}

        # pre-allocate blank image for speed
        self.img_blank = np.zeros(self.IMG_SHAPE[:2], dtype=np.float32)

    def _find_cars(self, img, ystart, ystop, xstart, xstop, scale):
        # TODO: make these into class members
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        spatial_size = (32, 32)
        hist_bins = 32

        on_windows = []
        win_confid = []
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, xstart:xstop, :]
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

        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = self.BASE_WIN_SIZE[0]
        nblocks_per_window = (window // pix_per_cell) - 1
        # Instead of overlap, define how many cells to step: there are 8 cells, and move 2 cells per step, 75% overlap
        cells_per_step = 2
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
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], self.BASE_WIN_SIZE)
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
                    on_windows.append(((xbox_left + xstart, ytop_draw + ystart),
                                       (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))
                    win_confid.append(self.classifier.decision_function(test_features))

        return on_windows, win_confid

    def _extract_features(self, img_rgb, hog=True, ch=True, sb=True, hog_orient=9, hog_pix_per_cell=8,
                          hog_cell_per_block=2, sb_size=(32, 32), ch_nbins=32):
        features = []
        img_con = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        if sb:
            # spatial binning (raw pixel values of sized down image)
            features.extend(self._get_raw_pixel_features(img_con, size=sb_size))
        if ch:
            # color histograms
            features.extend(self._get_color_histogram_features(img_con, nbins=ch_nbins))
        if hog:
            # histogram of oriented gradients
            features.extend(self._get_hog_features(img_con[..., 0], hog_orient, hog_pix_per_cell, hog_cell_per_block))
            features.extend(self._get_hog_features(img_con[..., 1], hog_orient, hog_pix_per_cell, hog_cell_per_block))
            features.extend(self._get_hog_features(img_con[..., 2], hog_orient, hog_pix_per_cell, hog_cell_per_block))
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
        for box, score in zip(car_boxes, box_confid):
            # add heat to all pixels inside each bbox
            hm[box[0][1]:box[1][1], box[0][0]:box[1][0]] += score
        self.hm_buffer.append(hm)

        # integrate heat-maps from past frames when the buffer fills up
        if len(self.hm_buffer) == self.HEATMAP_BUFFER_LEN:
            # hm = np.uint8(np.average(np.array(self.hm_buffer), axis=0, weights=self.hm_weights))
            hm = np.sum(self.hm_buffer, axis=0)

            # threshold away "cool" detections
            hm[hm <= 6] = 0  # works rather well

        # identify connected components
        img_labeled, num_objects = label(hm)

        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].set_title('Current frame')
        # ax[0, 0].imshow(heatmap)
        # ax[0, 1].set_title('Sum')
        # ax[0, 1].imshow(hm)
        # ax[1, 0].set_title('Weighted Average')
        # ax[1, 0].imshow(np.average(np.array(self.hm_buffer), weights=self.hm_weights, axis=0))
        # ax[1, 1].set_title('Weighted Average')
        # w = [1 / (2 ** i) for i in range(self.HEATMAP_BUFFER_LEN)]
        # ax[1, 1].imshow(np.average(np.array(self.hm_buffer), weights=w, axis=0))

        # compute bbox coordinates of the found objects
        car_boxes = []
        for car_number in range(num_objects):
            nonzero = (img_labeled == car_number+1).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            car_boxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

        return car_boxes, np.sum(self.hm_buffer, axis=0)

    def _process_frame(self, img_rgb):
        car_boxes = []
        box_confidences = []
        for roi in self.regions_of_interest:
            boxes, c = self._find_cars(img_rgb, *roi)
            car_boxes.extend(boxes)
            box_confidences.extend(c)

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

    def video_frames_visualization(self):
        infile = 'project_video.mp4'
        start_time, end_time = 41, None
        clip = VideoFileClip(infile).subclip(start_time, end_time)
        # y_start, y_stop, scale configurations
        configs = [[400, 556, 1.5], [400, 656, 2.0], [400, 496, 1.0]]
        hmaps = []
        fproc = []
        for idx, frame in enumerate(clip.iter_frames()):
            if idx > 5:  # only few frames is sufficient
                break
            car_boxes = []
            box_confidences = []
            for config in configs:
                boxes, c = self._find_cars(frame, *config)
                car_boxes.extend(boxes)
                box_confidences.extend(c)
            # draw bounding boxes around detected cars
            fproc.append(self._draw_boxes(frame, car_boxes, thick=3))
            # reduce false positives
            car_boxes, heat_map = self._reduce_false_positives(car_boxes, box_confidences)
            heat_map = plt.get_cmap('hot')(heat_map/heat_map.max())[..., :3]
            hmaps.append(heat_map)
        fig, ax = plt.subplots(5, 2)
        for i in range(5):
            ax[i, 0].imshow(fproc[i])
            ax[i, 1].imshow(hmaps[i])
        ax[0, 0].set_title('Frames')
        ax[0, 1].set_title('Heatmap')
        plt.subplots_adjust(wspace=0)
        plt.show()

        # cv2.imwrite('frame_{:d}'.format(idx), img_out)
        # cv2.imwrite('heatmap_{:d}'.format(idx), heat_map)

    def build_features(self, feature_specs, outfile=None):
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
            feat_vec = self._extract_features(img, **feature_specs)
            features.append(feat_vec)
        for file in noncar_list:
            img = mpimg.imread(file)
            feat_vec = self._extract_features(img, **feature_specs)
            features.append(feat_vec)

        # shuffle features and labels
        features, labels = sklearn.utils.shuffle(np.array(features), np.array(labels))
        # feature normalization
        self.scaler.fit(features)
        features = self.scaler.transform(features)

        # save extracted features and labels, if outfile provided
        if outfile is not None:
            data_dict = {'features': features, 'labels': labels,
                         'scaler': self.scaler, 'feature_specs': feature_specs}
            joblib.dump(data_dict, outfile)
            print('Features saved in {}'.format(outfile))
        return features, labels

    def train_classifier(self, feature_data_file, dump_file=None, diag=False):
        print('Loading features from {}'.format(feature_data_file))
        data = joblib.load(feature_data_file)
        X, y = data['features'], data['labels']
        self.scaler = data['scaler']
        self.feat_specs = data['feature_specs']

        # print some info about features
        print()
        print('Feature vector length: {:d}'.format(X.shape[1]))
        print('Feature specs: ')
        print(json.dumps(self.feat_specs, indent=4, sort_keys=True))
        print()

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

        # dump classifier, scaler and feature specs
        if dump_file is None:
            dump_file = 'clf_default'
        data_dict = {'classifier': self.classifier, 'scaler': self.scaler, 'feature_specs': feat_specs}
        joblib.dump(data_dict, dump_file)
        print('Classifier saved to {}.'.format(dump_file))

    def set_classifier(self, clf_file):
        dump = joblib.load(clf_file)
        self.classifier = dump['classifier']
        self.scaler = dump['scaler']
        self.feat_specs = dump['feature_specs']


if __name__ == '__main__':
    data_file = 'data_hog-all-ch-ycc.pkl'
    clf_file = 'linsvc_hog-all-ch-ycc.pkl'
    feat_specs = {
        'hog': True,  # histogram of oriented gradients
        'hog_orient': 9,
        'hog_pix_per_cell': 8,
        'hog_cell_per_block': 2,
        'ch': True,  # color histogram
        'ch_nbins': 32,
        'sb': True,  # spatial binning
        'sb_size': (32, 32),
    }
    vd = VehicleDetector()
    # vd.build_features(feat_specs, outfile=data_file)
    # vd.classifier.set_params(C=0.1)
    vd.train_classifier(data_file, dump_file=clf_file, diag=True)
    vd.set_classifier(clf_file)

    # process some test images
    # test_files = glob.glob(os.path.join(vd.IMGDIR_TEST, '*.jpg'))
    # test_files = [test_files[i] for i in [1, 2, 5, 7]]
    # out = []
    # for file in test_files:
    #     out.append(vd.process_image(file))
    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(cv2.cvtColor(out[0], cv2.COLOR_BGR2RGB))
    # ax[0, 1].imshow(cv2.cvtColor(out[1], cv2.COLOR_BGR2RGB))
    # ax[1, 0].imshow(cv2.cvtColor(out[2], cv2.COLOR_BGR2RGB))
    # ax[1, 1].imshow(cv2.cvtColor(out[3], cv2.COLOR_BGR2RGB))
    # plt.show()

    # process video
    vd.process_video('project_video.mp4', outfile='project_video_processed.mp4', start_time=38, end_time=43)
    # vd.process_video('test_video.mp4', outfile='test_video_processed.mp4')

    # some visualizations
    # vd.video_frames_visualization()
