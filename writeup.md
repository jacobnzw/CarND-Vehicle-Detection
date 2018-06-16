# Vehicle Detection

---

**The goals / steps of this project are the following:**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
[pred_test_image]: ./examples/predictions_test_images.jpg
[frames_heatmaps]: ./examples/frames_heatmaps.jpg
[labeled_blobs]: ./examples/labeled_blobs.jpg
[final_boxes]: ./examples/final_boxes.jpg

---

## Histogram of Oriented Gradients (HOG)

The HOG feature extraction is part of a function `_extract_features()` on lines 205-219 of the `main.py`.

```python
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
```

I started by reading in all the `vehicle` and `non-vehicle` images, for which I employed the very handy `glob` library 

```python
car_list = glob.glob(os.path.join('vehicles', '*', '*.png'))
noncar_list = glob.glob(os.path.join('non-vehicles', '*', '*.png'))
```

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


In my exploration of HOG parameters, I eventually saw no need to increase the number of orientations beyond 9 as the trade-off between computational requirements and increase in performance wasn't favorable. I kept the `pix_per_cell=8` and `cell_per_block=2`, because by inspection of the HOG visualization these choices seemed to give the best trade-off between detail of the image description and length of the feature vector.

In addition to HOG features I also included the color histograms and raw pixels values of the training images subsampled to 32x32 pixels. 

I designed the dataset to have equal representation of cars and noncars (balanced classes). I trained a linear SVM using several values of the parameter C (from 0.1 to 100 increasing exponentially). By this I was hoping to affect the false positive prediction rate. This eventually didn't prove as an effective strategy. The linear SVM worked well right out of the box, hence I saw no need to perform cross-validation grid search for optimal values of the hyper-parameter. With `C=1` I was able to obtain 99% training accuracy, precision and recall.


## Sliding Window Search

Using the sliding window technique, I searched the entire width of the image. I used 3 different scales of the search windows and generally limited my search to the bottom half of the image (ignoring the very bottom pixels showing the hood of the car). The following limits in the vertical direction (y-direction) were used with the correspoding window sizes shown next

| Y-start       | Y-stop        | Window size  |
| ------------- |:-------------:| ------------:|
| 400           | 656           | 128 x 128    |
| 400           | 556           |  96 x 96     |
| 400           | 496           |  64 x 64     |

The smaller windows were used for the areas closer to the horizon where the cars are likely to appear smaller. The window overlap in all cases was 75%. The result on test images can be seen in the following image

![][pred_test_image]

There are some false positives in the top-right image, but these would be filtered out later using the heatmap thresholding.

---
## Video Implementation

Here's a [link to my video result](https://youtu.be/0ija68tO0Rw)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. The heatmap was created by adding a prediction confidence to each area of the image demarked by the window classified by the SVM as containing a car.

```python
def _reduce_false_positives(self, car_boxes, box_confid):
    hm = self.img_blank.copy()
    for box, score in zip(car_boxes, box_confid):
        # add heat to all pixels inside each bbox
        hm[box[0][1]:box[1][1], box[0][0]:box[1][0]] += score

    self.hm_buffer.append(hm)
    if len(self.hm_buffer) == self.HEATMAP_BUFFER_LEN:
        # integrate heat-maps from past frames when the buffer fills up
        # hm = np.uint8(np.average(np.array(self.hm_buffer), axis=0, weights=self.hm_weights))
        hm = np.sum(self.hm_buffer, axis=0)

        # threshold away "cool" detections
        hm[hm <= 6] = 0  # works rather well

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
```

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

In my code I keep a buffer of heat-maps from 10 previous frames and combine them using a simple summation. I did experiment with different weighting schemes but ultimately simple summation works just as well as long as the threshold is adjusted accordingly. Speaking of threshold, the value was chosen purely experimentally and is entirely heuristic. During the processing of problematic parts of the `project_video.mp4` I plotted the accumulated heat-map and picked the threshold value that worked best.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are five frames and their corresponding cumulative heatmaps:

![][frames_heatmaps]

We can see the heatmaps consistently identify two patches in the area where the two cars are located, while suppressing the false positives on the left side of the frames.

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from past 10 frames:
![][labeled_blobs]

Here are the resulting bounding boxes drawn onto the last frame in the series:
![][final_boxes]



---

### Discussion

During implementation of this pipeline I had a pernicious bug that pertained to the way the `moviepy`'s method `fl_image()` loads the image. This bug caused me erroneous color channel conversion, which eventually led to extraction of wrong test features, which led to overly many false positives (that made no sense given the high test set accuracy).

My pipeline has weaknesses when encoutering cars viewed from the side. The windows in such cases are the least robust in their placement. The solution would be to include more data of cars viewed from the side during training and focus on principled optimization of the feature vector extraction process. As a note aside, I found that reverting the order of features has no to negligible effect on the final result.

The linear SVM works very well as a classifier, achieving near-perfect 99% accuracy, which why I don't think trying a different classifier is very productive.

The greatest difficutly in this pipeline is probably design of a suitable filter for false positives. The heatmap approach works well, but requires manual tuning of the threshold and number of accumulated frames. I think a good way to improve the bounding box detection would be incorporate some sort of model of the car's dynamics. If the car is in one place in one frame, it is highly unlikely it will disappear completely in the next frame (cars don't move like that :). I suspect this could be solved using Kalman filtering. Perhaps even some very simple smoothing of box positions in temporal domain would work well.

Finally, today's state-of-the-art detectors, such as YOLO and SSD (single shot detector), are based on deep learning which is definitely something to try out in the future.
