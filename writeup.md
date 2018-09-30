# **Advanced Lane Lines** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./img/calibration1.jpg "Before Calibration"
[image2]: ./img/calibration1_calibrated.jpg "After Calibration"
[image3]: ./img/straight_lines1_undistorted.jpg "Undistorted"
[image4]: ./img/straight_lines2_colorbinary.jpg "Color Binary"
[image5]: ./img/straight_lines2_findingthelines.jpg "Finding the Lines"
[image6]: ./img/rofc.JPG "Radius of Curvature Equation"
[image7]: ./img/straight_lines1_final.jpg "Final Image"
[video1]: ./P4_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained entirely in the python code file called `calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  These distortion coefficients are then saved as pickle file called `camera_calibration.p`.  These distortion coefficients will be pulled from the main pipeline as part of preprocessing.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Before:  
![alt text][image1]
After:  
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #103 through #142 in `pipeline.py`).  The color channels used are combination of S channel of HLS color space and V channel of HSV color space. These channels are then combined with sobel X gradient threshold.  Here's an example of my output for this step.  

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform process is in the function called `perspective_transform()`, which appears in lines 80 through 101 in the file `pipeline.py`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as a flag for using either source (`src`) or destination (`dst`) points to return warped or unwarped image respectively.  I chose the hardcode the source and destination points in the following manner:

```python
	# src : original image plane
	src = np.array([[585. /1280.*img.shape[1], 455./720.*img.shape[0]],
					[705. /1280.*img.shape[1], 455./720.*img.shape[0]],
					[1130./1280.*img.shape[1], 720./720.*img.shape[0]],
					[190. /1280.*img.shape[1], 720./720.*img.shape[0]]], np.float32)
	# dst : projection image plane
	dst = np.array([[320. /1280.*img.shape[1], 0./720.*img.shape[0]],
					[1000./1280.*img.shape[1], 0./720.*img.shape[0]],
					[1000./1280.*img.shape[1], 720./720.*img.shape[0]],
					[320. /1280.*img.shape[1], 720./720.*img.shape[0]]], np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 455      | 320, 0        | 
| 190, 720      | 320, 720      |
| 1130, 720     | 1000, 720     |
| 705, 455      | 1000, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Before identifying lane-line pixels, the function `region_of_interest()` (in lines 18 - 47 of `pipeline.py`) filters out unnecessary pixels on the sides of the image that do not belong to lanes.  Then the job of identifying lane-line pixels begin with finding histogram peaks to identify where the lanes start.  Then with sliding window technique on each lane (left and right), identifying non-zero pixels within each window helps refine the estimate of the overall 2nd order polynomial of the lane.  The initial image identifies its lane-line pixels this way, but result 2nd order polynomial from the initial image is used as a reference for next to reduce work.  Subsequent images' lanes use previously found 2nd order polynomials to find lane-line pixels within a pre-defined margin.  
Initial image processing is done by function `find_lane_pixels_initial()` in lines: 144 - 198  
Subsequent image processing is done by function `find_lane_pixels()` in lines: 200 - 213  
The 2nd order polynomials are found with numpy's function `np.polyfit()`  

The result of this process looks like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane is calculated by function `measure_curvature_real()` in lines 295 - 314 in `pipeline.py` with an equation I learned from Udacity class "Measuring Curvature I":  

![alt text][image6]

The position of the vehicle with respect to center is calculated 363 - 364 in `pipeline.py` by using previously found midpoint of two lanes with histogram and finding the offset between the center of camera and this midpoint of two lanes.  
Most importantly, conversions in x and y from pixels space to meters are considered to display real world values to be displayed.  

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I took the result image of projected lanes, converted into real space (unwarp), then overlayed these lanes on original image.  This process takes in lines 281 - 291 of `pipeline.py`  
Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Handling video pipeline is done by `video_pipeline.py`  
Here's a [link to my video result](./P4_video_final.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part of the project after I had a rough pipeline ready was finding the good combination of filters to find binary image in order to easily detect lanes.  Some filters eliminated the lanes completely which broke the pipeline.  I took advantage of region of interest knowing the vehicle will remain relatively centered.  Even though this ROI filters out after warping which makes it a bit more robust, it may not work as well once vehicle swerves around.  My pipeline will likely to fail when the lanes are shown in different colors or at significantly different lighting.  More research in binary filtering to find the most robust combination of colorspace and gradient threshold will likely to resolve this challenge.  