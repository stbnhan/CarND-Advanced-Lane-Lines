import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle

with open('./camera_calibration.p', mode='rb') as f:
	dist_pickle = pickle.load(f)
	mtx = dist_pickle["mtx"]
	dist = dist_pickle["dist"]

def imgshow(fname,img):

	cv2.imshow(fname,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    #defining ROI vertices
    imgsize = img.shape
    vertices = np.array([[(100,				0),
						(imgsize[1]-100, 	0),
						(imgsize[1]-100, 	imgsize[0]),
						(100, 				imgsize[0])]],
						dtype=np.int32)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by \"vertices\" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def color_binary(img, s_thresh=(170, 255), l_thresh=(40, 255), sx_thresh=(20, 100)):

	img = np.copy(img)
	# Convert to HLS color space and separate the V channel
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	h_channel = hls[:,:,0]
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]

	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
	abs_sobelx = np.absolute(sobelx)
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

	# Threshold saturation channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	# Threshold lightness channel
	l_binary = np.zeros_like(l_channel)
	l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

	# Stack each channel
	color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

	return color_binary

def perspective_transform(img, rev=False):

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

	warp_m = cv2.getPerspectiveTransform(src, dst)
	warp_r = cv2.getPerspectiveTransform(dst, src)

	if(rev):
		img_warp = cv2.warpPerspective(img, warp_r, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
	else:
		img_warp = cv2.warpPerspective(img, warp_m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
	#return warp_m, warp_r
	return img_warp

def binary_warp(undistorted):

	img_c = np.copy(undistorted)
	# Convert to HLS color space
	hls = cv2.cvtColor(img_c, cv2.COLOR_BGR2HLS)
	# Convert to HSV color space
	hsv = cv2.cvtColor(img_c, cv2.COLOR_BGR2HSV)
	h_channel = hls[:,:,0]
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	v_channel = hsv[:,:,2]

	s_thresh=(170, 255)
	l_thresh=(40, 255)
	sx_thresh=(20, 100)
	v_thresh=(120,255)

	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
	abs_sobelx = np.absolute(sobelx)
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

	# Threshold saturation channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	# Threshold value channel
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

	img_binary = np.zeros_like(scaled_sobel)
	img_binary[(sxbinary==1)|((s_binary==1)&(v_binary==1))] = 255

	binary_warped = perspective_transform(img_binary)

	return binary_warped

def find_lane_pixels_initial(binary_warped,nwindows,margin,minpix,nonzero,nonzeroy,nonzerox):
	roi = binary_warped[binary_warped.shape[0]//2:,:]
	histogram = np.sum(roi, axis=0)

	midpoint = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Set height of windows - based on nwindows above and image shape
	window_height = np.int(binary_warped.shape[0]//nwindows)
	
	# Current positions to be updated later for each window in nwindows
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y2 = binary_warped.shape[0] - (window+1)*window_height
		win_y1 = binary_warped.shape[0] - window*window_height
		#	11	x------Y=1------x 12	#
		#		|      top     	|		#
		#		|            	|		#
		#		|      bot     	|		#
		#	21	x------Y=2------x 22	#
		win_x11 = leftx_current + margin
		win_x12 = rightx_current + margin
		win_x21 = leftx_current - margin
		win_x22 = rightx_current - margin

		# Identify the nonzero pixels in x and y within the window #
		good_left_inds = ((nonzeroy >= win_y2) & (nonzeroy < win_y1) & 
		(nonzerox >= win_x21) &  (nonzerox < win_x11)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y2) & (nonzeroy < win_y1) & 
		(nonzerox >= win_x22) &  (nonzerox < win_x12)).nonzero()[0]

		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
        
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices (previously was a list of lists of pixels)
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	return left_lane_inds, right_lane_inds

def find_lane_pixels(binary_warped,margin,nonzero,nonzeroy,nonzerox):
	bottom_half = binary_warped[binary_warped.shape[0]//2:,:]
	histogram = np.sum(bottom_half, axis=0)

	midpoint = np.int(histogram.shape[0]//2)

	# within the +/- margin of our polynomial function #
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
					left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
					left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
					right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
					right_fit[1]*nonzeroy + right_fit[2] + margin)))
	return left_lane_inds, right_lane_inds

def fit_polynomial(binary_warped, isFirstImage):

	# Grab left_fit & right_fit
	global left_fit, right_fit

	# HYPERPARAMETERS
	# Number of sliding windows
	nwindows = 9
	# Width of the windows +/- margin
	margin = 100
	# Minimum number of pixels found to recenter window
	minpix = 50

	# Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# If first image, find lane pixels
	if(isFirstImage):

		left_lane_inds, right_lane_inds = find_lane_pixels_initial(binary_warped,nwindows,margin,minpix,nonzero,nonzeroy,nonzerox)
	# If not first image, search near previous polynomial
	else:
		left_lane_inds, right_lane_inds = find_lane_pixels(binary_warped,margin,nonzero,nonzeroy,nonzerox)

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each with np.polyfit() #
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	# Calc both polynomials using ploty, left_fit and right_fit ###
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Calculate vehicle center offset
	midpoint = (left_fitx[-1] + right_fitx[-1])//2
	# veh_pos = img.shape[1]//2
	# xm_per_pix = 3.7/680
	# dx = (veh_pos - midpoint)*xm_per_pix

	## Visualization ##
	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255,0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# Draw detected lanes
	dr_window_img = np.zeros_like(out_img)
	dr_left_line_pts = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
	dr_right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
	dr_line_pts = np.hstack((dr_left_line_pts,dr_right_line_pts))
	cv2.fillPoly(dr_window_img, np.int_([dr_line_pts]), (0,255,0))			# Green: Area between two lanes
	cv2.fillPoly(dr_window_img, np.int_([dr_left_line_pts]), (255,0,0))		# Red (RGB): left lane
	cv2.fillPoly(dr_window_img, np.int_([dr_right_line_pts]), (0,0,255))	# Blue (RGB): right lane
	
	# Reverse Warp
	img_lanes = perspective_transform(dr_window_img,rev=True)

	return result, ploty, midpoint, img_lanes

def measure_curvature_real(ploty):
	'''
	Calculates the curvature of polynomial functions in meters.
	'''
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/680 # meters per pixel in x dimension

	# Grab left_fit & right_fit
	global left_fit, right_fit

	# Define y-value where we want radius of curvature
	# Choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)

	# Calculation of R_curve (radius of curvature)
	left_curverad = ((1 + (2*left_fit[0]*y_eval* + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

	return left_curverad, right_curverad

########################################################
######################### MAIN #########################
########################################################

isFirstImage = True
left_fit = np.array([1.0, 1.0, 1.0])
right_fit = np.array([1.0, 1.0, 1.0])

def video_pipeline(img):
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)
	binary_warped = binary_warp(undistorted)
	img_roi = region_of_interest(binary_warped)

	global isFirstImage
	if(isFirstImage):
		out_img, ploty, midpoint, img_lanes = fit_polynomial(img_roi,isFirstImage=isFirstImage)
		# Calculate the radius of curvature in pixels for both lane lines
		left_curverad, right_curverad = measure_curvature_real(ploty)
		isFirstImage = False
	else:
		out_img, ploty, midpoint, img_lanes = fit_polynomial(img_roi,isFirstImage=isFirstImage)
		# Calculate the radius of curvature in pixels for both lane lines
		left_curverad, right_curverad = measure_curvature_real(ploty)

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img,'Left radius of curvature  = %.2f m'%(left_curverad),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img,'Right radius of curvature = %.2f m'%(right_curverad),(50,80), font, 1,(255,255,255),2,cv2.LINE_AA)

	veh_pos = img.shape[1]//2						# vehicle center
	xm_per_pix = 3.7/680							# meters per pixel in x dimension
	dx = (veh_pos - midpoint)*xm_per_pix			# vehicle center offset from midpoint - center of two detected lanes
	cv2.putText(img,'Vehicle position : %.2f m %s of center'%(abs(dx), 'left' if dx < 0 else 'right'),(50,110), font, 1,(255,255,255),2,cv2.LINE_AA)

	result = cv2.addWeighted(img, 1, img_lanes, 0.3, 0)
	return result

from moviepy.editor import VideoFileClip

video_output = 'P4_video_final.mp4'
clip = VideoFileClip('project_video.mp4')
output_clip = clip.fl_image(video_pipeline)
output_clip.write_videofile(video_output, audio=False)