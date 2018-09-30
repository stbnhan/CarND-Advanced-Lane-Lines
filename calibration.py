import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

# define number of inside corners
nx = 9 # inside corners in x
ny = 6 # inside corners in y

# prepare object points
objpoints = []
imgpoints = []
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Make a list of calibration images
fnames = glob.glob('./camera_cal/calibration*.jpg')

print("Obtaining image points and object points...")
for i, fname in enumerate(fnames):
	img = cv2.imread(fname)
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	# If found, append corners
	if ret == True:
		# Draw and display the corners
		#cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
		objpoints.append(objp)
		imgpoints.append(corners)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Correcting Distortion...")
for i, fname in enumerate(fnames):
	print("Calibrating Image: ", fname)
	img = cv2.imread(fname)
	# Undistort image
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)
	# Write image
	fname_out = fname[:-4] + "_calibrated.jpg"
	print("Calibrated to: ", fname_out)
	cv2.imwrite(fname_out,undistorted)

import pickle
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('camera_calibration.p', 'wb'))
