#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2
import dlib
import argparse

# import my local helper modules
import toolbox
import progressbar

# Handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', nargs=1, type=str,	metavar='input-path',	required=True,			help='Input video path')
parser.add_argument('-o', nargs=1, type=str,	metavar='output-path',	required=True,			help='Output video path')
parser.add_argument('-r', nargs=1, type=float,	metavar='degree',		default=[0],			help='Rotate the video by # degree')
parser.add_argument('-c', nargs=1, type=str,	metavar='1024x768',		default=['1024x768'],	help='Crop the video to a specified resolution')
parser.add_argument('-f', nargs=1, type=float,	metavar='frame-rate',	default=[30],			help='Specify the frame rate of the output video')
parser.add_argument('-k', nargs=1, type=int,	metavar='skip-rate',	default=[1],			help='Specify the skip frame rate of the recognition of the video')
args = parser.parse_args()

# Set constant variables
INPUT_PATH		= args.i[0]
OUTPUT_PATH		= args.o[0]
ROTATION		= args.r[0]
FRAME_RATE		= args.f[0]
OUTPUT_WIDTH	= int(args.c[0].split('x')[0])
OUTPUT_HEIGHT	= int(args.c[0].split('x')[1])
FRAME_SKIP_RATE	= args.k[0]
RECOGNIZE_SCALE	= 0.2
PREDICTOR_PATH	= "../shapes/shape_predictor_68_face_landmarks.dat"

# Create the output directory if it doesn't exist
output_dir = os.path.dirname(os.path.realpath(OUTPUT_PATH))
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Setup OpenCV capture and video using H264 codec for MP4
capture = cv2.VideoCapture(INPUT_PATH, 0)
fourcc = cv2.VideoWriter_fourcc(*'H264')
video = cv2.VideoWriter()
video.open(OUTPUT_PATH, fourcc, FRAME_RATE, (OUTPUT_WIDTH, OUTPUT_HEIGHT), True)

face_cascade = toolbox.loadCascade('haarcascade_frontalface_default.xml')
eye = cv2.imread('../resources/eye.jpg')

# Setup Dlib
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Initialize progressbar
progressbar.init()
total_frames = toolbox.getTotalFrames(INPUT_PATH)
frame_count = 0

# Define variables for frame skipping
skipped_frames = 0
skip_frame_x = None
skip_frame_y = None
skip_rect	 = None
skip_points	 = None
skip_deltas  = None

def rotateFrame(image, angle, crop=True):
	if (angle < 0):
		angle = 360 + angle

	if (angle == 0):
		return image

	if (angle != 90 and angle != 180 and angle != 270):
		raise NameError('You can only rotate the image in steps of 90 / -90 degree')
		return image

	if (angle == 180):
		(h, w) = image.shape[:2]
		center = (w / 2, h / 2)
		matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
		result = cv2.warpAffine(frame, matrix, (w, h))
		return result

	(h, w) = image.shape[:2]

	size = max(w, h)
	canvas = np.zeros((size, size, 3), np.uint8)

	x = int((size - w) / 2)
	y = int((size - h) / 2)

	canvas[y:y+h, x:x+w] = image

	center = (size / 2, size / 2)
	matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
	canvas = cv2.warpAffine(canvas, matrix, (size, size))

	if (crop):
		canvas = canvas[x:x+w, y:y+h]

	return canvas

def cropFrame(frame, x, y, w, h):
	rows, cols = frame.shape[:2]
	if (cols > w and rows > h):
		return frame[y:y+h, x:x+w]
	else:
		return frame

# Go through each frame
while True:
	_, frame = capture.read()
	
	if frame is None:
		exit()

	# Roate the frame
	frame = rotateFrame(frame, ROTATION)

	scale = (1 / RECOGNIZE_SCALE)

	# Check how many frames have been skipped
	if skipped_frames < FRAME_SKIP_RATE:
		skipped_frames += 1

		if skip_frame_x is not None and skip_frame_y is not None:
			frame = cropFrame(frame, skip_frame_x, skip_frame_y, OUTPUT_WIDTH, OUTPUT_HEIGHT)

		if skip_points is not None and skip_deltas is not None:
			toolbox.drawWarpedTriangles(frame, skip_points, skip_deltas, scale)
	else:
		skipped_frames = 0

		# Get current frame with and height for later usage
		(FRAME_HEIGHT, FRAME_WIDTH) = frame.shape[:2]

		# Create a low resolution version of the rotated frame
		small = cv2.resize(frame, (0,0), fx=RECOGNIZE_SCALE, fy=RECOGNIZE_SCALE)
		gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

		# Recognize a face on the low reslution frame
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)
		for (x, y, w, h) in faces:
			# Scale up coordinates
			x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

			# Crop the frame
			frame_x = int((FRAME_WIDTH - OUTPUT_WIDTH) / 2)
			frame_y = y - int((OUTPUT_HEIGHT - h) / 2)
			frame = cropFrame(frame, frame_x, frame_y, OUTPUT_WIDTH, OUTPUT_HEIGHT)

			# Normalize coordinates to the cropped frame
			x = x - frame_x
			y = y - frame_y

			# Create a low resolution version of the cropped frame
			small = cv2.resize(frame, (0,0), fx=RECOGNIZE_SCALE, fy=RECOGNIZE_SCALE)

			# Find all the landmarks on the face
			rs = RECOGNIZE_SCALE
			low_rect = (x * rs, y * rs, w * rs, h * rs)
			shape = predictor(small, toolbox.rect2rectangle(low_rect))
			points = np.array([[p.y, p.x] for p in shape.parts()])

			# Create an array of deltas points
			deltas = np.zeros(points.shape)

			# Left Eye
			deltas[36] = [ 0, -3]
			deltas[37] = [-2, -2]
			deltas[38] = [-2,  0]
			deltas[39] = [ 0,  1]
			deltas[40] = [ 2,  0]
			deltas[41] = [ 2, -2]

			# Right Eye
			deltas[42] = [ 0, -1]
			deltas[43] = [-2,  0]
			deltas[44] = [-2,  2]
			deltas[45] = [ 0,  3]
			deltas[46] = [ 2,  2]
			deltas[47] = [ 2,  0]

			# Mouth
			deltas[48] = [-2, -1]
			deltas[54] = [-2,  1]

			deltas = np.add(points, deltas)

			# Draw Delaunay pattern using the landmarks
			toolbox.drawWarpedTriangles(frame, points, deltas, scale)

			# Save values to use while skipping frames
			skip_frame_x = frame_x
			skip_frame_y = frame_y
			skip_rect	 = (x, y, w, h)
			skip_points	 = points
			skip_deltas  = deltas

	# Write the frame to the output video file
	video.write(frame)

	# Update the progress bar
	frame_count += 1
	progressbar.update(frame_count / total_frames)

# Release everything
capture.release()
video.release()
