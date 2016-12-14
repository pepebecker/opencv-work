#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2
import dlib
import argparse

from scipy.spatial import Delaunay

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
parser.add_argument('-k', nargs=1, type=int,	metavar='skip-rate',	default=[2],			help='Specify the skip frame rate of the recognition of the video')
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
skipped_frames = 0
points = None
tri = None

# Go through each frame
while True:
	_, frame = capture.read()
	
	if frame is None:
		exit()

	# Roate the frame by ROTATION if its not 0
	if (ROTATION != 0):
		rows, cols, _ = frame.shape
		matrix = cv2.getRotationMatrix2D((cols/2, rows/2), -ROTATION, 1)
		frame = cv2.warpAffine(frame, matrix, (cols, rows))

	# Crop the frame
	height, width, _ = frame.shape
	if (width != OUTPUT_WIDTH and height != OUTPUT_HEIGHT):
		offset_x = int((cols - OUTPUT_WIDTH) / 2)
		offset_y = int((rows - OUTPUT_HEIGHT) / 2)
		frame = frame[offset_y:offset_y + OUTPUT_HEIGHT, offset_x:offset_x + OUTPUT_WIDTH]

	# Check how many frames have been skipped
	if skipped_frames < FRAME_SKIP_RATE:
		skipped_frames += 1

		if points and tri:
			toolbox.drawTriangles(frame, points, tri, (1 / RECOGNIZE_SCALE))
	else:
		skipped_frames = 0

		small = cv2.resize(frame, (0,0), fx=RECOGNIZE_SCALE, fy=RECOGNIZE_SCALE)

		# Recognize Face
		gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)
		for rect in faces:
			shape = predictor(small, toolbox.rect2rectangle(rect))
			
			points = np.array([[p.y, p.x] for p in shape.parts()])

			left_eye_points = np.array([p for i, p in enumerate(points) if i > 35 and i < 42])
			right_eye_points = np.array([p for i, p in enumerate(points) if i > 41 and i < 48])

			points = right_eye_points

			tri = Delaunay(points)
			toolbox.drawTriangles(frame, points, tri, (1 / RECOGNIZE_SCALE))

	# Write the frame to the output video file
	video.write(frame)

	# Update the progress bar
	frame_count += 1
	progressbar.update(frame_count / total_frames)

# Release everything
capture.release()
video.release()
