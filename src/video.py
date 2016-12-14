#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2
import dlib
import argparse

from skimage import color, draw
from scipy.spatial import Delaunay

# import my local helper modules
import toolbox
import progressbar

# Handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', nargs=1, type=str, metavar='input-path', required=True, help='Input video path')
parser.add_argument('-o', nargs=1, type=str, metavar='output-path', required=True, help='Output video path')
parser.add_argument('-r', nargs=1, type=float, metavar='degree', default=[0], help='Rotate the video by # degree')
parser.add_argument('-c', nargs=1, type=str, metavar='1024x768', default=['1024x768'], help='Crop the video to a specified resolution')
parser.add_argument('-f', nargs=1, type=float, metavar='frame-rate', default=[30], help='Specify the frame rate of the output video')
args = parser.parse_args()

# Set constant variables
INPUT_PATH		= args.i[0]
OUTPUT_PATH		= args.o[0]
ROTATION		= args.r[0]
FRAME_RATE		= args.f[0]
OUTPUT_WIDTH	= int(args.c[0].split('x')[0])
OUTPUT_HEIGHT	= int(args.c[0].split('x')[1])
FRAME_SKIP_RATE	= 2
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

# Setup Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Initialize progressbar
progressbar.init()

total_frames = toolbox.getTotalFrames(INPUT_PATH)
frame_count = 0
skipped_frames = 0
tri = None

def drawTriangles(image, tri, pos_multiplier=1, draw_color=np.array([255, 255, 0])):
	# rows, cols, _ = image.shape
	# overlay = np.zeros((rows, cols, 3))

	for t in tri.simplices.copy():
		p1 = points[t[0]]
		p2 = points[t[1]]
		p3 = points[t[2]]

		p1x = int(p1[0] * pos_multiplier)
		p1y = int(p1[1] * pos_multiplier)
		p2x = int(p2[0] * pos_multiplier)
		p2y = int(p2[1] * pos_multiplier)
		p3x = int(p3[0] * pos_multiplier)
		p3y = int(p3[1] * pos_multiplier)

		image[draw.line(p1x, p1y, p2x, p2y)] = draw_color
		image[draw.line(p2x, p2y, p3x, p3y)] = draw_color
		image[draw.line(p3x, p3y, p1x, p1y)] = draw_color

	# image_hsv = color.rgb2hsv(image)
	# overlay_hsv = color.rgb2hsv(overlay)

	# image_hsv[..., 0] = overlay_hsv[..., 0]
	# image_hsv[..., 1] = overlay_hsv[..., 1]

	# image = color.hsv2rgb(image_hsv)

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

	# Check how many frames where skipped
	if skipped_frames < FRAME_SKIP_RATE:
		skipped_frames += 1

		if tri:
			drawTriangles(frame, tri, (1 / RECOGNIZE_SCALE))
	else:
		skipped_frames = 0

		small = cv2.resize(frame, (0,0), fx=RECOGNIZE_SCALE, fy=RECOGNIZE_SCALE)

		# Recognize Face
		dets = detector(small, 1)
		for k, d in enumerate(dets):
			shape = predictor(small, d)

			points = np.array([[p.y, p.x] for p in shape.parts()])

			tri = Delaunay(points)
			drawTriangles(frame, tri, (1 / RECOGNIZE_SCALE))

	# Write the frame to the output video file
	video.write(frame)

	# Update the progress bar
	frame_count += 1
	progressbar.update(frame_count / total_frames)

# Release everything
capture.release()
video.release()
