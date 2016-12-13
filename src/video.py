#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2

# import my local helper modules
import toolbox
import progressbar

# Input variables
INPUT_PATH = '../videos/90CC.mp4'
INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080

# Output variables
OUTPUT_PATH = '../output'
OUTPUT_FILE = '90CC.mp4'
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 768
ROTATION = 0

# Handle command line arguments
length = len(sys.argv)
for i in range(0, length):
	if sys.argv[i] == '--rotate':
		if i < length - 1:
			if toolbox.is_number(sys.argv[i+1]):
				ROTATION = float(sys.argv[i+1])
			else:
				print('rotate option requires a parameter of type int!')
				exit()
		else:
			print('rotate option requires a parameter of type int!')
			exit()

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Setup OpenCV capture and video using H264 codec for MP4
capture = cv2.VideoCapture(INPUT_PATH, 0)

screen_size = (OUTPUT_WIDTH, OUTPUT_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'H264')

video = cv2.VideoWriter()
video.open(os.path.join(OUTPUT_PATH, OUTPUT_FILE), fourcc, 20.0, screen_size, True)

# Initialize progressbar
progressbar.init()

total_frames = toolbox.getTotalFrames(INPUT_PATH)
frame_count = 0

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
	offset_x = (INPUT_WIDTH - OUTPUT_WIDTH) / 2
	offset_y = (INPUT_HEIGHT - OUTPUT_HEIGHT) / 2
	frame = frame[offset_y:offset_y + OUTPUT_HEIGHT, offset_x:offset_x + OUTPUT_WIDTH]

	# Write the frame to the output video file
	video.write(frame)

	# Update the progress bar
	frame_count += 1
	progressbar.update(frame_count / total_frames)

# Release everything
capture.release()
video.release()
