#!/usr/bin/env python

import os
import io
import sys
import numpy as np
import argparse

from PIL import Image

def handle_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', nargs=1, type=str, metavar='input-dir', required=True, help='Input video path')
	parser.add_argument('-n', nargs=1, type=int, metavar='number', default=[0], help='number of frames to process')
	return parser.parse_args()

def get_files(path):
	for (dirpath, _, filenames) in os.walk(path):
		for filename in filenames:
			yield os.path.join(dirpath, filename)

def main():
	args = handle_args()

	input_path  = args.i[0]
	n_of_frames = args.n[0]

	counter = 0
	list_files = get_files('../frames')

	with os.fdopen(sys.stdout.fileno(), 'wb') as output_file:
		for file_path in list_files:
			if (counter < n_of_frames or n_of_frames == 0):
				frame = Image.open(file_path, mode='r')

				if frame is not None:
					frame_data = io.BytesIO()
					frame.save(frame_data, format='JPEG')
					output_file.write(frame_data.getvalue())

				counter += 1
			else:
				break

if __name__ == '__main__':
	main()