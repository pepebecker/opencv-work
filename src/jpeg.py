#!/usr/bin/env python

import os
import io
import cv2
import sys
import numpy as np
import argparse

from PIL import Image

FRAME_START	= b'\xff\xd8'
FRAME_END	= b'\xff\xd9'
CHUNK_SIZE	= 10000

def handle_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', nargs=1, type=str, metavar='output-dir', required=True, help='Output directory')
	parser.add_argument('-n', nargs=1, type=str, metavar='file-name', default=['%04d.jpg'], help='Output namepattern/filename')
	return parser.parse_args()

def create_output_dir(output_path):
	if not os.path.exists(output_path):
		print('Create %s directory' % output_path)
		os.makedirs(output_path)

def main():
	args = handle_args()

	output_path		= args.o[0]
	output_pattern	= args.n[0]

	create_output_dir(output_path)

	counter = 0
	with os.fdopen(sys.stdin.fileno(), 'rb') as input_file:
		chunk = input_file.read(CHUNK_SIZE)
		buffer = chunk
		while chunk != b'':
			chunk = input_file.read(CHUNK_SIZE)
			buffer += chunk

			a = buffer.find(FRAME_START)
			b = buffer.find(FRAME_END)

			if a != -1 and b != -1:
				counter += 1
				frame_bytes = buffer[a:b+2]
				buffer = buffer[b+2:]

				frame = Image.open(io.BytesIO(frame_bytes))
				# frame = cv2.imdecode(np.fromstring(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

				output_name = output_pattern
				if '%' in output_name:
					output_name %= counter

				print('Saving frame: %s' % output_name)
				frame.save(os.path.join(output_path, output_name))
				# cv2.imwrite(os.path.join(output_path, output_name), frame)

if __name__ == '__main__':
	main()