import os
import shutil
import sys
from io import BytesIO
from PIL import Image


with os.fdopen(sys.stdin.fileno(), 'rb') as input_file, os.fdopen(sys.stdout.fileno(), 'wb') as output_file:
	byte = input_file.read(1)
	while byte != "":
		byte = input_file.read(1)
		output_file.write(byte)