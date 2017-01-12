#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2
import dlib
import argparse

from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image

# import my local helper modules
import toolbox

from shader import ShaderProgram
from polygon import Polygon
from quad import Quad
from triangle import Triangle

FRAME_START	= b'\xff\xd8'
FRAME_END	= b'\xff\xd9'
CHUNK_SIZE	= 10000

def handle_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', nargs=1, type=float,  metavar='degree',       default=[0],            help='Rotate the frames by # degree')
	parser.add_argument('-c', nargs=1, type=str,    metavar='1024x768',     default=['1024x768'],   help='Crop the frames to a specified resolution')
	return parser.parse_args()

def getGLFrame(frame, points, quad, quad_program, output_width, output_height):
	p1, p2, p3 = points[39], points[42], points[33]
	w, h = output_width, output_height

	p1 = [(p1[1] / w) * 2 - 1, -((p1[0] / h) * 2 - 1)]
	p2 = [(p2[1] / w) * 2 - 1, -((p2[0] / h) * 2 - 1)]
	p3 = [(p3[1] / w) * 2 - 1, -((p3[0] / h) * 2 - 1)]

	triangle = Triangle(p1, p2, p3)
	tri_program = ShaderProgram(fragment=triangle.fragment, vertex=triangle.vertex)
	triangle.loadVBOs(tri_program)
	triangle.loadElements()


	glClear(GL_COLOR_BUFFER_BIT)
	glClearColor (0.0, 0.0, 0.0, 1.0)

	#-----------------------------------#

	quad_program.start()
	toolbox.bind(quad)

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, frame)

	glDrawElements(GL_TRIANGLES, len(quad.elements) * 3, GL_UNSIGNED_INT, None)

	toolbox.unbind()
	quad_program.stop()

	#-----------------------------------#

	tri_program.start()
	toolbox.bind(triangle)

	glDrawElements(GL_TRIANGLES, len(triangle.elements) * 3, GL_UNSIGNED_INT, None)

	toolbox.unbind()
	tri_program.stop()

	#-----------------------------------#

	glFinish()

	glPixelStorei(GL_PACK_ALIGNMENT, 1)
	buffer = glReadPixels(0, 0, output_width, output_height, GL_BGR, GL_UNSIGNED_BYTE)

	image = Image.frombytes('RGB', (output_width, output_height), buffer)     
	image = image.transpose(Image.FLIP_TOP_BOTTOM)

	frame = np.asarray(image, dtype=np.uint8)

	glutSwapBuffers()

	return frame

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

def processFrame(frame, rotation, face_cascade, predictor, recognize_scale, output_width, output_height):
	points = None

	# Roate the frame
	frame = rotateFrame(frame, rotation)

	scale = (1 / recognize_scale)

	# Get current frame width and height for later usage
	(frame_height, frame_width) = frame.shape[:2]

	# Create a low resolution version of the rotated frame
	small = cv2.resize(frame, (0,0), fx=recognize_scale, fy=recognize_scale)
	gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

	# Recognize a face on the low reslution frame
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)
	for (x, y, w, h) in faces:
		# Scale up coordinates
		x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

		# Crop the frame
		frame_x = int((frame_width - output_width) / 2)
		frame_y = y - int((output_height - h) / 2)
		frame = cropFrame(frame, frame_x, frame_y, output_width, output_height)

		# Normalize coordinates to the cropped frame
		x = x - frame_x
		y = y - frame_y

		# Create a low resolution version of the cropped frame
		small = cv2.resize(frame, (0,0), fx=recognize_scale, fy=recognize_scale)

		# Find all the landmarks on the face
		rs = recognize_scale
		low_rect = (x * rs, y * rs, w * rs, h * rs)
		shape = predictor(small, toolbox.rect2rectangle(low_rect))
		points = np.array([[p.y * scale, p.x * scale] for p in shape.parts()])

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

		deltas = np.add(points, deltas)

		# Draw Delaunay pattern using the landmarks
		toolbox.drawTriangles(frame, points)

	return frame, points

def main():
	args = handle_args()

	rotation        = args.r[0]
	output_width    = int(args.c[0].split('x')[0])
	output_height   = int(args.c[0].split('x')[1])
	recognize_scale = 0.2

	face_cascade = toolbox.loadCascade('haarcascade_frontalface_default.xml')
	predictor = dlib.shape_predictor('../shapes/shape_predictor_68_face_landmarks.dat')

	 # Init OpenGL
	glutInit(sys.argv)
	glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
	glutInitWindowSize(output_width, output_height)
	glutCreateWindow("Virtual Window")

	if not glUseProgram:
		print('Missing Shader Objects!')
		sys.exit(1)

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

	quad = Quad()
	quad_program = ShaderProgram(fragment=quad.fragment, vertex=quad.vertex)
	quad.loadVBOs(quad_program)
	quad.loadElements()

	with os.fdopen(sys.stdin.fileno(), 'rb') as input_file, os.fdopen(sys.stdout.fileno(), 'wb') as output_file:
		chunk = input_file.read(CHUNK_SIZE)
		buffer = chunk
		while chunk != b'':
			chunk = input_file.read(CHUNK_SIZE)
			buffer += chunk

			a = buffer.find(FRAME_START)
			b = buffer.find(FRAME_END)

			if a != -1 and b != -1:
				frame_bytes = buffer[a:b+2]
				buffer = buffer[b+2:]

				frame = cv2.imdecode(np.fromstring(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
				frame, points = processFrame(frame, rotation, face_cascade, predictor, recognize_scale, output_width, output_height)

				if frame is not None:
					if points is not None:
						frame = getGLFrame(frame, points, quad, quad_program, output_width, output_height)

					frame_data = cv2.imencode('.jpg', frame)[1].tostring()
					output_file.write(frame_data)

if __name__ == '__main__':
	main()
