import cv2
import dlib
import numpy as np
from skimage import color, draw

def getTotalFrames(path):
	cap = cv2.VideoCapture(path)
	counter = 0
	while True:
		_, frame = cap.read()
		if frame is None:
			break
		counter += 1
	return counter

def rect2rectangle(rect):
	x, y, w, h = rect
	return dlib.rectangle(int(x), int(y), int(x+w), int(y+h))

def loadCascade(file):
	return cv2.CascadeClassifier('../cascades/' + file)

def drawTriangles(image, points, tri, pos_multiplier=1, draw_color=np.array([255, 255, 0])):
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
