import cv2

def getTotalFrames(path):
	cap = cv2.VideoCapture(path)
	counter = 0
	while True:
		_, frame = cap.read()
		if frame is None:
			break
		counter += 1
	return counter

def is_number(s):
	try:
		int(s)
		return True
	except ValueError:
		return False