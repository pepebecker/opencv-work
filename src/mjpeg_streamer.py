#!/usr/bin/env python

import cv2
import time

from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer

def createCustomHandlerClass(path):
	class CustomHandler(BaseHTTPRequestHandler, object):
		def __init__(self, *args, **kwargs):
			self.video_path = path
			self.boundary = '--boundarydonotcross'
			self.html = open('index.html', 'r').read()
			super(CustomHandler, self).__init__(*args, **kwargs)

		def do_GET(self):
			self.send_response(200)

			if self.path.endswith('.mjpg'):
				# Response headers (multipart)
				self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
				self.send_header('Connection', 'close')
				self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=%s' % self.boundary)
				self.send_header('Expires', 'Mon, 3 Jan 2000 12:34:56 GMT')
				self.send_header('Pragma', 'no-cache')

				capture = cv2.VideoCapture(self.video_path, 0)

				while True:
					try:
						_, frame = capture.read()
						if frame is None:
							capture.release()
							return

						# OpenCV Processing



						# Send frame to server

						jpg = cv2.imencode('.jpg', frame)[1]

						# Part boundary string
						self.end_headers()
						self.wfile.write(bytes(self.boundary.encode('utf-8')))
						self.end_headers()

						# Part headers
						self.send_header('X-Timestamp', time.time())
						self.send_header('Content-length', str(len(jpg)))
						self.send_header('Content-type', 'image/jpeg')
						self.end_headers()

						# Write Binary
						self.wfile.write(bytes(jpg))

					except KeyboardInterrupt:
						capture.release()
						break
			else:
				self.send_header('Content-type', 'text/html')
				self.end_headers()
				self.wfile.write(bytes(self.html.encode('utf-8')))

		def log_message(self, format, *args):
			return

	return CustomHandler

def main():
	try:
		PORT = 8000
		CustomHandler = createCustomHandlerClass('../videos/Portrait.mp4')
		httpd = HTTPServer(('', PORT), CustomHandler)
		print('Listening on port %s' % PORT)
		httpd.serve_forever()

	except KeyboardInterrupt:
		exit()

if __name__ == '__main__':
	main()
