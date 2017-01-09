#!/usr/bin/env python

import os
import io
import sys
import numpy as np
import cv2
import dlib
import time
import argparse

from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image

from http.server import BaseHTTPRequestHandler, HTTPServer

# import my local helper modules
import toolbox

from shader import ShaderProgram
from polygon import Polygon
from quad import Quad
from triangle import Triangle

def createCustomHandlerClass(_args):
    class CustomHandler(BaseHTTPRequestHandler, object):
        def __init__(self, *args, **kwargs):
            self.boundary = '--boundarydonotcross'
            self.html = open('index.html', 'r').read()
            
            # Handle Arguments
            self.input_path      = _args.i[0]
            self.output_path     = _args.o[0]
            self.rotation        = _args.r[0]
            self.output_width    = int(_args.c[0].split('x')[0])
            self.output_height   = int(_args.c[0].split('x')[1])
            self.frame_rate      = _args.f[0]
            self.frame_skip_rate = _args.k[0]
            self.recognize_scale = 0.2
            self.predictor = dlib.shape_predictor('../shapes/shape_predictor_68_face_landmarks.dat')
            self.face_cascade = toolbox.loadCascade('haarcascade_frontalface_default.xml')

            # Define skipping vars
            self.skip_frames    = None
            self.skip_frame_x   = None
            self.skip_frame_y   = None
            self.skip_rect      = None
            self.skip_points    = None

            # OpenGL
            self.quad = Quad()
            self.quad_program = ShaderProgram(fragment=self.quad.fragment, vertex=self.quad.vertex)
            self.quad.loadVBOs(self.quad_program)
            self.quad.loadElements()

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

                # capture = cv2.VideoCapture(self.input_path, 0)
                with open(self.input_path, 'rb') as f:
                    fbuf = io.BufferedReader(f)
                    print(fbuf.read(20))

                exit()

                while True:
                    try:
                        success, frame = capture.read()
                        if success and frame is not None:
                            frame, points = self.processFrame(frame, self.rotation, self.frame_skip_rate, self.face_cascade, self.predictor, self.recognize_scale, self.output_width, self.output_height)
                            if frame is not None:
                                # if points is not None:
                                    # frame = self.getGLFrame(frame, points, self.output_width, self.output_height)

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
                            else:
                               break
                        else:
                            break
                    except KeyboardInterrupt:
                        break

                capture.release()
            else:
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(bytes(self.html.encode('utf-8')))

        def log_message(self, format, *args):
            return

        def rotateFrame(self, image, angle, crop=True):
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

        def cropFrame(self, frame, x, y, w, h):
            rows, cols = frame.shape[:2]
            if (cols > w and rows > h):
                return frame[y:y+h, x:x+w]
            else:
                return frame

        def getGLFrame(self, frame, points, output_width, output_height):
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

            self.quad_program.start()
            toolbox.bind(self.quad)

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, frame)

            glDrawElements(GL_TRIANGLES, len(self.quad.elements) * 3, GL_UNSIGNED_INT, None)

            toolbox.unbind()
            self.quad_program.stop()

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

        def processFrame(self, frame, rotation, frame_skip_rate, face_cascade, predictor, recognize_scale, output_width, output_height):
            points = None

            # Roate the frame
            frame = self.rotateFrame(frame, rotation)

            scale = (1 / recognize_scale)

            # Check how many frames have been skipped
            if self.skip_frames is not None and self.skip_frames > 0:
                self.skip_frames -= 1

                if self.skip_frame_x is not None and self.skip_frame_y is not None:
                    frame = self.cropFrame(frame, self.skip_frame_x, self.skip_frame_y, output_width, output_height)

                points = self.skip_points
            else:
                self.skip_frames = frame_skip_rate

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
                    frame = self.cropFrame(frame, frame_x, frame_y, output_width, output_height)

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

                    toolbox.drawTriangles(frame, points)

                    # Save values to use while skipping frames
                    self.skip_frame_x   = frame_x
                    self.skip_frame_y   = frame_y
                    self.skip_rect      = (x, y, w, h)
                    self.skip_points    = points

            return (frame, points)

    return CustomHandler

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs=1, type=str,    metavar='input-path',   required=True,          help='Input video path')
    parser.add_argument('-o', nargs=1, type=str,    metavar='output-path',  required=True,          help='Output video path')
    parser.add_argument('-r', nargs=1, type=float,  metavar='degree',       default=[0],            help='Rotate the video by # degree')
    parser.add_argument('-c', nargs=1, type=str,    metavar='1024x768',     default=['1024x768'],   help='Crop the video to a specified resolution')
    parser.add_argument('-f', nargs=1, type=float,  metavar='frame-rate',   default=[30],           help='Specify the frame rate of the output video')
    parser.add_argument('-k', nargs=1, type=int,    metavar='skip-rate',    default=[1],            help='Specify the skip frame rate of the recognition of the video')
    return parser.parse_args()

def main():
    with os.fdopen(sys.stdin.fileno(), 'rb') as input_file:
        byte = input_file.read(1)
        while byte != "":
            print('Got a byte')
        # byte = input_file.read(1)

    # input_file = os.open('../videos/stream', os.O_RDWR|os.O_CREAT)
    # with os.fdopen(input_file, 'rb') as stream:
    
    #     fbuf = io.BufferedReader(input_file)
    #     print(fbuf.read(20))


    # with open('../videos/stream', 'rb', buffering=0) as f:
    #     fbuf = io.BufferedReader(f)
    #     print(fbuf.read(20))

    exit()

    try:
        args = handle_args()
        width  = int(args.c[0].split('x')[0])
        height = int(args.c[0].split('x')[1])

        # Init OpenGL
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutCreateWindow("Virtual Window")

        print('Vendor:          ' + glGetString(GL_VENDOR).decode('utf-8'))
        print('Renderer:        ' + glGetString(GL_RENDERER).decode('utf-8'))
        print('OpenGL Version:  ' + glGetString(GL_VERSION).decode('utf-8'))
        print('Shader Version:  ' + glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8'))

        if not glUseProgram:
            print('Missing Shader Objects!')
            sys.exit(1)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Create Stream
        PORT = 3000
        CustomHandler = createCustomHandlerClass(args)
        httpd = HTTPServer(('', PORT), CustomHandler)
        print('Listening on port %s' % PORT)
        httpd.serve_forever()

    except KeyboardInterrupt:
        exit()

if __name__ == '__main__':
    main()
