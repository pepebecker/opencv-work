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
import progressbar

from shader import ShaderProgram
from polygon import Polygon
from quad import Quad
from triangle import Triangle

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs=1, type=str,    metavar='input-path',   required=True,          help='Input video path')
    parser.add_argument('-o', nargs=1, type=str,    metavar='output-path',  required=True,          help='Output video path')
    parser.add_argument('-r', nargs=1, type=float,  metavar='degree',       default=[0],            help='Rotate the video by # degree')
    parser.add_argument('-c', nargs=1, type=str,    metavar='1024x768',     default=['1024x768'],   help='Crop the video to a specified resolution')
    parser.add_argument('-f', nargs=1, type=float,  metavar='frame-rate',   default=[30],           help='Specify the frame rate of the output video')
    parser.add_argument('-k', nargs=1, type=int,    metavar='skip-rate',    default=[1],            help='Specify the skip frame rate of the recognition of the video')
    return parser.parse_args()

def create_output_dir(output_path):
    output_dir = os.path.dirname(os.path.realpath(output_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def setupOpenCV(input_path, output_path, frame_rate):
    global CAPTURE, VIDEO, FACE_CASCADE
    CAPTURE = cv2.VideoCapture(input_path, 0)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    VIDEO = cv2.VideoWriter()
    VIDEO.open(output_path, fourcc, frame_rate, (OUTPUT_WIDTH, OUTPUT_HEIGHT), True)
    FACE_CASCADE = toolbox.loadCascade('haarcascade_frontalface_default.xml')

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

def save():
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    buffer = glReadPixels(0, 0, OUTPUT_WIDTH, OUTPUT_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE)

    image = Image.frombytes('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT), buffer)     
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    output = np.asarray(image, dtype=np.uint8)

    VIDEO.write(output)

def loop():
    global TOTAL_FRAMES, FRAME_COUNT
    global SKIPPED_FRAMES, SKIP_FRAME_X, SKIP_FRAME_Y, SKIP_RECT, SKIP_POINTS, SKIP_DELTAS

    _, frame = CAPTURE.read()
    
    if frame is None:
        quit()

    # Roate the frame
    frame = rotateFrame(frame, ROTATION)

    scale = (1 / RECOGNIZE_SCALE)

    # Check how many frames have been skipped
    if SKIPPED_FRAMES < FRAME_SKIP_RATE:
        SKIPPED_FRAMES += 1

        if SKIP_FRAME_X is not None and SKIP_FRAME_Y is not None:
            frame = cropFrame(frame, SKIP_FRAME_X, SKIP_FRAME_Y, OUTPUT_WIDTH, OUTPUT_HEIGHT)

        if SKIP_POINTS is not None and SKIP_DELTAS is not None:
            toolbox.drawWarpedTriangles(frame, SKIP_POINTS, SKIP_DELTAS)
    else:
        SKIPPED_FRAMES = 0

        # Get current frame width and height for later usage
        (frame_height, frame_width) = frame.shape[:2]

        # Create a low resolution version of the rotated frame
        small = cv2.resize(frame, (0,0), fx=RECOGNIZE_SCALE, fy=RECOGNIZE_SCALE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Recognize a face on the low reslution frame
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            # Scale up coordinates
            x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

            # Crop the frame
            frame_x = int((frame_width - OUTPUT_WIDTH) / 2)
            frame_y = y - int((OUTPUT_HEIGHT - h) / 2)
            frame = cropFrame(frame, frame_x, frame_y, OUTPUT_WIDTH, OUTPUT_HEIGHT)

            # Normalize coordinates to the cropped frame
            x = x - frame_x
            y = y - frame_y

            # Create a low resolution version of the cropped frame
            small = cv2.resize(frame, (0,0), fx=RECOGNIZE_SCALE, fy=RECOGNIZE_SCALE)

            # Find all the landmarks on the face
            rs = RECOGNIZE_SCALE
            low_rect = (x * rs, y * rs, w * rs, h * rs)
            shape = PREDICTOR(small, toolbox.rect2rectangle(low_rect))
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
            toolbox.drawWarpedTriangles(frame, points, deltas)

            # Save values to use while skipping frames
            SKIP_FRAME_X = frame_x
            SKIP_FRAME_Y = frame_y
            SKIP_RECT    = (x, y, w, h)
            SKIP_POINTS  = points
            SKIP_DELTAS  = deltas

    ############## OpengGL ##############

    if SKIP_POINTS is None:
        return

    p1, p2, p3 = SKIP_POINTS[39], SKIP_POINTS[42], SKIP_POINTS[33]
    w, h = OUTPUT_WIDTH, OUTPUT_HEIGHT

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

    QUAD_PROGRAM.start()
    toolbox.bind(QUAD)

    height, widht = frame.shape[:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, widht, height, 0, GL_BGR, GL_UNSIGNED_BYTE, frame)

    glDrawElements(GL_TRIANGLES, len(QUAD.elements) * 3, GL_UNSIGNED_INT, None)

    toolbox.unbind()
    QUAD_PROGRAM.stop()

    #-----------------------------------#

    tri_program.start()
    toolbox.bind(triangle)

    glDrawElements(GL_TRIANGLES, len(triangle.elements) * 3, GL_UNSIGNED_INT, None)

    toolbox.unbind()
    tri_program.stop()

    #-----------------------------------#

    glFinish()

    save()

    glutSwapBuffers()

    #####################################

    # Update the progress bar
    FRAME_COUNT += 1
    progressbar.update(FRAME_COUNT / TOTAL_FRAMES)

def quit():
    CAPTURE.release()
    VIDEO.release()
    exit()

def initGL():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    glutCreateWindow("Virtual Window")
    glutDisplayFunc(loop)
    glutIdleFunc(loop)

    print('')
    print('Vendor:          ' + glGetString(GL_VENDOR).decode('utf-8'))
    print('Renderer:        ' + glGetString(GL_RENDERER).decode('utf-8'))
    print('OpenGL Version:  ' + glGetString(GL_VERSION).decode('utf-8'))
    print('Shader Version:  ' + glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8'))

    if not glUseProgram:
        print('Missing Shader Objects!')
        sys.exit(1)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    global QUAD
    QUAD = Quad()

    global QUAD_PROGRAM
    QUAD_PROGRAM = ShaderProgram(fragment=QUAD.fragment, vertex=QUAD.vertex)

    QUAD.loadVBOs(QUAD_PROGRAM)
    QUAD.loadElements()

    progressbar.init()

    glutMainLoop()

def main():
    args = handle_args()

    input_path      = args.i[0]
    output_path     = args.o[0]
    frame_rate      = args.f[0]

    global ROTATION, OUTPUT_WIDTH, OUTPUT_HEIGHT, FRAME_SKIP_RATE, RECOGNIZE_SCALE, PREDICTOR

    ROTATION        = args.r[0]
    OUTPUT_WIDTH    = int(args.c[0].split('x')[0])
    OUTPUT_HEIGHT   = int(args.c[0].split('x')[1])
    FRAME_SKIP_RATE = args.k[0]
    RECOGNIZE_SCALE = 0.2
    PREDICTOR = dlib.shape_predictor('../shapes/shape_predictor_68_face_landmarks.dat')

    create_output_dir(output_path)

    setupOpenCV(input_path, output_path, frame_rate)

    global TOTAL_FRAMES, FRAME_COUNT
    TOTAL_FRAMES = toolbox.getTotalFrames(input_path)
    FRAME_COUNT = 0

    global SKIPPED_FRAMES, SKIP_FRAME_X, SKIP_FRAME_Y, SKIP_RECT, SKIP_POINTS, SKIP_DELTAS
    SKIPPED_FRAMES = 0
    SKIP_FRAME_X = None
    SKIP_FRAME_Y = None
    SKIP_RECT    = None
    SKIP_POINTS  = None
    SKIP_DELTAS  = None

    initGL()

if __name__ == '__main__':
    main()