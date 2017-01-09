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

def setupOpenCV(input_path, output_path, frame_rate, output_width, output_height):
    capture = cv2.VideoCapture(input_path, 0)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter()
    video.open(output_path, fourcc, frame_rate, (output_width, output_height), True)
    face_cascade = toolbox.loadCascade('haarcascade_frontalface_default.xml')
    return (capture, video, face_cascade)

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

def getGLFrame(frame, points, output_width, output_height):
    ############## OpengGL ##############

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

    QUAD_PROGRAM.start()
    toolbox.bind(QUAD)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, frame)

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

    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    buffer = glReadPixels(0, 0, output_width, output_height, GL_BGR, GL_UNSIGNED_BYTE)

    image = Image.frombytes('RGB', (output_width, output_height), buffer)     
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    frame = np.asarray(image, dtype=np.uint8)

    glutSwapBuffers()

    #####################################

    return frame

def processFrame(frame, rotation, frame_skip_rate, face_cascade, predictor, recognize_scale, output_width, output_height, last_params):
    s_frames    = None
    s_frame_x   = None
    s_frame_y   = None
    s_rect      = None
    s_points    = None
    s_deltas    = None

    if last_params is not None:
        s_frames, s_frame_x, s_frame_y, s_rect, s_points, s_deltas = last_params

    points = None

    # Roate the frame
    frame = rotateFrame(frame, rotation)

    scale = (1 / recognize_scale)

    # Check how many frames have been skipped
    if s_frames is not None and s_frames > 0:
        s_frames -= 1

        if s_frame_x is not None and s_frame_y is not None:
            frame = cropFrame(frame, s_frame_x, s_frame_y, output_width, output_height)

        if s_points is not None and s_deltas is not None:
            toolbox.drawWarpedTriangles(frame, s_points, s_deltas)

        points = s_points

        last_params = (s_frames, s_frame_x, s_frame_y, s_rect, s_points, s_deltas)
    else:
        s_frames = frame_skip_rate

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
            toolbox.drawWarpedTriangles(frame, points, deltas)

            # Save values to use while skipping frames
            last_params = (s_frames, frame_x, frame_y, (x, y, w, h), points, deltas)

    return (frame, points, last_params)

def quit(capture, video):
    capture.release()
    video.release()
    exit()

def initGL(output_width, output_height):
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(output_width, output_height)
    glutCreateWindow("Virtual Window")

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

def main():
    args = handle_args()

    input_path      = args.i[0]
    output_path     = args.o[0]
    rotation        = args.r[0]
    output_width    = int(args.c[0].split('x')[0])
    output_height   = int(args.c[0].split('x')[1])
    frame_rate      = args.f[0]
    frame_skip_rate = args.k[0]
    recognize_scale = 0.2
    predictor = dlib.shape_predictor('../shapes/shape_predictor_68_face_landmarks.dat')

    create_output_dir(output_path)

    capture, video, face_cascade = setupOpenCV(input_path, output_path, frame_rate, output_width, output_height)

    initGL(output_width, output_height)

    total_frames = toolbox.getTotalFrames(input_path)
    frame_count = 0

    progressbar.init()

    last_params = None

    while True:
        success, frame = capture.read()
        if success and frame is not None:
            frame, points, last_params = processFrame(frame, rotation, frame_skip_rate, face_cascade, predictor, recognize_scale, output_width, output_height, last_params)
            if frame is not None:
                if points is not None:
                    frame = getGLFrame(frame, points, output_width, output_height)

                video.write(frame)
                frame_count += 1
                progress = frame_count / total_frames
                progressbar.update(progress)

                if progress >= 0.5:
                    break
            else:
                break
        else:
            break

    quit(capture, video)

if __name__ == '__main__':
    main()
