#!/usr/bin/env python

import os
import numpy as np
import cv2
import dlib
import argparse
import json

# import my local helper modules
import toolbox

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs=1, type=str,    metavar='input-path',   required=True,          help='Input frame path')
    parser.add_argument('-o', nargs=1, type=str,    metavar='output-path',  required=True,          help='Output frame path')
    parser.add_argument('-r', nargs=1, type=float,  metavar='degree',       default=[0],            help='Rotate the frame by # degree')
    parser.add_argument('-c', nargs=1, type=str,    metavar='1024x768',     default=['1024x768'],   help='Crop the frame to a specified resolution')
    return parser.parse_args()

def create_output_dir(output_path):
    output_dir = os.path.dirname(os.path.realpath(output_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def setupOpenCV(input_path):
    frame = cv2.imread(input_path, -1)
    face_cascade = toolbox.loadCascade('haarcascade_frontalface_default.xml')
    return (frame, face_cascade)

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

def convert(frame, rotation, face_cascade, predictor, recognize_scale, output_width, output_height):
    # Roate the frame
    frame = rotateFrame(frame, rotation)

    scale = (1 / recognize_scale)

    # Get current frame width and height for later usage
    (frame_height, frame_width) = frame.shape[:2]

    # Create a low resolution version of the rotated frame
    small = cv2.resize(frame, (0,0), fx=recognize_scale, fy=recognize_scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    points = None

    # Recognize a face on the low reslution frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        # Scale up coordinates
        x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

        # Create a low resolution version of the frame
        small = cv2.resize(frame, (0,0), fx=recognize_scale, fy=recognize_scale)

        # Find all the landmarks on the face
        rs = recognize_scale
        low_rect = (x * rs, y * rs, w * rs, h * rs)
        shape = predictor(small, toolbox.rect2rectangle(low_rect))
        points = np.array([[p.y * scale, p.x * scale] for p in shape.parts()])

        y = int(min(points[:, 0]))
        h = max(points[:, 0]) - y

        # Crop the frame
        frame_x = int((frame_width - output_width) / 2)
        frame_y = y - int((output_height - h) / 2)
        frame = cropFrame(frame, frame_x, frame_y, output_width, output_height)

        # Normalize points to the cropped frame
        points[:, 0] = np.array([py - frame_y for py in points[:, 0]])
        points[:, 1] = np.array([px - frame_x for px in points[:, 1]])

        # Clamp points into cropped frame
        points[:, 0] = np.array([toolbox.clamp(py, 0, output_height - 1) for py in points[:, 0]])
        points[:, 1] = np.array([toolbox.clamp(px, 0, output_width  - 1) for px in points[:, 1]])

        # Draw Delaunay pattern using the landmarks
        toolbox.drawTriangles(frame, points)

    return frame, points

def main():
    args = handle_args()

    input_path      = args.i[0]
    output_path     = args.o[0]
    rotation        = args.r[0]
    output_width    = int(args.c[0].split('x')[0])
    output_height   = int(args.c[0].split('x')[1])
    recognize_scale = 0.2
    predictor = dlib.shape_predictor('../shapes/shape_predictor_68_face_landmarks.dat')

    create_output_dir(output_path)

    frame, face_cascade = setupOpenCV(input_path)

    frame, points = convert(frame, rotation, face_cascade, predictor, recognize_scale, output_width, output_height)

    filename = os.path.splitext(output_path)[0]
    ext = os.path.splitext(input_path)[1]

    with open(filename + '.json', "w") as file:
        json.dump(points.tolist(), file)

    cv2.imwrite(filename + ext, frame)

if __name__ == '__main__':
    main()
