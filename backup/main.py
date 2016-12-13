#!/usr/bin/env python

import sys
import os
import dlib
import glob
import skimage
from skimage import io, draw

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACES_PATH = "faces"

if len(sys.argv) == 2:
    FACES_PATH = sys.argv[1]
elif len(sys.argv) == 3:
    PREDICTOR_PATH = sys.argv[1]
    FACES_PATH = sys.argv[2]
elif len(sys.argv) > 3:
    print("usage: " + sys.argv[0] + " <predictor_file> <faces_dir>")
    exit()

predictor_path = PREDICTOR_PATH
faces_folder_path = FACES_PATH

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

# def get_landmarks(im):
#     rects = detector(im, 1)
    
#     if len(rects) > 1:
#         raise "TooManyFaces"
#     if len(rects) == 0:
#         raise "NoFaces"

#     return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    # win.set_image(img)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        win.add_overlay(shape)

        points = np.array([[p.y, p.x] for p in shape.parts()])

        tri = Delaunay(points)

        for t in tri.simplices.copy():
            p1 = points[t[0]]
            p2 = points[t[1]]
            p3 = points[t[2]]

            yellow = np.array([255, 255, 0])

            img[draw.line(p1[0], p1[1], p2[0], p2[1])] = yellow
            img[draw.line(p2[0], p2[1], p3[0], p3[1])] = yellow
            img[draw.line(p3[0], p3[1], p1[0], p1[1])] = yellow

    win.set_image(img)

    # win.add_overlay(dets)
    dlib.hit_enter_to_continue()
