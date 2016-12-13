#!/usr/bin/env python

import sys
import os
import glob
from skimage import io, draw
from skimage import transform as tf

import math
import numpy as np
from scipy.spatial import Delaunay

import dlib

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
INPUT_PATH = "input"
OUTPUT_PATH = "output"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

tform3 = tf.ProjectiveTransform()

for f in glob.glob(os.path.join(INPUT_PATH, "*.jpg")):
    file_name = f.replace(INPUT_PATH + "/", "")

    print("Processing file: {}".format(file_name))
    img = io.imread(f)

    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)

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

            ################# Draw Sub-Triangles #################
            # Get length of all sides to figure out the size of the triangle
            dist1 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            dist2 = math.hypot(p3[0] - p2[0], p3[1] - p2[1])
            dist3 = math.hypot(p1[0] - p3[0], p1[1] - p3[1])

            # Only draw if the triangle is big enough
            if dist1 > 30 and dist2 > 30 and dist3 > 30:
                cX = int((p1[0] + p2[0] + p3[0]) / 3) # X position of center point
                cY = int((p1[1] + p2[1] + p3[1]) / 3) # Y position of center point

                red = np.array([255, 0, 0])

                img[draw.line(p1[0], p1[1], cX, cY)] = red
                img[draw.line(p2[0], p2[1], cX, cY)] = red
                img[draw.line(p3[0], p3[1], cX, cY)] = red
            ############## End (Draw Sub-Triangles) ##############

            # src = np.array((
            #     (0, 0),
            #     (0, 50),
            #     (300, 50),
            #     (300, 0)
            # ))
            # dst = np.array((
            #     (155, 15),
            #     (65, 40),
            #     (260, 130),
            #     (360, 95)
            # ))

            # tform3.estimate(src, dst)
            # warped = tf.warp(img, tform3)
            # warped = tf.warp(img, tform3, output_shape=(50, 300, 3))

            # img[50:100, 50:350] = warped

    io.imsave(os.path.join(OUTPUT_PATH, file_name), img)
