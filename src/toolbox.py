import cv2
import dlib
import numpy as np
from skimage import color, draw
from scipy.spatial import Delaunay

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

def drawTriangles(image, points, pos_multiplier=1, draw_color=np.array([255, 255, 0])):
	tri = Delaunay(points)
	for t in tri.simplices.copy():
		p1 = np.array([p * pos_multiplier for p in points[t[0]]]).astype(int)
		p2 = np.array([p * pos_multiplier for p in points[t[1]]]).astype(int)
		p3 = np.array([p * pos_multiplier for p in points[t[2]]]).astype(int)

		image[draw.line(p1[0], p1[1], p2[0], p2[1])] = draw_color
		image[draw.line(p2[0], p2[1], p3[0], p3[1])] = draw_color
		image[draw.line(p3[0], p3[1], p1[0], p1[1])] = draw_color

def warpTriangle(src_img, dst_image, src_tri, dst_tri) :    
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(src_tri)
    r2 = cv2.boundingRect(dst_tri)
    
    # Offset points by left top corner of the respective rectangles
    src_tri_cropped = []
    dst_tri_cropped = []
    
    for i in range(0, 3):
        src_tri_cropped.append(((src_tri[0][i][0] - r1[0]), (src_tri[0][i][1] - r1[1])))
        dst_tri_cropped.append(((dst_tri[0][i][0] - r2[0]), (dst_tri[0][i][1] - r2[1])))

    # Crop input image
    src_img_cropped = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(src_tri_cropped), np.float32(dst_tri_cropped) )
    
    # Apply the Affine Transform just found to the src image
    dst_image_cropped = cv2.warpAffine(src_img_cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), (1.0, 1.0, 1.0), 16, 0);

    dst_image_cropped = dst_image_cropped * mask
    
    # Copy triangular region of the rectangular patch to the output image
    dst_image[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_image[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    
    dst_image[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_image[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + dst_image_cropped

def drawWarpedTriangles(image, points, deltas, pos_multiplier=1):
	image_orig = image.copy()

	tri = Delaunay(points)
	for t in tri.simplices.copy():
		p1 = points[t[0]] * pos_multiplier
		p2 = points[t[1]] * pos_multiplier
		p3 = points[t[2]] * pos_multiplier

		d1 = deltas[t[0]] * pos_multiplier
		d2 = deltas[t[1]] * pos_multiplier
		d3 = deltas[t[2]] * pos_multiplier

		tri_points_src = np.float32([[[p1[1], p1[0]], [p2[1], p2[0]], [p3[1], p3[0]]]])
		tri_points_dst = np.float32([[[d1[1], d1[0]], [d2[1], d2[0]], [d3[1], d3[0]]]])

		warpTriangle(image_orig, image, tri_points_src, tri_points_dst)
