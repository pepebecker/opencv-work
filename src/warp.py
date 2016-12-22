#!/usr/bin/env python3

import cv2
import numpy as np

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

if __name__ == '__main__' :
    imgIn = cv2.imread("robot.jpg")
    
    triIn = np.float32([[[220,200], [360,200], [290,250]]])
    triOut = np.float32([[[400,200], [160,270], [400,400]]])

    cv2.polylines(imgIn, triIn.astype(int), True, (0, 0, 255), 2, 16)
    
    warpTriangle(imgIn, imgIn, triIn, triOut)

    cv2.imshow("Input", imgIn)
    
    cv2.waitKey(0)
