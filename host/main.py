#!/usr/bin/env python3

from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image
import numpy as np
import cv2
import sys

from shader import ShaderProgram

from quad import *

SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768

def init(width, height):
    print('Vendor:          ' + glGetString(GL_VENDOR).decode('utf-8'))
    print('Renderer:        ' + glGetString(GL_RENDERER).decode('utf-8'))
    print('OpenGL Version:  ' + glGetString(GL_VERSION).decode('utf-8'))
    print('Shader Version:  ' + glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8'))

    if not glUseProgram:
        print('Missing Shader Objects!')
        sys.exit(1)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    global capture, video
    capture = cv2.VideoCapture('input.mp4', 0)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter()
    video.open('output.mp4', fourcc, 30, (SCREEN_WIDTH, SCREEN_HEIGHT), True)

    # color_data = np.array(
    #     [
    #         [1, 0, 0],
    #         [1, 1, 0],
    #         [0, 1, 0],
    #         [0, 0, 1]

    #     ],
    #     dtype=np.float32)

    global quad
    quad = Quad()

    global program
    program = ShaderProgram(fragment=quad.fragment, vertex=quad.vertex)

    quad.loadVBOs(program)
    quad.loadElements()

def prepare():
    glClear(GL_COLOR_BUFFER_BIT)
    glClearColor (0.0, 0.0, 0.0, 1.0)

def bind(model):
    glBindVertexArray(model.vao)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, model.ebo)

def unbind():
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

def save():
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    buffer = glReadPixels(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE)

    image = Image.frombytes('RGB', (SCREEN_WIDTH, SCREEN_HEIGHT), buffer)     
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    output = np.asarray(image, dtype=np.uint8)

    video.write(output)

def loop():
    prepare()

    program.start()
    bind(quad)

    _, frame = capture.read()
    
    if frame is None:
        capture.release()
        video.release()
        exit()

    height, widht = frame.shape[:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, widht, height, 0, GL_BGR, GL_UNSIGNED_BYTE, frame)

    glDrawElements(GL_TRIANGLES, len(quad.elements) * 3, GL_UNSIGNED_INT, None)

    unbind()
    program.stop()

    glFinish()

    save()

    glutSwapBuffers()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT)
    glutCreateWindow("Virtual Window")
    glutDisplayFunc(loop)
    glutIdleFunc(loop)
    init(SCREEN_WIDTH, SCREEN_HEIGHT)
    glutMainLoop()

if __name__ == '__main__':
    main()
