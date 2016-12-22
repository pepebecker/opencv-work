#!/usr/bin/env python3

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
from PIL import Image
import numpy as np
import cv2
import sys

from shader import ShaderProgram

SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
ESCAPE = '\x1b'

def keyPressed(*args):
    if (args[0].decode("utf-8") == ESCAPE):
        sys.exit()

def init(width, height):
    glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')
    print('Vendor:          ' + glGetString(GL_VENDOR).decode('utf-8'))
    print('Renderer:        ' + glGetString(GL_RENDERER).decode('utf-8'))
    print('OpenGL Version:  ' + glGetString(GL_VERSION).decode('utf-8'))
    print('Shader Version:  ' + glslVersion)

    if not glUseProgram:
        print('Missing Shader Objects!')
        sys.exit(1)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    global capture
    capture = cv2.VideoCapture(0)

    vertex_data = np.array(
        [
            [-1,  1],
            [ 1,  1],
            [ 1, -1],
            [-1, -1]
        ],
        dtype=np.float32)

    uv_data = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]

        ],
        dtype=np.float32)

    color_data = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1]

        ],
        dtype=np.float32)

    element_data = np.array(
        [
            [0, 1, 2],
            [2, 3, 0]

        ],
        dtype=np.int32)

    vertex = """
        #version {version}

        in vec2 position;
        in vec2 uv;
        in vec3 color;

        out vec2 pass_uv;
        out vec3 pass_color;

        void main(void)
        {
            pass_uv = uv;
            pass_color = color;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """.replace('{version}', glslVersion.replace('.', ''))

    fragment = """
        #version {version}

        in vec2 pass_uv;
        in vec3 pass_color;

        out vec4 out_color;

        uniform sampler2D textureSampler;

        void main(void)
        {
            vec4 textureColor = texture(textureSampler, pass_uv);
            vec4 color = vec4(pass_color, 1.0);
            out_color = mix(color, textureColor, 0.5f);
        }
        """.replace('{version}', glslVersion.replace('.', ''))

    # projection = gluOrtho2D(0, 0, 0, 0)

    global program
    program = ShaderProgram(fragment=fragment, vertex=vertex)

    global vao
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(3)

    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(vertex_data), vertex_data, GL_STATIC_DRAW)
    glVertexAttribPointer(program.attribute_location('position'), 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(uv_data), uv_data, GL_STATIC_DRAW)
    glVertexAttribPointer(program.attribute_location('uv'), 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
    glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(color_data), color_data, GL_STATIC_DRAW)
    glVertexAttribPointer(program.attribute_location('color'), 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(2)

    elementbuffer = glGenBuffers(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(element_data), element_data, GL_STATIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

def loop():
    glClear(GL_COLOR_BUFFER_BIT)
    glClearColor (0.0, 0.0, 0.0, 1.0)

    glUseProgram(program.program_id)
    glBindVertexArray(vao)

    _, frame = capture.read()
    
    if frame is None:
        exit()

    height, widht = frame.shape[:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, widht, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame)

    # glDrawArrays(GL_TRIANGLE_STRIP, 0, 6)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    glUseProgram(0)
    glBindVertexArray(0)

    glutSwapBuffers()

def main():
    global window
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("OpenGL Window")
    glutDisplayFunc(loop)
    glutIdleFunc(loop)
    glutKeyboardFunc(keyPressed)
    init(SCREEN_WIDTH, SCREEN_HEIGHT)
    glutMainLoop()


if __name__ == '__main__':
    main()