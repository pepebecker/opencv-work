from OpenGL.GL import *
import numpy as np

class Polygon:
    def __init__(self, verticies, uvs, elements):
        self.verticies = verticies
        self.uvs = uvs
        self.elements = elements

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(2)
        self.ebo = glGenBuffers(1)

    def loadVBOs(self, program):
        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo[0])
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(self.verticies), self.verticies, GL_STATIC_DRAW)
        glVertexAttribPointer(program.attribute_location('position'), 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        if self.uvs is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo[1])
            glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(self.uvs), self.uvs, GL_STATIC_DRAW)
            glVertexAttribPointer(program.attribute_location('uv'), 2, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def loadElements(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(self.elements), self.elements, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
