from OpenGL.GL import *
import numpy as np
from polygon import Polygon

class Triangle(Polygon):
    def __init__(self, p1, p2, p3):
        verticies = np.array(
            [
                p1,
                p2,
                p3
            ],
            dtype=np.float32)

        elements = np.array(
            [
                [0, 1, 2]
            ],
            dtype=np.int32)

        self.vertex = """
            #version {version}

            in vec2 position;

            void main(void)
            {
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """

        self.fragment = """
            #version {version}

            out vec4 out_color;

            void main(void)
            {
                out_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
            """

        Polygon.__init__(self, verticies, None, elements)
