from polygon import Polygon

class Quad(Polygon):
    def __init__(self):
        verticies = np.array(
            [
                [-1,  1],
                [ 1,  1],
                [ 1, -1],
                [-1, -1]
            ],
            dtype=np.float32)

        uvs = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]

            ],
            dtype=np.float32)

        elements = np.array(
            [
                [0, 1, 2],
                [2, 3, 0]

            ],
            dtype=np.int32)

        self.vertex = """
            #version {version}

            in vec2 position;
            in vec2 uv;

            out vec2 pass_uv;

            void main(void)
            {
                pass_uv = uv;
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """

        self.fragment = """
            #version {version}

            in vec2 pass_uv;

            out vec4 out_color;

            uniform sampler2D textureSampler;

            void main(void)
            {
                vec4 textureColor = texture(textureSampler, pass_uv);
                out_color = textureColor;
            }
            """

        Polygon.__init__(self, verticies, uvs, elements)
