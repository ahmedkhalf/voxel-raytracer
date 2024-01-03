import moderngl
import numpy as np
from PIL import Image
from midvoxio.voxio import vox_to_arr
import time
from pathlib import Path


filename = Path.cwd() / "resources" / "chr_knight.vox"
arr = vox_to_arr(filename)
arr = arr.swapaxes(1, 2)
arr = arr[:, :, ::-1, :]
arr_size = width, height, length = (arr.shape[0], arr.shape[1], arr.shape[2])
arr = arr.astype(np.float32)
# arr = np.reshape(arr, (arr.shape[0] * arr.shape[1] * arr.shape[2], ))


model_arr = np.zeros(arr.shape[0] * arr.shape[1] * arr.shape[2] * arr.shape[3], dtype=np.float32)
for x in range(width):
    for y in range(height):
        for z in range(length):
            model_arr[(x + y * width + z * width * height) * 4] = arr[x, y, z][0]
            model_arr[(x + y * width + z * width * height) * 4 + 1] = arr[x, y, z][1]
            model_arr[(x + y * width + z * width * height) * 4 + 2] = arr[x, y, z][2]
            model_arr[(x + y * width + z * width * height) * 4 + 3] = arr[x, y, z][3]


# Read fragment shader code from a file
with open('gpu.frag', 'r') as file:
    fragment_shader_code = file.read()

# Minimal vertex shader
vertex_shader_code = """
#version 430

in vec2 in_pos;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Create a moderngl context and program
ctx = moderngl.create_standalone_context()
prog = ctx.program(
    vertex_shader=vertex_shader_code,
    fragment_shader=fragment_shader_code
)

# Set up iResolution uniform as uvec2
prog['iResolution'] = (800, 600)
prog['models[0].size'] = arr_size
# prog['numModels'] = 1
# prog['models[0].position'] = np.array([0.5, 2, 3], dtype=np.float32)

# Set up the SSBO
ssbo_data = model_arr
ssbo = ctx.buffer(ssbo_data)
ssbo.bind_to_storage_buffer(0)

# Create a framebuffer and render to texture
fbo = ctx.simple_framebuffer((800, 600))
fbo.use()

# Render a quad to trigger the fragment shader
vbo = ctx.buffer(
    np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
)
vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')

start = time.time()
vao.render(moderngl.TRIANGLE_STRIP)

# Read the result from the framebuffer
pixels = fbo.read(components=3)
end = time.time()
print(f"Rendered in {(end - start) * 100:.2f} milliseconds")

image = Image.frombytes('RGB', fbo.size, pixels)
image = image.transpose(Image.FLIP_TOP_BOTTOM)
image.save("temp.png")
