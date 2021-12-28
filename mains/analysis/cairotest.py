import cairo
import numpy as np
import imageio

WIDTH = 400
HEIGHT = 200

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
context = cairo.Context(surface)
x, y, x1, y1 = 0.1, 0.5, 0.4, 0.9
x2, y2, x3, y3 = 0.6, 0.1, 0.9, 0.5
context.scale(WIDTH, HEIGHT)
context.set_line_width(0.04)
context.move_to(x, y)
context.curve_to(x1, y1, x2, y2, x3, y3)
context.stroke()
context.set_source_rgba(1, 0.2, 0.2, 0.6)
context.set_line_width(0.02)
context.move_to(x, y)
context.line_to(x1, y1)
context.move_to(x2, y2)
context.line_to(x3, y3)
context.stroke()

buf = surface.get_data()
array = np.ndarray(shape=(HEIGHT, WIDTH, 4), dtype=np.uint8, buffer=buf)[:, :, [2, 1, 0, 3]]
imageio.imsave("cairoexample.png", array)
