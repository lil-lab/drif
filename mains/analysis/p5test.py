import os
import p5


def setup():
    p5.size(640, 360)
    p5.no_stroke()
    p5.background(204)


def draw():
    if mouse_is_pressed:
        p5.fill(p5.random_uniform(255), p5.random_uniform(127), p5.random_uniform(51), 127)
    else:
        p5.fill(255, 15)

    circle_size = p5.random_uniform(low=10, high=80)

    p5.circle((mouse_x, mouse_y), circle_size)
    os.makedirs("cakeshop", exist_ok=True)
    print("ding")
    return False


def key_pressed(event):
    p5.background(204)
    p5.save("cakeshop/cake.png")

p5.loop()
print("end")