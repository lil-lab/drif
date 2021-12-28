import math

OBJ_RADIUS = 0.3
HFOV = 84


def calc_object_radius_px(dst_to_object, image_width):
    obj_radius_rad = math.atan2(OBJ_RADIUS, dst_to_object)
    obj_radius_px = math.degrees(obj_radius_rad) * image_width / HFOV
    return obj_radius_px

# TODO:
# shift=0.5 for data8
# shift=0.0 for data6
# what shift for data7?
def crop_square_recenter(image, center, rad_px, shift=0.5):
    clip = lambda x, b, t: int(min(max(x, b), t))

    min_x = clip(center[1] - rad_px, 0, image.shape[1])
    max_x = clip(center[1] + rad_px, 0, image.shape[1])
    min_y = clip(center[0] - rad_px * (1.0 + shift), 0, image.shape[0])
    max_y = clip(center[0] + rad_px * (1.0 - shift), 0, image.shape[0])

    return image[min_y:max_y, min_x:max_x, :]
