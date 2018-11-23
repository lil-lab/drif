import os

import imageio
from scipy.misc import imsave

from data_io.paths import get_results_dir
from visualization import filter_for_gif


def save_results_extra_image(run_name, env_id, set_id, seg_id, name, image_np):
    results_dir = get_results_dir(run_name, makedir=True)
    extra_dir = os.path.join(results_dir, "extra")
    os.makedirs(extra_dir, exist_ok=True)

    img_filename = str(env_id) + "_" + str(set_id) + "_" + str(seg_id) + "_extra-" + name + ".jpg"
    img_path = os.path.join(extra_dir, img_filename)

    imsave(img_path, image_np)


def save_results_extra_gif(run_name, env_id, set_id, seg_id, name, image_list):
    results_dir = get_results_dir(run_name, makedir=True)
    extra_dir = os.path.join(results_dir, "extra")
    os.makedirs(extra_dir, exist_ok=True)

    if len(image_list) == 0:
        print ("Empty image list: Not saving GIF!")
        return

    img_filename = str(env_id) + "_" + str(set_id) + "_" + str(seg_id) + "_extra-" + name + ".gif"
    img_path = os.path.join(extra_dir, img_filename)

    image_list_filtered = [filter_for_gif(image) for image in image_list]

    imageio.mimsave(img_path, image_list_filtered, fps=5.0)