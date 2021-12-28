import os
import numpy as np
import imageio

data_dir_in = "/media/clic/BigStore/grounding_data_both_sim_3_big/raw"

if __name__ == "__main__":

    all_files = os.listdir(data_dir_in)
    bg_files = [f for f in all_files if "background" in f]
    bg_paths = [os.path.join(data_dir_in, f) for f in bg_files]

    min_brightness = np.asarray([1.0, 1.0, 1.0])
    max_brightness = np.asarray([0.0, 0.0, 0.0])
    for i, path in enumerate(bg_paths):
        print(f"Img: {i}/{len(bg_paths)}")
        background_image = np.asarray(imageio.imread(path), dtype=np.float32) / 255
        mx = background_image.reshape((-1, 3)).max(0)
        mn = background_image.reshape((-1, 3)).min(0)
        min_brightness = np.minimum(min_brightness, mn)
        max_brightness = np.maximum(max_brightness, mx)

    print("Max brightness: ", max_brightness)
    print("Min brightness: ", min_brightness)