import json
import numpy as np

object_ids_path = "/home/clic/shelf_space/shapenet_sem/object_ids.json"
out_colors_path = "/home/clic/shelf_space/shapenet_sem/object_colors.json"

COLOR_PROB = 0.9

if __name__ == "__main__":
    with open(object_ids_path, "r") as fp:
        object_ids = json.load(fp)
    colors = {}
    for object_id in object_ids:
        coin = np.random.binomial(1, COLOR_PROB, 1)
        if coin > 0.5:
            rand_color = list(np.random.uniform(20, 255, 3))
            rand_color = [int(c) for c in rand_color]
            colors[object_id] = rand_color

    with open(out_colors_path, "w") as fp:
        json.dump(colors, fp)
    print(f"Done! Generated colors for {len(colors)} out of {len(object_ids)} objects!")