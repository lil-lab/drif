from PIL import Image
import numpy as np


def imresize(arr, size, interp, mode):
    return np.array(Image.fromarray(arr).resize(size))