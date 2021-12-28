import imageio
import numpy as np
import scipy
import random
import cv2
from PIL import Image


def resize_image_to_square(img, side_length):
    raise DeprecationWarning("resize_image_to_square has been deprecated and is no longer in use")
    sized_image = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((side_length, side_length)))
    sized_image = sized_image.astype(np.float32) / 255
    return sized_image


def image_to_grayscale(np_image):
    img_out = np.mean(np_image, axis=2, keepdims=True).repeat(3, axis=2)
    return img_out


def load_query_image(path, grayscale=False):
    query_img = np.asarray(imageio.imread(path)).astype(np.float32) * (1.0 / 255.0)
    if grayscale:
        query_img = image_to_grayscale(query_img)
    return query_img


def eval_augment_query_image(query_img):
    # Blur
    sigma = 0.5
    img = scipy.ndimage.gaussian_filter(query_img, [sigma, sigma, 0])
    # Noise
    #value_range = np.max(img) - np.min(img)
    #sigma = value_range * 0.001
    #img = np.random.normal(img, sigma)
    #img = np.clip(img, 0.0, 1.0)
    return img


def augment_scene(scene, eval, blur, grain, test_eval_augmentation=False):
        #print("---")
        if not eval and blur and np.random.binomial(1, 0.5, 1) > 0.25:
            sigma = random.uniform(0.2, 2.0)
            scene = scipy.ndimage.gaussian_filter(scene, [sigma, sigma, 0])
            #print("Blur sigma: ", sigma)
        if not eval and grain and np.random.binomial(1, 0.5, 1) > 0.25:
            value_range = np.max(scene) - np.min(scene)
            sigma = random.uniform(value_range * 0.001, value_range * 0.05)
            scene = np.random.normal(scene, sigma)
            #print("grain sigma: ", sigma)
        if eval and test_eval_augmentation:
            scene = eval_augment_query_image(scene)
        #Presenter().show_image(scene, "scene_aug", scale=4, waitkey=True)
        return scene


def augment_query_image(query_img, eval, blur, grain, rotate, flip, test_eval_augmentation):
    # First, transform the image into a square-shaped image
    max_side = max(query_img.shape[:2])

    # With 50% probability, rotate by a random angle
    if not eval and rotate and np.random.binomial(1, 0.5, 1) > 0.5:
        angle = np.random.normal(0, 20, 1)
    else:
        angle = 0
    rows, cols = query_img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    output_image = cv2.warpAffine(query_img, M, (max_side, max_side), borderMode=cv2.BORDER_REPLICATE)
    #Presenter().show_image(output_image, "rotated_img", waitkey=True)
    #print("---")
    # Then randomly flip the image
    if not eval and flip and np.random.binomial(1, 0.5, 1) > 0.5:
        output_image = np.flip(output_image, axis=1)

    if not eval and blur and np.random.binomial(1, 0.5, 1) > 0.5:
        sigma = random.uniform(0.2, 2.0)
        output_image = scipy.ndimage.gaussian_filter(output_image, [sigma, sigma, 0])
        #print("blur sigma: ", sigma)

    if not eval and grain and np.random.binomial(1, 0.5, 1) > 0.5:
        value_range = np.max(output_image) - np.min(output_image)
        sigma = random.uniform(value_range * 0.001, value_range * 0.05)
        output_image = np.random.normal(output_image, sigma)
        #print("grain sigma: ", sigma)

    # Eval augmentation testing
    if eval and test_eval_augmentation:
        output_image = eval_augment_query_image(output_image)
    #Presenter().show_image(output_image, "query_aug", scale=4, waitkey=True)

    return output_image