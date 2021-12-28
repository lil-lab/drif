import json
import random
import os
import imageio
import numpy as np
from scipy import ndimage
import math

from grounding_data.geom_calc import calc_object_radius_px, crop_square_recenter
from grounding_data.panda3d_gendata import MATCHING_DATA_ONLY

from visualization import Presenter

data_dir_in = "/media/clic/BigStore/grounding_data_rss2020/grounding_data_sim_8/raw"
data_dir_out = "/media/clic/BigStore/grounding_data_rss2020/grounding_data_sim_8/composed_03"

mscoco_images_dir = "/media/clic/BigStore/mscoco/train2017"
mscoco_images_list = os.listdir(mscoco_images_dir)

MAX_PAD = 20
#MIN_CROP_SIZE = 24
#TODO: Check this:
MIN_CROP_SIZE = 16

GRAIN_PROB = 0.0
#GRAIN_PROB = 0.0  # Don't add noise in simulation, because simulated images don't tend to have noise
CONTRAST_PROB = 0.5  # Probability of altering object contrast
BRIGHTNESS_PROB = 0.5  # Probability of altering object brightness

NO_SHADOWS = False

SWAP_BACKGROUND_PROB = 0.0


def _get_scene_indices(data_dir):
    all_files = os.listdir(data_dir)
    prefixes = [f.split("--")[0] for f in all_files]
    long_prefixes = [p for p in prefixes if len(p.split("_")) >= 3]
    scene_indices = list(sorted(set([tuple([int(x) for x in p.split("_")]) for p in long_prefixes])))
    return scene_indices


def _load_config(scene_index, data_dir_in):
    fname = os.path.join(data_dir_in, f"{scene_index[0]}--config.json")
    with open(fname, "r") as fp:
        config = json.load(fp)
    return config


def _load_bboxes(si, data_dir_in):
    fname = os.path.join(data_dir_in, f"{si[0]}_{si[1]}_{si[2]}--geometry.json")
    with open(fname, "r") as fp:
        bboxes = json.load(fp)
    return bboxes


def _convert_to_rgba(img, mask=False):
    # Duplicate channels to convert from grayscale to RGB
    if len(img.shape) == 2:
        img = np.concatenate([img[:, :, np.newaxis]] * 3, axis=2)
    # Add alpha channel to convert from RGB to RGBA
    if img.shape[2] == 3:
        if mask and np.max(img) - np.min(img) < 1e-2:
            img = np.concatenate([img, np.zeros_like(img[:, :, 0])[:, :, np.newaxis]], axis=2)
        else:
            img = np.concatenate([img, np.ones_like(img[:, :, 0])[:, :, np.newaxis]], axis=2)
    return img


def _load_mscoco_background():
    img_name = random.sample(mscoco_images_list, 1)[0]
    path = os.path.join(mscoco_images_dir, img_name)
    img = np.asarray(imageio.imread(path), dtype=np.float32) / 255
    return img


def _maybe_swap_background(bg_img):
    if np.random.binomial(1, SWAP_BACKGROUND_PROB, 1)[0] > 0.5:
        alt_bg = _load_mscoco_background()
        # Make sure the background is big enough
        if alt_bg.shape[0] < bg_img.shape[0] or alt_bg.shape[1] < bg_img.shape[1]:
            return bg_img
        # Take a random crop the size of the current background
        top = int(random.uniform(0, alt_bg.shape[0] - bg_img.shape[0] - 1))
        left = int(random.uniform(0, alt_bg.shape[1] - bg_img.shape[1] - 1))
        alt_bg_crop = alt_bg[top:top+bg_img.shape[0], left:left+bg_img.shape[1]]
        bg_img = alt_bg_crop
    return bg_img


def _augment_colors(img):
    # Randomly reduce contrast with 50% probability
    if np.random.binomial(1, CONTRAST_PROB, 1) > 0.5:
        factor = float(np.random.uniform(0.6, 0.8))
        img = np.clip(0.5 + factor * img - factor * 0.5, 0, 1)
    # Randomly change the brightness with 50% probability
    if np.random.binomial(1, BRIGHTNESS_PROB, 1) > 0.5:
        offset = float(np.random.uniform(-0.2, 0.0))
        img = np.clip(img + offset, 0, 1)
    return img


def _scale_object_brightness(img, scene_img):
    max_scene_brightness = np.max(scene_img)
    #max_scene_brightness = 0.9
    max_obj_brightness = np.max(img)
    factor = max_scene_brightness / (max_obj_brightness + 1e-9)
    if factor < 1.0:
        img = img * factor
    return img


def _load_images(si, config, data_dir_in):
    scene_image = np.asarray(imageio.imread(
        os.path.join(data_dir_in, f"{si[0]}_{si[1]}_{si[2]}--scene.png")), dtype=np.float32) / 255
    noshadow_image = np.asarray(imageio.imread(
        os.path.join(data_dir_in, f"{si[0]}_{si[1]}_{si[2]}--noshadows.png")), dtype=np.float32) / 255
    ground_image = np.asarray(imageio.imread(
        os.path.join(data_dir_in, f"{si[0]}_{si[1]}_{si[2]}--ground.png")), dtype=np.float32) / 255
    background_image = np.asarray(imageio.imread(
        os.path.join(data_dir_in, f"{si[0]}_{si[1]}_{si[2]}--background.png")), dtype=np.float32) / 255

    background_image = _maybe_swap_background(background_image)
    #Presenter().show_image(background_image, "bg", scale=4, waitkey=True)

    object_image = np.asarray(imageio.imread(
        os.path.join(data_dir_in, f"{si[0]}_{si[1]}_{si[2]}--objects.png")), dtype=np.float32) / 255

    # Convert everything to RGBA for consistency
    # (e.g. frames that are all black and white are saved as Grayscale, background doesn't have alpha)
    scene_image = _convert_to_rgba(scene_image)
    noshadow_image = _convert_to_rgba(noshadow_image)
    ground_image = _convert_to_rgba(ground_image, mask=True)
    background_image = _convert_to_rgba(background_image)
    object_image = _convert_to_rgba(object_image, mask=True)

    if NO_SHADOWS:
        scene_image = noshadow_image

    # Re-size background image to match the rendered images
    scale = int(scene_image.shape[0] / background_image.shape[0])
    background_image = Presenter().scale_image(background_image, scale)
    assert background_image.shape == scene_image.shape, \
        f"Background ({background_image.shape}) and scene ({scene_image.shape}) shapes don't match!"

    assert len(scene_image.shape) == 3 and scene_image.shape[2] == 4
    assert len(noshadow_image.shape) == 3 and noshadow_image.shape[2] == 4
    assert len(ground_image.shape) == 3 and ground_image.shape[2] == 4
    assert len(background_image.shape) == 3 and background_image.shape[2] == 4
    assert len(object_image.shape) == 3 and object_image.shape[2] == 4

    # Threshold the ground to a 0-1 valued mask
    ground_image = (ground_image > 0.05).all(2).astype(np.float32)[:, :, np.newaxis]

    images_in = {
        "scene": scene_image,
        "objects": object_image,
        "noshadow": noshadow_image,
        "ground": ground_image,
        "background": background_image,
        "masks": [],
        "mask_ids": []
    }
    if not MATCHING_DATA_ONLY:
        for landmark_name in set(config["landmarkName"]):
            enabled = False
            for lmn, lmenbl in zip(config["landmarkName"], config["enabled"]):
                if lmn == landmark_name and lmenbl:
                    enabled = True
            if not enabled:
                continue
            landmark_mask_image = np.asarray(imageio.imread(
                os.path.join(data_dir_in, f"{si[0]}_{si[1]}_{si[2]}--object_{landmark_name}.png")), dtype=np.float32) / 255
            # Threshold
            # .all(2)
            landmark_mask_image = (landmark_mask_image > 0.05)
            if len(landmark_mask_image.shape) > 2:
                landmark_mask_image = landmark_mask_image[:, :, :3].any(2)
            landmark_mask_image = landmark_mask_image.astype(np.float32)[:, :, np.newaxis]
            images_in["masks"].append(landmark_mask_image)
            images_in["mask_ids"].append(landmark_name)
    return images_in


def _matching_compose_scene_image(images_in):
    # First compute shadow mask that represents shadows on the ground plane only
    shadow_mask_absolute = (images_in["scene"] - images_in["noshadow"]) * images_in["ground"]
    # Then get relative attentuation as a fraction of no-shadow intensity
    shadow_mask_relative = shadow_mask_absolute / (images_in["noshadow"] + 1e-9)
    # Clip below zero - shadows should only attenuate and not increase intensity
    shadow_mask_relative_attenuation = (shadow_mask_relative * 1.0).clip(-1, 0)

    # Overlay objects over the background
    # Create a mask indicating where the objects are
    #objects_mask = (images_in["objects"][:, :, 3] > 0.1).astype(np.float32)[:, :, np.newaxis]

    objects_mask = images_in["objects"][:, :, 3:4]
    objects = images_in["objects"][:, :, :3]

    # Slightly blur object boundaries, but make sure not to bleed object boundaries bigger than they should be
    objects_mask_precompose = objects_mask
    #objects_mask_precompose = np.minimum(ndimage.gaussian_filter(objects_mask, sigma=1), objects_mask)
    #objects_precompose = ndimage.gaussian_filter(images_in["objects"][:, :, :3], sigma=1)

    background_precompose = images_in["background"][:, :, :3]
    shadow_mask_ra_precompose = shadow_mask_relative_attenuation[:, :, :3]

    # Randomly alter object overlay
    objects_precompose = _random_grain(objects)
    objects_precompose = _augment_colors(objects_precompose)
    objects_precompose = _scale_object_brightness(objects_precompose, background_precompose)

    scene_with_objects = background_precompose * (1 - objects_mask_precompose) + objects_precompose * objects_mask_precompose
    # Overlay shadows over the objects image
    composed_scene = scene_with_objects * (1 + shadow_mask_ra_precompose)
    return composed_scene


def _compose_scene_image(images_in):
    # First compute shadow mask that represents shadows on the ground plane only
    shadow_mask_absolute = (images_in["scene"] - images_in["noshadow"]) * images_in["ground"]
    # Then get relative attentuation as a fraction of no-shadow intensity
    shadow_mask_relative = shadow_mask_absolute / (images_in["noshadow"] + 1e-9)
    # Clip below zero - shadows should only attenuate and not increase intensity
    shadow_mask_relative_attenuation = (shadow_mask_relative * 1.0).clip(-1, 0)

    # Overlay objects over the background
    # Create a mask indicating where the objects are
    #objects_mask = (images_in["objects"][:, :, 3] > 0.1).astype(np.float32)[:, :, np.newaxis]

    objects_mask = images_in["objects"][:, :, 3:4]
    objects = images_in["objects"][:, :, :3]

    objects_mask = _resize(objects_mask, 0.25, lanczos=True)
    objects_mask = _resize(objects_mask, 4, linear=True)[:, :, np.newaxis]
    objects = _resize(objects, 0.25, lanczos=True)
    objects = _resize(objects, 4, linear=True)

    # Slightly blur object boundaries, but make sure not to bleed object boundaries bigger than they should be
    # Resize down and back up to blur it a bit
    objects_mask_precompose = objects_mask

    background_precompose = images_in["background"][:, :, :3]
    shadow_mask_ra_precompose = shadow_mask_relative_attenuation[:, :, :3]

    # Randomly alter object overlay
    objects_precompose = _random_grain(objects)
    objects_precompose = _augment_colors(objects_precompose)
    objects_precompose = _scale_object_brightness(objects_precompose, background_precompose)

    scene_with_objects = background_precompose * (1 - objects_mask_precompose) + objects_precompose * objects_mask_precompose
    # Overlay shadows over the objects image
    composed_scene = scene_with_objects * (1 + shadow_mask_ra_precompose)
    return composed_scene


def _random_grain(image):
    if np.random.binomial(1, GRAIN_PROB, 1) > 0.5:
        amplitude = np.random.uniform(0, 0.05)
        grain = np.random.normal(0, amplitude, image.shape)
        image = np.clip(image + grain, 0.0, 1.0)
    return image


def _get_object_pictures(composed_image, bboxes):
    object_ids_out = []
    object_crops_out = []
    for object_id, bbox in zip(bboxes["landmarkNames"], bboxes["landmarkBboxes"]):
        if bbox is not None:
            lb, ru = bbox
            pad_l, pad_r, pad_u, pad_b = np.random.uniform(1, MAX_PAD, 4).astype(np.int32)
            min_y = max(min(lb[1] - pad_u, composed_image.shape[0]), 0)
            max_y = max(min(ru[1] + pad_b, composed_image.shape[0]), 0)
            min_x = max(min(lb[0] - pad_l, composed_image.shape[1]), 0)
            max_x = max(min(ru[0] + pad_r, composed_image.shape[1]), 0)
            object_crop = composed_image[min_y:max_y, min_x:max_x, :]
            object_ids_out.append(object_id)
            object_crops_out.append(object_crop)
    return object_ids_out, object_crops_out


def _get_center_based_object_pictures(composed_image, bboxes):
    object_ids_out = []
    object_crops_out = []
    for object_id, object_bottom_center, object_coords in zip(bboxes["landmarkNames"], bboxes["landmarkCenters"], bboxes["landmarkCoordsInCam"]):
        if object_bottom_center is not None:
            dst_to_object = np.linalg.norm(object_coords)
            obj_radius_px = calc_object_radius_px(dst_to_object, composed_image.shape[1])

            noisy_center = np.asarray(object_bottom_center) + np.random.normal([0, 0], obj_radius_px * 0.2)
            noisy_center = [noisy_center[1], noisy_center[0]]
            object_crop = crop_square_recenter(composed_image, noisy_center, obj_radius_px, shift=0.2)

            object_ids_out.append(object_id)
            object_crops_out.append(object_crop)
    return object_ids_out, object_crops_out


def _resize(img, scale, lanczos=False, linear=False):
    #sigma = [1, 1] + [0] * (len(img.shape) - 2)
    #img_blur = ndimage.gaussian_filter(img, sigma=sigma)
    img_blur = img
    return Presenter().scale_image(img_blur, scale, interpolation="lanczos" if lanczos else ("linear" if linear else "nearest"))


def _save_data(si, data_dir_out, out_images, config):
    scene_id = f"{si[0]}_{si[1]}_{si[2]}"
    #scenes_dir = os.path.join(data_dir_out, "scenes_full")
    scenes_small_dir = os.path.join(data_dir_out, "scenes")
    objects_dir = os.path.join(data_dir_out, "objects")
    c_objects_dir = os.path.join(data_dir_out, "c_objects")
    masks_dir = os.path.join(data_dir_out, "masks")
    masks_subdir = os.path.join(masks_dir, scene_id)
    #os.makedirs(scenes_dir, exist_ok=True)
    os.makedirs(objects_dir, exist_ok=True)
    os.makedirs(c_objects_dir, exist_ok=True)
    os.makedirs(masks_subdir, exist_ok=True)
    os.makedirs(scenes_small_dir, exist_ok=True)

    #toi8 = lambda x: (x * 255).astype(np.uint8)
    toi8 = lambda x: x

    # Save composed scene
    composed_scene = out_images["composed_scene"]
    #imageio.imsave(os.path.join(scenes_dir, f"{scene_id}--composed_scene.png"), composed_scene)
    # Save down-sized scene at drone camera resolution
    out_height = 96
    scale = out_height / float(composed_scene.shape[0])
    # Blur to iron-out aliasing around object edges
    scene_small = _resize(composed_scene, scale, lanczos=True)
    imageio.imsave(os.path.join(scenes_small_dir, f"{scene_id}--composed_scene.png"), toi8(scene_small))

    # Save object instance masks
    for obj_id, mask in zip(out_images["object_ids"], out_images["object_masks"]):
        try:
            mask_small = _resize(mask, scale)
        except Exception as e:
            print("ERROR SAVING MASK!")
            continue
        imageio.imsave(os.path.join(masks_subdir, f"{obj_id}.png"), toi8(mask_small))

    # Save object query images
    for obj_id, cropped_img in zip(out_images["crop_ids"], out_images["crop_images"]):
        object_dir = os.path.join(objects_dir, obj_id)
        # Skip observations that are way too small to recognize an object
        if min(cropped_img.shape[:2]) < (MIN_CROP_SIZE / scale):
            continue
        os.makedirs(object_dir, exist_ok=True)
        cropped_img_small = _resize(cropped_img, scale)
        imageio.imsave(os.path.join(object_dir, f"query_{obj_id}_{scene_id}.png"), toi8(cropped_img_small))

    # Save center-based object query images
    for obj_id, cropped_img in zip(out_images["c_crop_ids"], out_images["c_crop_images"]):
        c_object_dir = os.path.join(c_objects_dir, obj_id)
        # Skip observations that are way too small to recognize an object
        if min(cropped_img.shape[:2]) < (MIN_CROP_SIZE / scale):
            continue
        os.makedirs(c_object_dir, exist_ok=True)
        cropped_img_small = _resize(cropped_img, scale)
        imageio.imsave(os.path.join(c_object_dir, f"query_{obj_id}_{scene_id}.png"), toi8(cropped_img_small))


def process_scene_index(scene_index):
    random.seed()
    np.random.seed()
    config = _load_config(scene_index, data_dir_in)
    bboxes = _load_bboxes(scene_index, data_dir_in)
    try:
        images_in = _load_images(scene_index, config, data_dir_in)
    except FileNotFoundError as e:
        print(f"Skipping scene: {scene_index}")
        return
    print(f"                                           Processing scene: {scene_index}")

    if MATCHING_DATA_ONLY:
        composed_scene = _matching_compose_scene_image(images_in)
    else:
        composed_scene = _compose_scene_image(images_in)

    cropped_ids, cropped_objects = _get_object_pictures(composed_scene, bboxes)
    c_cropped_ids, c_cropped_objects = _get_center_based_object_pictures(composed_scene, bboxes)

    out_images = {
        "composed_scene": composed_scene,
        "object_masks": images_in["masks"],
        "object_ids": images_in["mask_ids"],
        "crop_ids": cropped_ids,
        "crop_images": cropped_objects,
        "c_crop_ids": c_cropped_ids,
        "c_crop_images": c_cropped_objects
    }
    _save_data(scene_index, data_dir_out, out_images, config)


MULTI_THREADED = False


def composit_data():
    scene_indices = _get_scene_indices(data_dir_in)

    if MULTI_THREADED:
        from multiprocessing import Pool
        pool = Pool(28)
        pool.map(process_scene_index, scene_indices)
    else:
        for scene_index in scene_indices:
            process_scene_index(scene_index)


if __name__ == "__main__":
    composit_data()
