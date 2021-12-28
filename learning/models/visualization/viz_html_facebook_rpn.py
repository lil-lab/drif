import os
import imageio
import numpy as np
import torch
from learning.inputs.partial_2d_distribution import Partial2DDistribution
from visualization import Presenter

import learning.models.visualization.object_recognition_templates as tplt


def export_resources(root_dir, resource_dict):
    for n, img in resource_dict.items():
        fpath = os.path.join(root_dir, n)
        if isinstance(img, Partial2DDistribution):
            img = img.visualize()
        img_save = Presenter().prep_image(img)
        imageio.imsave(fpath, img_save)


def save_html(root_dir, html_string):
    fname = os.path.join(root_dir, "0dashboard.html")
    with open(fname, "w") as fp:
        fp.write(html_string)


def update_index(run_dir):
    all_subdirs = [dir for dir in os.listdir(run_dir) if dir != "index.html"]
    all_subdirs = [int(subdir) for subdir in all_subdirs]
    index_html = "<html><body><ul>"
    for subdir in sorted(all_subdirs):
        index_html += f"<li><a href={subdir}/0dashboard.html>{subdir}</a></li>"
    index_html += "</ul></body></html>"
    fname = os.path.join(run_dir, "index.html")
    with open(fname, "w") as fp:
        fp.write(index_html)


def draw_boxes(scene_image, bboxes):
    scene_image = scene_image.clone()
    col = torch.tensor([0, 0, 0])
    col[0] = scene_image.max()
    col = col[:, np.newaxis]
    for bbox in bboxes:
        bbox = bbox.detach().cpu()
        x_min, y_min, x_max, y_max = tuple([int(x.item()) for x in bbox])
        y_min = min(max(y_min, 0), scene_image.shape[1] - 1)
        x_min = min(max(x_min, 0), scene_image.shape[2] - 1)
        y_max = min(max(y_max, 0), scene_image.shape[1] - 1)
        x_max = min(max(x_max, 0), scene_image.shape[2] - 1)

        scene_image[:, y_min:y_max, x_min] = col
        scene_image[:, y_min: y_max, x_max] = col
        scene_image[:, y_min, x_min:x_max] = col
        scene_image[:, y_max, x_min:x_max] = col
    return scene_image


def visualize_model(scene_images, bboxes, iteration, run_name):
    # ------------------ Prep inputs ------------------------------------------
    scene_images = scene_images.detach().cpu()
    #bboxes = bboxes.detach().cpu()

    # ------------------ Set up  ----------------------------------------------
    run_dir = os.path.expanduser(f"~/dashboard_test_dir/{run_name}/")
    root_dir = os.path.join(run_dir, f"{iteration}")
    rel_image_dir = "images"
    abs_image_dir = os.path.join(root_dir, rel_image_dir)
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(abs_image_dir, exist_ok=True)
    resources = {}
    html = tplt.OBJECT_PROPOSAL_TEMPLATE_HTML
    rows = ""

    # ------------------ Layout  ----------------------------------------------
    for i, (scene_image, bboxes) in enumerate(zip(scene_images, bboxes)):
        image_fname = f"prediction_{i}.png"
        image_with_boxes = draw_boxes(scene_image, bboxes)
        resources[image_fname] = image_with_boxes
        rows += tplt.fill(tplt.OBJECT_PROPOSAL_ROW_HTML_TEMPLATE, "IMG_SRC", image_fname)
    html = tplt.fill(html, "OBJECT_PROPOSALS", rows)

    # ------------------ Save  ----------------------------------------------
    export_resources(root_dir, resources)
    save_html(root_dir, html)
    update_index(run_dir)
