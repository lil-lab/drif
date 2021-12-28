import numpy as np
import torch
import os
import cv2
import matplotlib.pyplot as plt
import imageio
from utils.dict_tools import dict_merge

from data_io.env import load_env_img

from visualization import Presenter

from learning.models.visualization import dashboard_template as tplt

from learning.inputs.partial_2d_distribution import Partial2DDistribution

VIZ_WIDTH = 1000
VIZ_HEIGHT = 700

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def gen_overlay_string(full_instruction, snippets):
    # Find obj refs in the instruction
    overlay_str = ""
    rem = full_instruction
    for obj_ref in snippets:
        idx = rem.find(obj_ref)
        overlay_str += " " * idx
        overlay_str += obj_ref
        rem = rem[idx + len(obj_ref):]
    return overlay_str


def visualize_instruction_and_obj_refs(instruction_nl, obj_refs, noun_chunks):
    instr_html = tplt.HTML_INSTR_TEMPLATE
    obj_ref_overlay = gen_overlay_string(instruction_nl, obj_refs)
    noun_chunk_overlay = gen_overlay_string(instruction_nl, noun_chunks)
    instr_html = tplt.multi_fill(instr_html, ["INSTR", "CHUNK_OVERLAY", "REF_OVERLAY"],
                                 [instruction_nl, noun_chunk_overlay, obj_ref_overlay])
    return instr_html


def visualize_similarity_matrix(similarity_matrix, y_labels, x_labels):
    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.matshow(similarity_matrix, interpolation='nearest')
    ax.grid(True)
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)
    fig.colorbar(cax)
    plot_img = fig2data(fig)

    sim_html = tplt.fill(tplt.HTML_SIMILARITY_MATRIX_TEMPLATE, "IMG_SRC", "similarity_matrix.png")
    resources = {"similarity_matrix.png": plot_img}
    return sim_html, resources


def draw_on_canvas(canvas, img, h, w):
    canvas[h:h+img.shape[0], w:w+img.shape[1], :] = img


def visualize_seg_masks(image, seg_masks, novel_obj_refs, frame_width):
    col = np.zeros((400, frame_width, 3))
    w = int((frame_width - image.shape[1]) / 2)
    draw_on_canvas(col, image, 0, w)
    h = image.shape[0]
    for i, (seg_mask, obj_ref) in enumerate(zip(seg_masks, novel_obj_refs)):
        img = seg_mask.visualize()
        draw_on_canvas(col, img, h, w)
        h += img.shape[0]
        if h > col.shape[0] - img.shape[0]:
            break
    return col


def visualize_seg_masks_sequence(images, seg_masks, novel_obj_refs):
    sequence_viz = np.zeros((400, VIZ_WIDTH, 3))
    num_frames = images.shape[0]
    assert images.shape[0] == len(seg_masks)
    # Figure out which 5 frames to draw
    num_draw_frames = 5
    if num_frames > num_draw_frames:
        frames_to_draw = [int(x * num_frames / num_draw_frames) for x in range(num_draw_frames)]
    else:
        frames_to_draw = list(range(num_frames))
        num_draw_frames = num_frames
    frame_width = int(VIZ_WIDTH / num_draw_frames)
    for i, frame in enumerate(frames_to_draw):
        frame_viz = visualize_seg_masks(images[frame], seg_masks[frame], novel_obj_refs, frame_width)
        draw_on_canvas(sequence_viz, frame_viz, 0, i * frame_width)
    return sequence_viz


def visualize_object_database(object_images, object_descriptions, seg_masks):
    nod_html = ""
    nod_res = {}
    # Loop through all novel objects in the dataset and create a row
    first_masks = seg_masks[0]
    mid_masks = seg_masks[int(len(seg_masks) / 2)]
    for i, (obj_imgs, obj_refs, first_mask, mid_mask) in enumerate(zip(object_images, object_descriptions, first_masks, mid_masks)):
        row_html = ""
        # First create the object label
        row_html += tplt.fill(tplt.HTML_NOD_ROW_LABEL_TEMPLATE, "LABEL", letters[i])
        # Then add in all the query images
        for o, obj_img in enumerate(obj_imgs):
            obj_name = f"nod-row-{i}-query-{o}.png"
            row_html += tplt.fill(tplt.HTML_NOD_ROW_QUERY_BLOCK_TEMPLATE, "SRC_IMG", obj_name)
            nod_res[obj_name] = obj_img
        # Then add in all the object names
        row_html += tplt.multi_fill(tplt.HTML_NOD_ROW_NAMES_TEMPLATE, list(range(1, len(obj_refs) + 1)), obj_refs)

        # Finally add in the recognition results
        # .. in first frame
        first_mask_fname = f"nod-row-{i}-first-mask.png"
        row_html += tplt.multi_fill(tplt.HTML_NOD_ROW_MASK_TEMPLATE, ["IMG_SRC", "IMG_CAPTION"],
                                    [first_mask_fname, "First Mask"])
        nod_res[first_mask_fname] = first_mask

        # .. in middle frame
        #mid_mask_fname = f"nod-row-{i}-mid-mask.png"
        #row_html += tplt.multi_fill(tplt.HTML_NOD_ROW_MASK_TEMPLATE, ["IMG_SRC", "IMG_CAPTION"],
        #                            [mid_mask_fname, "Mid Mask"])
        #nod_res[mid_mask_fname] = mid_mask

        nod_html += tplt.fill(tplt.HTML_NOD_ROW_TEMPLATE, "CONTENTS", row_html)
    return nod_html, nod_res


def visualize_top_down_view(env_id):
    fname = f"overhead-view-{env_id}.png"
    tdimg = load_env_img(env_id, flipdiag=True)
    html = tplt.fill(tplt.HTML_OVERHEAD_VIEW_TEMPLATE, "IMG_SRC", fname)
    res = {fname: tdimg}
    return html, res


def select_timesteps_evenly(sequence, max_timesteps):
    seq_len = len(sequence)
    if seq_len > max_timesteps:
        timesteps = [int(x * seq_len / max_timesteps) for x in range(max_timesteps)]
    else:
        timesteps = list(range(seq_len))
    new_sequence = [sequence[x] for x in timesteps]
    return new_sequence, timesteps


def visualize_visual_trajectory(name, label, image_tensor, max_images=10):
    html = tplt.HTML_VISUAL_TRAJECTORY_TEMPLATE
    res = {}
    image_sequence, timesteps = select_timesteps_evenly(image_tensor, max_images)
    blocks_html = ""
    for timestep, image in zip(timesteps, image_sequence):
        img_fname = f"{name}-{timestep}.png"
        res[img_fname] = image
        blocks_html += tplt.multi_fill(tplt.HTML_VISUAL_TRAJECTORY_BLOCK_TEMPLATE,
                                       ["IMG_SRC", "CAPTION"], [img_fname, f"Timestep {timestep}"])
    html = tplt.multi_fill(html, ["BLOCKS", "LABEL"], [blocks_html, label])
    return html, res


def visualize_faux_top_down_trajectory(env_id, sequence_length=10):
    fname = f"overhead-view-{env_id}.png"
    html = tplt.HTML_VISUAL_TRAJECTORY_TEMPLATE
    blocks_html = ""
    for timestep in range(sequence_length):
        blocks_html += tplt.multi_fill(tplt.HTML_VISUAL_TRAJECTORY_BLOCK_TEMPLATE,
                                       ["IMG_SRC", "CAPTION"], [fname, f"Timestep {timestep}"])
    html = tplt.multi_fill(html, ["BLOCKS", "LABEL"], [blocks_html, "Overhead View"])
    return html


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
    index_html = "<html><body><ul>"
    for subdir in all_subdirs:
        index_html += f"<li><a href={subdir}/0dashboard.html>{subdir}</a></li>"
    index_html += "</ul></body></html>"
    fname = os.path.join(run_dir, "index.html")
    with open(fname, "w") as fp:
        fp.write(index_html)


def visualize_model_dashboard(batch, tensor_store, iteration, run_name):
    # -------------------------------- EXTRACT DATA --------------------------------------------
    canvas = np.zeros((VIZ_HEIGHT, VIZ_WIDTH, 3))
    instruction_nl = batch["instr_nl"][0][0]
    anon_instruction_nl = batch["anon_instr_nl"][0]
    images = batch["images"][0].cpu().numpy().transpose((0, 2, 3, 1))
    obj_refs = batch["obj_ref"][0]
    noun_chunks = batch["noun_chunks"][0]
    env_id = batch["md"][0][0]["env_id"]
    similarity_matrix = batch["similarity_matrix"][0]

    # FPV view segmentation masks for each object in the novel object dataset
    seg_masks_inner = tensor_store.get_inputs_batch("obj_masks_fpv_inner")[:, 0, :, :, :]
    seg_masks_outer = tensor_store.get_inputs_batch("obj_masks_fpv_outer")[:, 0, :]
    seg_masks = [Partial2DDistribution(inner, outer) for inner, outer in zip(seg_masks_inner, seg_masks_outer)]

    # FPV segmentation masks per each highly scored object
    seg_masks_per_object = Partial2DDistribution.stack(seg_masks)
    seg_masks_per_object.inner_distribution.permute((1, 0, 2, 3, ))
    seg_masks_per_object.outer_prob_mass.permute((1, 0))
    if similarity_matrix.shape[0] == 0:
        important_object_indices = []
    else:
        important_object_indices = torch.argmax(similarity_matrix, dim=1)
        important_object_indices = list(sorted(set([x.item() for x in important_object_indices])))
    important_object_letters = [letters[i] for i in important_object_indices]
    important_object_seg_masks = [seg_masks_per_object[:, i] for i in important_object_indices]
    important_object_viz_masks = [[m.visualize() for m in mask] for mask in important_object_seg_masks]

    # Novel object dataset
    novel_obj_refs = batch["object_database"][0]["object_references"]
    novel_obj_imgs = batch["object_database"][0]["object_images"]

    # Object reference top-down masks - fit into 3 channels for visualization
    object_reference_masks_w = tensor_store.get_inputs_batch("object_reference_masks_w")[:, 0]
    img_shape = list(object_reference_masks_w.shape)
    img_shape[1] = 3
    viz_ref_masks = torch.zeros(img_shape)
    c = object_reference_masks_w.shape[1]
    viz_ref_masks[:, :c, :, :] = object_reference_masks_w[:, :3, :, :]

    # Coerage masks
    cov_mask_img = tensor_store.get_inputs_batch("accum_M_w")[:, 0, :, :, :]

    # Visitation distributions
    log_vdist_w = tensor_store.get_inputs_batch("log_v_dist_w")[0]
    vdist_w = log_vdist_w.softmax()
    vdist_w_img = [v.visualize() for v in vdist_w]

    # -------------------------------------------------------------------------------------------
    root_dir = os.path.expanduser(f"~/dashboard_test_dir/{run_name}/{iteration}/")
    run_dir = os.path.expanduser(f"~/dashboard_test_dir/{run_name}/")
    os.makedirs(root_dir, exist_ok=True)
    resources = {}
    html = tplt.HTML_TEMPLATE
    # ---------------------------------- HTML LAYOUT --------------------------------------------

    # Layout novel object dtaset
    nod_html, nod_res = visualize_object_database(novel_obj_imgs, novel_obj_refs, seg_masks)
    html = tplt.fill(html, "NOD", nod_html)
    resources = dict_merge(resources, nod_res)

    # Layout instruction with object references
    instr_html = visualize_instruction_and_obj_refs(instruction_nl, obj_refs, noun_chunks)
    html = tplt.fill(html, "INSTR", instr_html)

    # Layout similarity matrix
    similarity_html, similarity_res = visualize_similarity_matrix(similarity_matrix, obj_refs, letters[:len(novel_obj_refs)])
    html = tplt.fill(html, "SIMILARITY_MATRIX", similarity_html)
    resources = dict_merge(resources, similarity_res)

    # Layout overhead view
    td_html, td_res = visualize_top_down_view(env_id)
    html = tplt.fill(html, "OVERHEAD_VIEW", td_html)
    resources = dict_merge(resources, td_res)

    # Layout image sequence
    img_html, img_res = visualize_visual_trajectory("fpv-image", "First-Person View", images)
    html = tplt.fill(html, "FPV_TRAJECTORY", img_html)
    resources = dict_merge(resources, img_res)

    # Layout important object masks
    obj_seg_html = ""
    for letter, viz_mask in zip(important_object_letters, important_object_viz_masks):
        img_html, img_res = visualize_visual_trajectory(f"object-{letter}-mask", f"Object {letter} Recognition", viz_mask)
        obj_seg_html += img_html
        resources = dict_merge(resources, img_res)
    html = tplt.fill(html, "OBJ_TRAJECTORIES", obj_seg_html)

    img_html = visualize_faux_top_down_trajectory(env_id)
    html = tplt.fill(html, "TOP_DOWN_TRAJECTORY", img_html)

    # Layout maps in top-down view
    img_html, img_res = visualize_visual_trajectory("obj-ref-mask", "Object Reference Masks", viz_ref_masks)
    html = tplt.fill(html, "MASK_TRAJECTORY", img_html)
    resources = dict_merge(resources, img_res)

    # Layout coverage masks in top-down view
    img_html, img_res = visualize_visual_trajectory("accum-mask", "Observability Mask", cov_mask_img)
    html = tplt.fill(html, "COVERAGE_TRAJECTORY", img_html)
    resources = dict_merge(resources, img_res)

    # Layout visitation distribution sequence
    img_html, img_res = visualize_visual_trajectory("visitation-dist", "Visitation Distributions", vdist_w_img)
    html = tplt.fill(html, "VISITATION_TRAJECTORY", img_html)
    resources = dict_merge(resources, img_res)

    export_resources(root_dir, resources)
    save_html(root_dir, html)
    update_index(run_dir)

    return canvas