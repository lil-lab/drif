import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import imageio
from utils.dict_tools import dict_merge
from data_io.env import load_env_img
from data_io import paths

from visualization import Presenter
from learning.models.visualization import rpn_dashboard_template as tplt
from learning.models.visualization.viz_model_common import has_nowrite_flag

from learning.inputs.partial_2d_distribution import Partial2DDistribution

from rollout_vizualizer import RolloutVisualizer


VIZ_WIDTH = 1000
VIZ_HEIGHT = 700

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
bbox_labels = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21"]


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


def build_similarity_matrix(similarity_matrix, y_labels, x_labels):
    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.matshow(similarity_matrix, interpolation='nearest')
    ax.grid(True)
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)
    fig.colorbar(cax)
    plot_img = fig2data(fig)
    return plot_img


def visualize_similarity_matrix(similarity_matrix, y_labels, x_labels):
    plot_img = build_similarity_matrix(similarity_matrix, y_labels, x_labels)
    sim_html = tplt.fill(tplt.HTML_SIMILARITY_MATRIX_TEMPLATE, "IMG_SRC", "similarity_matrix.png")
    resources = {"similarity_matrix.png": plot_img}
    return sim_html, resources


def draw_on_canvas(canvas, img, h, w):
    canvas[h:h+img.shape[0], w:w+img.shape[1], :] = img


def visualize_seg_masks(image, seg_masks, object_descriptions, frame_width):
    col = np.zeros((400, frame_width, 3))
    w = int((frame_width - image.shape[1]) / 2)
    draw_on_canvas(col, image, 0, w)
    h = image.shape[0]
    for i, (seg_mask, obj_ref) in enumerate(zip(seg_masks, object_descriptions)):
        img = seg_mask.visualize()
        draw_on_canvas(col, img, h, w)
        h += img.shape[0]
        if h > col.shape[0] - img.shape[0]:
            break
    return col


def visualize_seg_masks_sequence(images, seg_masks, object_descriptions):
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
        frame_viz = visualize_seg_masks(images[frame], seg_masks[frame], object_descriptions, frame_width)
        draw_on_canvas(sequence_viz, frame_viz, 0, i * frame_width)
    return sequence_viz


def visualize_object_database(object_images, object_descriptions):
    nod_html = ""
    nod_res = {}
    for i, (obj_imgs, obj_refs) in enumerate(zip(object_images, object_descriptions)):
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
        nod_html += tplt.fill(tplt.HTML_NOD_ROW_TEMPLATE, "CONTENTS", row_html)
    return nod_html, nod_res


def visualize_top_down_view(env_id, top_down_frames=None):
    if top_down_frames:
        fnames = [f"overhead-view-{env_id}-f-{i}.png" for i in range(len(top_down_frames))]
        html = tplt.fill(tplt.HTML_OVERHEAD_VIEW_TEMPLATE, "IMG_SRC", fnames[-1])
        res = {fname: frame for fname, frame in zip(fnames, top_down_frames)}
        res[f"overhead-view-{env_id}.png"] = top_down_frames[-1]
    else:
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
    return timesteps


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
    if "nonorm" in resource_dict:
        nonorm = resource_dict["nonorm"]
        del resource_dict["nonorm"]
    else:
        nonorm = []
    for n, img in resource_dict.items():
        fpath = os.path.join(root_dir, n)
        if isinstance(img, Partial2DDistribution):
            img = img.visualize()
        dontnormthis = n in nonorm
        img_save = Presenter().prep_image(img, no_norm=dontnormthis)
        imageio.imsave(fpath, img_save)


def save_html(root_dir, html_string):
    fname = os.path.join(root_dir, "0dashboard.html")
    with open(fname, "w") as fp:
        fp.write(html_string)


def update_index(run_dir):
    all_subdirs = [dir for dir in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, dir))]
    index_html = "<html><body><ul>"
    for subdir in sorted(all_subdirs):
        index_html += f"<li><a href={subdir}/0dashboard.html>{subdir}</a></li>"
    index_html += "</ul></body></html>"
    fname = os.path.join(run_dir, "index.html")
    with open(fname, "w") as fp:
        fp.write(index_html)


def update_base_index(base_dir):
    all_subdirs = [dir for dir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, dir))]
    index_html = "<html><body><ul>"
    for subdir in sorted(all_subdirs):
        index_html += f"<li><a href={subdir}/index.html>{subdir}</a></li>"
    index_html += "</ul></body></html>"
    fname = os.path.join(base_dir, "index.html")
    with open(fname, "w") as fp:
        fp.write(index_html)


object_colors = torch.from_numpy(np.asarray([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 0.5, 0.0],
    [0.5, 1.0, 0.0],
    [0.0, 1.0, 0.5],
    [0.0, 0.5, 1.0],
    [1.0, 0.0, 0.5],
    [0.5, 0.0, 1.0]
]))


def visualize_rpn_trajectory(resources, fpv_images, all_region_crops, visual_sim_matrices,
                             grounding_matrices, obj_refs, full_masks_w, full_masks_fpv,
                             vdist, env_id, intrinsic_rewards=None, max_steps=12):
    timesteps = select_timesteps_evenly(fpv_images, max_steps)
    rpn_trajectory_steps_html = ""

    for t in timesteps:
        step_html = tplt.HTML_RPN_STEP_TEMPLATE
        step_html = tplt.fill(step_html, "TIMESTEP_ID", str(t))
        fpv_image = fpv_images[t]
        if t >= len(all_region_crops):
            break
        t_region_grops = all_region_crops[t]
        visual_sim_matrix = visual_sim_matrices[t].detach().cpu().transpose(0, 1) # Have this as nod x bounding boxes
        grounding_matrix = grounding_matrices[t].detach().cpu().transpose(0, 1) # Have them be obj_refs x bounding boxes
        num_bboxes = grounding_matrix.shape[1]
        num_objects = visual_sim_matrix.shape[0]
        # Unpack FPV masks
        full_mask_fpv = full_masks_fpv[t][0].detach().cpu()
        all_obj_masks_fpv = full_mask_fpv[0:1]
        obj_ref_masks_fpv = full_mask_fpv[1:]

        # Unpack full masks:
        all_obj_mask_w = full_masks_w[t][0:1]
        obj_ref_mask_w = full_masks_w[t][1:]

        # Visualize first-person image
        fpv_img_src = f"rpn-{t}-fpv-image.png"
        step_html = tplt.fill(step_html, "FPV_IMG_SRC", fpv_img_src)
        resources[fpv_img_src] = fpv_image

        # Visualize proposed regions
        crops_html = ""
        for i, crop in enumerate(t_region_grops):
            label = bbox_labels[i]
            fname = f"rpn-{t}-crop-{label}.png"
            crop_html = tplt.multi_fill(tplt.HTML_RPN_REGION_CROP_TEMPLATE, ["IMG_SRC", "REGION_LABEL"], [fname, label])
            resources[fname] = crop
            crops_html += crop_html
        step_html = tplt.fill(step_html, "REGION_CROPS", crops_html)

        # Visualize visual similarity matrix
        visual_matrix_img = build_similarity_matrix(visual_sim_matrix, letters[:num_objects], bbox_labels[:num_bboxes])
        resources[f"rpn-{t}-visual-similarity.png"] = visual_matrix_img
        step_html = tplt.fill(step_html, "VISUAL_SIM_MATRIX_SRC", f"rpn-{t}-visual-similarity.png")

        # Visualize grounding matrix
        grounding_matrix_img = build_similarity_matrix(grounding_matrix, obj_refs, bbox_labels[:num_bboxes])
        resources[f"rpn-{t}-grounding-matrix.png"] = grounding_matrix_img
        step_html = tplt.fill(step_html, "GROUNDING_MATRIX_SRC", f"rpn-{t}-grounding-matrix.png")

        # Visualize object reference masks
        obj_ref_mask_path = f"rpn-{t}-obj-ref-mask-w.png"
        resources[obj_ref_mask_path] = obj_ref_mask_w
        step_html = tplt.fill(step_html, "OBJECT_MAP_SRC", obj_ref_mask_path)

        # Visualize visitation distributions
        vdist_t = vdist[t]
        vdist_path = f"rpn-{t}-vdist-w.png"
        resources[vdist_path] = vdist_t
        step_html = tplt.fill(step_html, "VDIST_SRC", vdist_path)

        # Visualize top-down view
        if f"overhead-view-{env_id}-f-{t}.png" in resources:
            step_html = tplt.fill(step_html, "TOP_DOWN_SRC", f"overhead-view-{env_id}-f-{t}.png")
        else:
            step_html = tplt.fill(step_html, "TOP_DOWN_SRC", f"overhead-view-{env_id}.png")

        # Visualize object reference FPV masks:
        ref_masks_html = ""

        # Color object FPV masks in channel order
        for i, (obj_ref, obj_ref_mask_fpv) in enumerate(zip(obj_refs, obj_ref_masks_fpv)):
            fname = f"rpn-{t}-ref-fpv-mask-{i}-{obj_ref}.png"
            ref_mask_html = tplt.multi_fill(tplt.HTML_RPN_REGION_FPV_MASK_TEMPLATE, ["IMG_SRC", "LABEL"], [fname, obj_ref])
            ref_masks_html += ref_mask_html
            color = object_colors[i]
            resources[fname] = obj_ref_mask_fpv[np.newaxis, :, :] * color[:, np.newaxis, np.newaxis]
            if "nonorm" not in resources:
                resources["nonorm"] = []
            resources["nonorm"].append(fname)
        step_html = tplt.fill(step_html, "REGION_FPV_MASKS", ref_masks_html)

        # Visualize all object mask - FPV:
        fpath = f"rpn-{t}-all-obj-mask-fpv.png"
        step_html = tplt.fill(step_html, "ALL_OBJ_FPV_SRC", fpath)
        resources[fpath] = all_obj_masks_fpv
        # Visualize all object mask - W:
        fpath = f"rpn-{t}-all-obj-mask-w.png"
        step_html = tplt.fill(step_html, "ALL_OBJ_W_SRC", fpath)
        resources[fpath] = all_obj_mask_w

        # Visualize intrinsic rewards:
        if intrinsic_rewards:
            intrinsic_rewards_t = intrinsic_rewards[t]
            step_html = tplt.add_table(step_html, "INTRINSIC_REWARDS", intrinsic_rewards_t)

        rpn_trajectory_steps_html += step_html
    return rpn_trajectory_steps_html, resources


def visualize_model_dashboard_during_training(batch, tensor_store, iteration, run_name):

    # -------------------------------- EXTRACT DATA --------------------------------------------
    argbomb = {}
    argbomb["instruction_nl"] = batch["instr_nl"][0][0]
    #argbomb["anon_instruction_nl"] = batch["anon_instr_nl"][0]
    argbomb["images"] = batch["images"][0].cpu().numpy().transpose((0, 2, 3, 1))
    argbomb["object_references"] = batch["object_references"][0]
    argbomb["noun_chunks"] = batch["noun_chunks"][0]
    argbomb["env_id"] = batch["md"][0][0]["env_id"]
    argbomb["text_similarity_matrix"] = batch["text_similarity_matrix"][0]
    argbomb["visual_similarity_matrix"] = tensor_store.get_inputs_batch("visual_similarity_matrix", raw=True)
    argbomb["grounding_matrix"] = tensor_store.get_inputs_batch("grounding_matrix", raw=True)
    argbomb["all_region_crops"] = tensor_store.get_inputs_batch("region_crops", raw=True)
    argbomb["full_masks_fpv"] = tensor_store.get_inputs_batch("full_masks_fpv")
    # Novel object dataset
    argbomb["object_descriptions"] = batch["object_database"][0]["object_references"]
    argbomb["object_images"] = batch["object_database"][0]["object_images"]
    # Object reference top-down masks - fit into 3 channels for visualization
    argbomb["seg_masks_w"] = tensor_store.get_inputs_batch("accum_seg_masks_w")[:, 0]
    argbomb["log_vdist_w"] = tensor_store.get_inputs_batch("log_v_dist_w")[0]
    argbomb["default_nowrite"] = True

    visualize_model_dashboard(run_name,
                              rollout_name=iteration,
                              **argbomb)


def visualize_model_dashboard_from_rollout(rollout, run_name):
    argbomb = {}

    if len(rollout) == 0:
        return

    # First extract images and external inputs from the rollout
    env_id = rollout[0]['env_id']
    seg_idx = rollout[0]['seg_idx']
    images = np.stack([r["state"].image for r in rollout])        # Tx96x128x3 numpy.ndarray
    instruction_nl = rollout[0]['instruction']          # string

    argbomb['env_id'] = env_id
    argbomb['run_name'] = run_name
    argbomb['rollout_name'] = f"{env_id}-{seg_idx}"
    argbomb["instruction_nl"] = instruction_nl
    #argbomb["anon_instruction_nl"] = viz_data[0]["anon_instruction_nl"]
    argbomb["images"] = images
    argbomb["intrinsic_rewards"] = [r["intrinsic_rewards"] for r in rollout]

    # Then extract internal state data from the model_state
    final_model_state = rollout[-1]["model_state"]
    tensor_store = final_model_state.tensor_store

    # Stuff directly stored in model state (stuff that doesn't depend on timestep)
    argbomb["object_references"] = final_model_state.get("object_references")[0]
    argbomb["noun_chunks"] = final_model_state.get("noun_chunks")[0]
    argbomb["text_similarity_matrix"] = final_model_state.get("text_similarity_matrix")
    nod = final_model_state.get("object_database")
    argbomb["object_descriptions"] = nod["object_references"]
    argbomb["object_images"] = nod["object_images"]

    # Stuff that depends on timestep is stored in the tensor store instead
    argbomb["visual_similarity_matrix"] = tensor_store.get("visual_similarity_matrix")
    argbomb["grounding_matrix"] = tensor_store.get("grounding_matrix")
    argbomb["all_region_crops"] = tensor_store.get("region_crops")
    argbomb["full_masks_fpv"] = tensor_store.get("full_masks_fpv")
    argbomb["seg_masks_w"] = tensor_store.get_inputs_batch("accum_seg_masks_w", cat_not_stack=True)
    argbomb["log_vdist_w"] = Partial2DDistribution.stack([x[0] for x in tensor_store.get("log_v_dist_w")])

    # Plot drone trajectory:
    viz_params = {
        "draw_topdown": True,
        "draw_trajectory": True,
        "draw_drone": True
    }
    viz = RolloutVisualizer(resolution=400)
    frames = viz.top_down_visualization(env_id, seg_idx, rollout, "sim", viz_params)
    frames = [np.flipud(np.fliplr(np.rot90(f))) for f in frames]
    argbomb["top_down_frames"] = frames

    argbomb["default_nowrite"] = False
    visualize_model_dashboard(**argbomb)


def visualize_model_dashboard(
        run_name,
        rollout_name,
        env_id,
        images,
        visual_similarity_matrix,
        all_region_crops,
        grounding_matrix,
        full_masks_fpv,
        object_images,
        object_descriptions,
        seg_masks_w,
        log_vdist_w,
        instruction_nl,
        object_references,
        noun_chunks,
        text_similarity_matrix,
        intrinsic_rewards=None,
        top_down_frames=None,
        default_nowrite=True):

    # -------------------------------------------------------------------------------------------
    base_dashboard_dir = paths.get_dashboard_dir()
    run_dir = paths.get_run_dashboard_dir(run_name)
    root_dir = os.path.join(run_dir, str(rollout_name))
    resources = {}
    html = tplt.HTML_TEMPLATE
    os.makedirs(run_dir, exist_ok=True)
    if has_nowrite_flag(run_dir, default_flag=default_nowrite):
        print(f"Skipping logging - disabled by nowrite.txt at: {run_dir}")
        return
    os.makedirs(root_dir, exist_ok=True)

    print("Writing visualization dashboard to: ", root_dir)

    # Data prep:
    # Object segmentation masks
    img_shape = list(seg_masks_w.shape)
    img_shape[1] = 3
    viz_full_masks_w = torch.zeros(img_shape)
    c = seg_masks_w.shape[1]
    viz_full_masks_w[:, :c, :, :] = seg_masks_w[:, :3, :, :]

    # Visitation distributions
    vdist_w = log_vdist_w.softmax()
    vdist_w_img = [v.visualize() for v in vdist_w]

    # Layout novel object dtaset
    nod_html, nod_res = visualize_object_database(object_images, object_descriptions)
    html = tplt.fill(html, "NOD", nod_html)
    resources = dict_merge(resources, nod_res)

    # Layout instruction with object references
    instr_html = visualize_instruction_and_obj_refs(instruction_nl, object_references, noun_chunks)
    html = tplt.fill(html, "INSTR", instr_html)

    # Layout text similarity matrix
    text_similarity_html, text_similarity_res = visualize_similarity_matrix(text_similarity_matrix, object_references, letters[:len(object_descriptions)])
    html = tplt.fill(html, "TEXT_SIMILARITY_MATRIX", text_similarity_html)
    resources = dict_merge(resources, text_similarity_res)

    # Layout grounding matrix
    # TODO: Adjust labels
    # TODO: Incorporate notion of a "set of regions"

    # Layout overhead view
    td_html, td_res = visualize_top_down_view(env_id, top_down_frames)
    html = tplt.fill(html, "OVERHEAD_VIEW", td_html)
    resources = dict_merge(resources, td_res)

    # Add intrinsic returns
    if intrinsic_rewards:
        keys = list(intrinsic_rewards[0].keys())
        intrinsic_returns = {k: sum([x[k] for x in intrinsic_rewards]) for k in keys}
        html = tplt.add_table(html, "INTRINSIC_REWARDS", intrinsic_returns)

    # Layout RPN reasoning step by step:
    rpn_trajectory_html, rpn_res = visualize_rpn_trajectory(
        resources=resources,
        fpv_images=images,
        visual_sim_matrices=visual_similarity_matrix,
        all_region_crops=all_region_crops,
        grounding_matrices=grounding_matrix,
        obj_refs=object_references,
        full_masks_w=viz_full_masks_w,
        full_masks_fpv=full_masks_fpv,
        vdist=vdist_w_img,
        env_id=env_id,
        intrinsic_rewards=intrinsic_rewards)
    html = tplt.fill(html, "RPN_TRAJECTORY_STEPS", rpn_trajectory_html)
    resources = dict_merge(resources, rpn_res)

    export_resources(root_dir, resources)
    save_html(root_dir, html)
    update_index(run_dir)
    update_base_index(base_dashboard_dir)
