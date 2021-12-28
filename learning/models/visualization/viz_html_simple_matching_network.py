import os
import imageio
import torch
import numpy as np

from data_io import paths
from visualization import Presenter
from learning.models.visualization import matching_template as tplt
from learning.models.visualization.viz_model_common import has_nowrite_flag

from learning.inputs.partial_2d_distribution import Partial2DDistribution


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
    all_subdirs = [dir for dir in os.listdir(run_dir) if os.path.isdir(dir)]
    all_subdirs = [int(subdir) for subdir in all_subdirs]
    index_html = "<html><body><ul>"
    for subdir in sorted(all_subdirs):
        index_html += f"<li><a href={subdir}/0dashboard.html>{subdir}</a></li>"
    index_html += "</ul></body></html>"
    fname = os.path.join(run_dir, "index.html")
    with open(fname, "w") as fp:
        fp.write(index_html)


def visualize_model_from_state(model_state, iteration, run_name):
    images_a = model_state.get("obj_a")
    images_b = model_state.get("objs_b")
    images_c = model_state.get("objs_c")
    scores_ab = model_state.get("scores_ab")
    scores_ac = model_state.get("scores_ac")
    count = model_state.get("total_count")
    mistakes = model_state.get("total_mistakes")
    object_totals = model_state.get("object_totals")
    object_mistake_counts = model_state.get("object_mistake_counts")
    object_mistake_object_counts = model_state.get("object_mistake_object_count")
    object_total_object_count = model_state.get("object_total_object_count")
    b_best = model_state.get("b_best")
    c_best = model_state.get("c_best")

    visualize_model(images_a=images_a,
                    images_b=images_b,
                    images_c=images_c,
                    scores_ab=scores_ab,
                    scores_ac=scores_ac,
                    cumulative_count=count,
                    cumulative_mistakes=mistakes,
                    object_totals=object_totals,
                    object_mistake_counts=object_mistake_counts,
                    object_mistake_object_counts=object_mistake_object_counts,
                    object_total_object_count=object_total_object_count,
                    iteration=iteration,
                    run_name=run_name,
                    multi=True,
                    b_best=b_best,
                    c_best=c_best)


def visualize_model(images_a, images_b, images_c,
                    scores_ab, scores_ac,
                    cumulative_count, cumulative_mistakes,
                    object_mistake_counts, object_totals, object_mistake_object_counts, object_total_object_count,
                    iteration, run_name, multi=False, b_best=None, c_best=None):
    # -------------------------------- EXTRACT DATA --------------------------------------------
    images_a = images_a.detach().cpu()
    images_b = images_b.detach().cpu()
    images_c = images_c.detach().cpu()
    # Create a collage
    if multi:
        img_w = images_b.shape[4]
        # dims: b, q, c, h, w
        images_b = [images_b[:, i] for i in range(images_b.shape[1])]
        # dims: b, c, h, w
        images_b = torch.cat(images_b, dim=3)
        images_c = [images_c[:, i] for i in range(images_c.shape[1])]
        images_c = torch.cat(images_c, dim=3)
    scores_ab = scores_ab.detach().cpu()
    scores_ac = scores_ac.detach().cpu()
    # -------------------------------------------------------------------------------------------
    run_dir = paths.get_run_dashboard_dir(run_name)
    root_dir = os.path.join(run_dir, f"{iteration}")
    rel_image_dir = "images"
    abs_image_dir = os.path.join(root_dir, rel_image_dir)
    os.makedirs(run_dir, exist_ok=True)
    if has_nowrite_flag(run_dir, default_flag=True):
        print(f"Skipping logging - disabled by nowrite.txt at: {run_dir}")
        return
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(abs_image_dir, exist_ok=True)
    resources = {}
    html = tplt.TEMPLATE_HTML
    correct_rows = ""
    wrong_rows = ""
    total_correct = 0
    total_wrong = 0
    total_count = 0
    # ---------------------------------- HTML LAYOUT --------------------------------------------
    # Layout correct and wrong images
    for i, (img_a, img_b, img_c, score_ab, score_ac) in enumerate(zip(images_a, images_b, images_c, scores_ab, scores_ac)):
        score_b = score_ab.item()
        score_c = score_ac.item()
        correct = score_b < score_c
        a_path, b_path, c_path = [os.path.join(rel_image_dir, f"{l}_{i}.png") for l in ["a", "b", "c"]]
        if multi:
            b_left = (img_w * b_best[i]).item()
            c_left = (img_w * c_best[i]).item()
            img_b[:, 0:2, b_left:b_left+img_w] = torch.tensor([img_b.max(), img_b.min(), img_b.min()])[:, np.newaxis, np.newaxis]
            img_c[:, 0:2, c_left:c_left+img_w] = torch.tensor([img_c.max(), img_c.min(), img_c.min()])[:, np.newaxis, np.newaxis]
        resources[a_path] = img_a
        resources[b_path] = img_b
        resources[c_path] = img_c
        row_html = tplt.multi_fill(tplt.ROW_HTML_TEMPLATE,
                                   ["SRC_A", "SRC_B", "SRC_C", "SCORE_B", "SCORE_C"],
                                   [a_path, b_path, c_path, score_b, score_c])
        if correct:
            correct_rows += row_html
            total_correct += 1
        else:
            wrong_rows += row_html
            total_wrong += 1
        total_count += 1

    accuracy = float(total_correct) / float(total_count + 1e-9)
    html = tplt.fill(html, "CORRECT_PREDICTIONS_ROWS", correct_rows)
    html = tplt.fill(html, "WRONG_PREDICTIONS_ROWS", wrong_rows)
    html = tplt.multi_fill(html, ["TOTAL_COUNT", "TOTAL_CORRECT", "TOTAL_WRONG", "ACCURACY"],
                                 [total_count, total_correct, total_wrong, accuracy])
    html = tplt.multi_fill(html, ["CUM_COUNT", "CUM_CORRECT", "CUM_MISTAKES", "CUM_ACCURACY"],
                                 [cumulative_count, cumulative_count - cumulative_mistakes, cumulative_mistakes, 1 - (float(cumulative_mistakes) / (cumulative_count + 1e-9))])

    # Layout per-object stats
    header_html = "".join([f"<th>{obj_id}</th>" for obj_id in sorted(object_totals.keys())])
    rows = ""
    for obj_id in sorted(object_totals.keys()):
        obj_total = object_totals[obj_id]
        obj_mistakes = object_mistake_counts.get(obj_id, 0)
        obj_acc = "{:.2f}".format(1 - float(obj_mistakes) / (obj_total + 1e-9))
        obj_per_obj_mistakes = object_mistake_object_counts.get(obj_id, {})
        obj_per_obj_count = object_total_object_count.get(obj_id, {})
        other_obj_rows_html = ""
        for other_obj_id in sorted(object_totals.keys()):
            if other_obj_id == obj_id:
                accuracy = "N/A"
            elif other_obj_id in obj_per_obj_mistakes:
                num_mistakes = obj_per_obj_mistakes[other_obj_id]
                num_total = obj_per_obj_count[other_obj_id]
                accuracy = "{:.2f}".format(1 - float(num_mistakes) / (num_total + 1e-9))
            else:
                accuracy = 1.0
            other_obj_rows_html += f"<td>{accuracy}</td>"
        obj_row_html = tplt.multi_fill(tplt.OBJECT_MISTAKE_ROW_TEMPLATE,
                                       ["OBJ_ID", "OBJ_ACC", "OTHER_OBJECT_ACC"],
                                       [obj_id, obj_acc, other_obj_rows_html])
        rows += obj_row_html
    html = tplt.fill(html, "OTHER_OBJECT_IDS", header_html)
    html = tplt.fill(html, "OBJECT_MISTAKE_ROWS", rows)
    #--------------------------------------------------------------------------------------------
    export_resources(root_dir, resources)
    save_html(root_dir, html)
    update_index(run_dir)
