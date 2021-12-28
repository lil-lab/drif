import numpy as np
import cv2
from copy import deepcopy
import torch
import os
import imageio
from scipy.ndimage import sobel, gaussian_filter

from data_io.env import load_env_img
from data_io import paths

from visualization import Presenter

from learning.inputs.partial_2d_distribution import Partial2DDistribution

from rollout_vizualizer import RolloutVisualizer


VIZ_WIDTH = 1000
VIZ_HEIGHT = 700

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
bbox_labels = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21"]

viz_params = {
    "draw_landmarks": False,
    "draw_topdown": True,
    "draw_drone": True,
    "draw_trajectory": True,
    "draw_fov": False,
    "include_vdist": False,
    "include_layer": None,
    "include_instr": False
}


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


def export_media(media_dir_vids, media_dir_frames, media_dict):
    p = Presenter()
    for n, frames in media_dict.items():
        fpath_vids = os.path.join(media_dir_vids, n)
        fpath_frames = os.path.join(media_dir_frames, n)
        if isinstance(frames, str):
            with open(f"{fpath_vids}.txt", "w") as fp:
                fp.write(frames)
            with open(f"{fpath_frames}.txt", "w") as fp:
                fp.write(frames)
        else:
            frames = [p.prep_image(img, no_norm=True) for img in frames]
            save_frames(frames, fpath_vids, fpath_frames, fps=5.0, start_lag=1.0, end_lag=1.0)


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

def generate_color_object_mask(grayscale_masks, use_color_index=-1):
    b, n, h, w = grayscale_masks.shape
    np_object_colors = object_colors.detach().cpu().numpy()
    p = Presenter()
    masks = [p.prep_image(grayscale_masks[:, i], no_norm=True) for i in range(n)]

    # If color index is specified, use that. Otherwise order accordingly
    if use_color_index != -1:
        colors = [np_object_colors[use_color_index] for i in range(n)]
    else:
        colors = [np_object_colors[i] for i in range(n)]

    col_masks = [m[:, :, None] * c[None, None, :] for m,c in zip(masks, colors)]
    col_mask = np.asarray(col_masks).sum(0)
    return col_mask

def mark_boundary(obj_masks):
    p = Presenter()
    if obj_masks.shape[1] == 0:
        return np.zeros((0, obj_masks.shape[2], obj_masks.shape[3]))
    out_masks = []
    for i in range(obj_masks.shape[1]):
        t_mask = obj_masks[:, i:i+1, :, :]
        np_mask = p.prep_image(t_mask, no_norm=True)
        edge_x = sobel(np_mask, axis=0)
        edge_y = sobel(np_mask, axis=1)
        edge = abs(edge_x) + abs(edge_y)
        edge = edge.clip(0, 1)

        # Zero out low-gradient areas
        edge[edge < 0.3] = 0.0

        # Zero out object internal areas
        non_obj = np_mask < 0.2
        edge = edge * non_obj

        # Blur and threshold the edge image
        edge = gaussian_filter(edge, 1)

        out_masks.append(edge)
    out_masks = np.stack(out_masks, axis=0)
    return out_masks

def make_fpv_masks(outputs, rollout):
    p = Presenter()
    model_state = rollout[-1]['model_state']
    object_refs = model_state.get("object_references")[0]

    # Extract RGB background
    rgb_images = [s['state'].image for s in rollout]

    # Extract FPV overlay masks
    object_reference_masks_fpv = model_state.tensor_store.get('object_reference_masks_fpv')
    object_masks_fpv_color = [generate_color_object_mask(f) for f in object_reference_masks_fpv]

    # Create object boundary images
    object_boundary_masks_fpv = [torch.from_numpy(mark_boundary(f)) for f in object_reference_masks_fpv]
    object_boundary_masks_fpv_color = [generate_color_object_mask(f[np.newaxis, :, :, :]) for f in object_boundary_masks_fpv]
    overlaid_boundary_frames = [p.overlaid_image(r, m) for r, m in zip(rgb_images, object_boundary_masks_fpv_color)]
    outputs["object_boundaries_fpv"] = object_boundary_masks_fpv_color
    outputs["object_boundaries_fpv_over_rgb"] = overlaid_boundary_frames
    # Boundary masks for each object separately:
    for i in range(len(object_refs)):
        # Take the slice corresponding to the specific object reference
        objref = object_refs[i]
        obj_masks = [m[i:i+1, :, :] for m in object_boundary_masks_fpv]
        color_obj_mask = [generate_color_object_mask(m.unsqueeze(0), i) for m in obj_masks]
        color_obj_mask_over_rgb = [p.overlaid_image(r, m) for r, m in zip(rgb_images, color_obj_mask)]
        outputs[f"object_boundaries_fpv/{objref}"] = color_obj_mask
        outputs[f"object_boundaries_fpv_over_rgb/{objref}"] = color_obj_mask_over_rgb

    overlaid_frames = [p.overlaid_image(r, m) for r, m in zip(rgb_images, object_masks_fpv_color)]
    outputs["object_masks_fpv"] = object_masks_fpv_color
    outputs["object_masks_fpv_over_rgb"] = overlaid_frames

    # Masks for each object separately:
    for i in range(len(object_refs)):
        # Take the slice corresponding to the specific object reference
        objref = object_refs[i]
        obj_masks = [m[:, i:i+1] for m in object_reference_masks_fpv]
        color_obj_mask = [generate_color_object_mask(m, i) for m in obj_masks]
        color_obj_mask_over_rgb = [p.overlaid_image(r, m) for r, m in zip(rgb_images, color_obj_mask)]
        outputs[f"objects_fpv/{objref}"] = color_obj_mask
        outputs[f"objects_fpv_over_rgb/{objref}"] = color_obj_mask_over_rgb


def make_allo_masks(outputs, rollout, env_img):
    model_state = rollout[0]["model_state"]
    object_refs = model_state.get("object_references")[0]
    p = Presenter()

    # Object Masks - Non-accumulated
    seg_masks_w = model_state.tensor_store.get("seg_masks_w")
    # Notice the difference in indexing:
    seg_all_obj_masks_w = [m[0, 0:1] for m in seg_masks_w]
    seg_obj_masks_w = [m[:, 1:] for m in seg_masks_w]

    seg_color_obj_masks_w = [generate_color_object_mask(m) for m in seg_obj_masks_w]
    seg_color_obj_masks_w_over_rgb = [p.overlaid_image(env_img, f) for f in seg_color_obj_masks_w]
    outputs["object_masks_w"] = seg_color_obj_masks_w
    outputs["object_masks_w_over_rgb"] = seg_color_obj_masks_w_over_rgb

    # Object Masks - Accumulated
    accum_seg_masks_w = model_state.tensor_store.get("accum_seg_masks_w")
    # Notice the difference in indexing:
    accum_all_obj_masks_w = [m[0, 0:1] for m in accum_seg_masks_w]
    accum_obj_masks_w = [m[:, 1:] for m in accum_seg_masks_w]

    accum_color_obj_masks_w = [generate_color_object_mask(m) for m in accum_obj_masks_w]
    accum_color_obj_masks_w_over_rgb = [p.overlaid_image(env_img, f) for f in accum_color_obj_masks_w]
    outputs["accum_object_masks_w"] = accum_color_obj_masks_w
    outputs["accum_object_masks_w_over_rgb"] = accum_color_obj_masks_w_over_rgb
    #outputs["all_obj_masks_w"] = accum_all_obj_masks_w

    # Masks for each object separately:
    for i in range(len(object_refs)):
        # Take the slice corresponding to the specific object reference
        objref = object_refs[i]
        obj_masks = [m[:, i:i+1] for m in accum_obj_masks_w]
        color_obj_mask = [generate_color_object_mask(m, i) for m in obj_masks]
        color_obj_mask_over_rgb = [p.overlaid_image(env_img, f) for f in color_obj_mask]
        outputs[f"accum_object_masks_w/{objref}"] = color_obj_mask
        outputs[f"accum_objects_masks_w_over_rgb/{objref}"] = color_obj_mask_over_rgb

    # Observability masks:
    accum_obs_masks_w = model_state.tensor_store.get_inputs_batch("accum_obs_masks_w")[:, 0]
    accum_obs_mask_w_over_rgb = [p.overlaid_image(env_img, f, interpolation=cv2.INTER_LINEAR) for f in accum_obs_masks_w]
    emptybg = np.zeros_like(env_img)
    accum_obs_masks_w = [p.overlaid_image(emptybg, f, strength=1.0, interpolation=cv2.INTER_LINEAR) for f in accum_obs_masks_w]
    outputs["accum_obs_masks_w"] = [m for m in accum_obs_masks_w]
    outputs["accum_obs_mask_w_over_rgb"] = [m for m in accum_obs_mask_w_over_rgb]


def save_frames(frames, base_path_videos, base_path_frames, fps=5.0, start_lag=0.0, end_lag=0.0):
    p = Presenter()
    print(f"Saving files: {base_path_videos}, {base_path_frames}")
    if start_lag > 0:
        frames = [frames[0]] * int(fps * start_lag) + frames
    if end_lag > 0:
        frames = frames + [frames[-1]] * int(fps * end_lag)
    #viz.presenter.save_gif(frames, f"{base_path}.gif", fps=5.0)
    #viz.presenter.save_video(frames, f"{base_path}.mp4", fps=fps)
    #viz.presenter.save_frames(frames, f"{base_path}-frames")
    dirname_vids = os.path.dirname(base_path_videos)
    os.makedirs(dirname_vids, exist_ok=True)
    dirname_frames = os.path.dirname(base_path_frames)
    os.makedirs(dirname_frames, exist_ok=True)
    p.save_video(frames, f"{base_path_videos}.mp4", fps=fps)
    # TODO: Uncomment to enable saving frames
    #p.save_frames(frames, f"{base_path_frames}-frames")


def corl2019_visualizations(outputs, rollout, gt_distr, domain):
    viz = RolloutVisualizer(resolution=576)

    env_id = rollout[0]['env_id']
    seg_idx = rollout[0]['seg_idx']

    mod_params = deepcopy(viz_params)
    mod_params["draw_drone"] = True
    mod_params["draw_trajectory"] = True
    # frames = viz.top_down_visualization(env_id, seg_idx, seg, domain, mod_params)

    # Animation with just the drone
    frames = viz.top_down_visualization(env_id, seg_idx, rollout, domain, viz_params)
    outputs["top_down_viz"] = frames

    # Animation with bare visitation distributions
    mod_params = deepcopy(viz_params)
    mod_params["include_vdist"] = True
    mod_params["draw_drone"] = False
    mod_params["draw_topdown"] = False
    mod_params["draw_trajectory"] = False
    # frames = viz.top_down_visualization(env_id, seg_idx, seg, domain, mod_params)

    # Animation with visitation distributions the drone
    mod_params = deepcopy(viz_params)
    mod_params["include_vdist"] = True
    mod_params["draw_trajectory"] = True
    frames = viz.top_down_visualization(env_id, seg_idx, rollout, domain, mod_params)
    outputs["visitation_distributions"] = frames

    # Animation with ground-truth visitation distributions and the drone
    mod_params = deepcopy(viz_params)
    mod_params["include_vdist"] = True
    mod_params["draw_trajectory"] = True
    frames = viz.top_down_visualization(env_id, seg_idx, rollout, domain, mod_params, replace_vdist=gt_distr)
    outputs["visitation_distributions_ground_truth"] = frames


def visualize_instruction_chunks(outputs, rollout):
    p = Presenter()
    model_state = rollout[0]["model_state"]
    object_refs = model_state.get("object_references")[0]
    chunks = model_state.get("noun_chunks")[0]
    instruction_nl = rollout[0]["instruction"]

    clean = lambda r: r.replace("  ", " ")

    instruction_clean = clean(instruction_nl)
    chunks_clean = [clean(c) for c in chunks]
    objrefs_clean = [clean(c) for c in chunks]

    instr_image = np.ones((100, 800, 3), dtype=np.float32)
    cv2.putText(instr_image, instruction_clean, (20, 20), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 0), 1)
    outputs["instruction_img"] = [instr_image]

stage1 = None
def process_rollout_with_stage1_model(rollout):
    from data_io.models import load_model
    from learning.inputs.vision import standardize_images
    from learning.modules.generic_model_state import GenericModelState
    global stage1
    if stage1 is None:
        stage1, model_loaded = load_model("rpn_fspvn_stage1", "emnlp/fspvn/SuReAL_train_rpn_fspvn_aug1-2_noD_affinity_stage1_RL_epoch_189")

    # Reset model state - otherwise it drags around stuff from previous rollouts
    stage1.model_state = GenericModelState()

    images = torch.from_numpy(np.stack(standardize_images([s["state"].image for s in rollout])))
    states = torch.from_numpy(np.stack([s["state"].state for s in rollout]))
    instruction = rollout[0]["instruction"]
    reset_mask = [True] + [False for _ in range(len(rollout)-1)]

    #images, states, instruction, reset_mask, noisy_start_poses = None, start_poses = None, rl = False)
    ms = rollout[0]["model_state"]
    stage1.model_state.put("object_database", ms.get("object_database"))

    _ = stage1(images, states, instruction, reset_mask)
    new_model_state = stage1.model_state
    return new_model_state


def project_context_map_pca(context_map_w, all_obj_map_w=None):
    # Run collect_context_vectors_for_pca.py to obtain the values to key in here:
    projection = [[-0.018931491300463676, -0.18338057398796082, -0.17437446117401123, -0.1731671690940857, -0.1762692928314209, -0.18822386860847473, 0.18769529461860657, -0.18719397485256195, -0.149457648396492, 0.16913475096225739, 0.10103949904441833, -0.18551959097385406, -0.15537717938423157, 0.18577776849269867, 0.1861017495393753, -0.18409079313278198, -0.1854674071073532, -0.10422209650278091, 0.15432293713092804, -0.08339425921440125, -0.17420737445354462, -0.16766910254955292, -0.17774559557437897, 0.0647096112370491, 0.006504961755126715, -0.18781223893165588, 0.17455199360847473, -0.13503940403461456, 0.187985360622406, -0.02535456232726574, 0.18740534782409668, -0.1621875762939453, -0.1851835995912552, -0.18808887898921967, 0.18821066617965698, -0.1333964765071869, 0.18687893450260162, -0.031066354364156723, -0.08287868648767471, -0.16957314312458038], [0.28987374901771545, -0.0658140778541565, -0.10976985096931458, -0.1142561137676239, 0.10226374119520187, 0.004490386229008436, 0.02227458357810974, 0.030763491988182068, -0.17713724076747894, -0.12790930271148682, -0.24582630395889282, -0.049408700317144394, -0.16448445618152618, -0.04702785983681679, -0.043852582573890686, 0.06087895855307579, 0.04987587407231331, -0.24262244999408722, -0.16684432327747345, -0.26120108366012573, 0.11040323972702026, 0.13245083391666412, -0.0959482416510582, 0.2735961079597473, 0.29117655754089355, -0.01976737380027771, 0.10909110307693481, 0.20298591256141663, -0.015334892086684704, -0.2886958122253418, -0.027508288621902466, -0.1479000747203827, 0.05233996361494064, -0.011903966777026653, 0.005667541176080704, -0.20557180047035217, -0.035051874816417694, 0.2873556911945343, 0.2615939974784851, 0.1265110820531845], [-0.05288025364279747, 0.08749561011791229, 0.02370234951376915, 0.033248305320739746, 0.04045689105987549, 0.040643736720085144, -0.025619901716709137, 0.0316869281232357, 0.026297323405742645, -0.02610204368829727, -0.03423607349395752, 0.026118630543351173, 0.02402302995324135, -0.03985055163502693, -0.03896165266633034, 0.03553289175033569, -0.0010733414674177766, -0.27600663900375366, 0.32506704330444336, -0.06506498903036118, 0.02080586925148964, -0.2620297074317932, 0.19109006226062775, -0.1412237137556076, -0.1989353895187378, -0.6227127909660339, 0.04047978296875954, 0.1716797649860382, -0.29953163862228394, -0.22214116156101227, 0.05699804425239563, -0.03489818796515465, -0.06132280454039574, 0.13610908389091492, 0.030024342238903046, 0.005466270260512829, -0.16252972185611725, -0.05422830954194069, 0.0598214827477932, 0.07056742161512375]]
    projection = torch.tensor(projection)
    mean = [0.03934922814369202, 0.052875734865665436, 0.2698698937892914, 0.7091682553291321, 0.6862526535987854, 0.130939319729805, -0.47668886184692383, -0.018375584855675697, 0.3066449463367462, -0.15669572353363037, -0.21662086248397827, 0.03333839029073715, -0.047647517174482346, 0.1279371827840805, -0.7733791470527649, 0.1453520506620407, -0.05441907048225403, -0.8477436304092407, -0.20989462733268738, 0.13106195628643036, -0.004919820465147495, -0.6491976976394653, 0.11921393871307373, -0.0056121316738426685, 0.003704467322677374, -0.33916765451431274, -0.009462110698223114, 0.26798760890960693, -0.49255725741386414, -0.003006115322932601, -0.22368746995925903, -0.4210205078125, -0.2548539638519287, -0.10908783972263336, -0.3058468997478485, -0.5132524967193604, 0.10598336905241013, 0.0024959698785096407, 0.17877252399921417, 0.5246145725250244]
    mean = torch.tensor(mean)
    std = [0.02440059371292591, 0.13847635686397552, 0.17511118948459625, 0.17358258366584778, 0.21333931386470795, 0.11486600339412689, 0.2535282075405121, 0.025500783696770668, 0.3711088001728058, 0.17320752143859863, 0.3749515414237976, 0.07444033771753311, 0.1173565611243248, 0.24150291085243225, 0.0831538513302803, 0.5501543879508972, 0.08247661590576172, 0.15106499195098877, 0.6191529035568237, 0.23960384726524353, 0.025015754625201225, 0.07026197016239166, 0.029275167733430862, 0.003750394331291318, 0.06409405171871185, 0.2904139757156372, 0.0962347462773323, 0.19499096274375916, 0.4669228792190552, 0.07343295961618423, 0.7820687294006348, 0.5017632842063904, 0.394433856010437, 0.15536971390247345, 0.16691777110099792, 0.5115931630134583, 0.12964129447937012, 0.0024208202958106995, 0.20268209278583527, 0.4090252220630646]
    std = torch.tensor(std)

    # no renormalization
    projection = torch.tensor([[-0.0027037840336561203, -0.08148398995399475, -0.09658628702163696, -0.09492669999599457, -0.12662667036056519, -0.07077253609895706, 0.15462346374988556, -0.015739021822810173, -0.17024314403533936, 0.09951603412628174, 0.13941332697868347, -0.044529836624860764, -0.05635813996195793, 0.14859962463378906, 0.05120908096432686, -0.33676812052726746, -0.05070476606488228, -0.04528922587633133, 0.3298184871673584, -0.05476640537381172, -0.014713719487190247, -0.040086064487695694, -0.016538284718990326, 0.0006202862714417279, -0.0017860964871942997, -0.1773538589477539, 0.05314718186855316, -0.09276622533798218, 0.28817620873451233, -0.0025097832549363375, 0.4828016757965088, -0.25353801250457764, -0.24228715896606445, -0.09522974491119385, 0.1025499701499939, -0.2053707391023636, 0.07997456938028336, -0.00036325998371466994, -0.06386575102806091, -0.23549425601959229], [0.02932084910571575, -0.05120819807052612, -0.09610843658447266, -0.09841566532850266, 0.07174588739871979, -0.009015520103275776, 0.04818788543343544, 0.00081205228343606, -0.3033839762210846, -0.07745187729597092, -0.36563023924827576, -0.022506481036543846, -0.09009315818548203, -0.024282434955239296, -0.0072435177862644196, 0.08764219284057617, 0.009287695400416851, -0.16130831837654114, -0.3823468089103699, -0.2718757390975952, 0.009290372021496296, 0.03280523046851158, -0.014427438378334045, 0.004413594026118517, 0.07820964604616165, -0.052173417061567307, 0.05255355313420296, 0.15180739760398865, 0.015428833663463593, -0.08955920487642288, -0.014179501682519913, -0.3521866500377655, 0.04853672906756401, -0.02282879315316677, 0.020185712724924088, -0.4747794568538666, -0.006472825538367033, 0.00286831334233284, 0.21290111541748047, 0.18041805922985077], [-0.011421685107052326, -0.009718647226691246, 0.06396491825580597, 0.38964009284973145, -0.16714203357696533, 0.05355110764503479, -0.03060741536319256, 0.0060660443268716335, -0.07332358509302139, -0.0345040038228035, -0.019120827317237854, 0.011260947212576866, -0.005373836029320955, 0.03213369473814964, -0.09551378339529037, -0.01621846854686737, -0.04817216470837593, -0.23781391978263855, -0.12053996324539185, -0.03673354536294937, -0.009852023795247078, 0.12458959221839905, -0.026257222518324852, -0.00020086258882656693, -0.007914635352790356, -0.08571451902389526, -0.0713663101196289, 0.05242770165205002, -0.0737561509013176, 0.0042506493628025055, 0.5390014052391052, 0.5287283658981323, 0.07982102036476135, 0.08901482820510864, -0.07267850637435913, -0.16096335649490356, -0.0584758035838604, -0.001792166382074356, -0.02762274444103241, 0.2511129677295685]])

    # Channel zero is all-object mask
    norm_context_map_w = context_map_w[:, :, :, :]# - mean[None, :, None, None]
    norm_context_map_w = norm_context_map_w# / (std[None, :, None, None] + 1e-9)

    viz_context_maps_w = torch.einsum("bchw,ac->bahw", norm_context_map_w, projection)

    #mode_r = viz_context_maps_w[:, 0, :, :].reshape([-1]).mode()
    #mode_g = viz_context_maps_w[:, 1, :, :].reshape([-1]).mode()
    #mode_b = viz_context_maps_w[:, 1, :, :].reshape([-1]).mode()

    # Squash to zero-one range
    #viz_context_maps_w = viz_context_maps_w - viz_context_maps_w.min()
    #viz_context_maps_w = viz_context_maps_w / (viz_context_maps_w.max() + 1e-9)
    viz_context_maps_w = torch.sigmoid(viz_context_maps_w * 2)

    # Blend in the all-object mask
    #if all_obj_map_w is not None:
    #    full_viz_context_maps_w = (viz_context_maps_w + all_obj_map_w * 0.4).clamp(0, 1)
    #else:
    full_viz_context_maps_w = viz_context_maps_w

    frames = [map for map in full_viz_context_maps_w]
    return frames


def visualize_object_context_masks(outputs, rollout, model_state, env_img):
    p = Presenter()
    object_refs = model_state.get("object_references")[0]

    context_map_w = model_state.tensor_store.get_inputs_batch("context_grounding_map_w")[:, 0, :, :, :]
    frames = project_context_map_pca(context_map_w[:, 1:, :, :]) # For not, don't use all-object map: #, context_map_w[:, 0:1, :, :])
    frames_over_rgb = [p.overlaid_image(env_img, f, strength=0.7) for f in frames]
    outputs["object_context_map"] = frames
    outputs["object_context_map_over_rgb"] = frames_over_rgb

    # Save individual context maps for each object
    per_object_context_maps = model_state.tensor_store.get_inputs_batch("per_object_context_maps")[:, 0]
    for i in range(len(object_refs)):
        obj_ctx_map = per_object_context_maps[:, i]
        objref = object_refs[i]
        frames = project_context_map_pca(obj_ctx_map)
        frames_over_rgb = [p.overlaid_image(env_img, f, strength=0.7) for f in frames]
        outputs[f"per_object_context_maps/{objref}"] = frames
        outputs[f"per_object_context_maps_over_rgb/{objref}"] = frames_over_rgb

    # All object mask
    all_obj_mask = context_map_w[:, 0:1, :, :]
    all_obj_mask = all_obj_mask.repeat((1, 3, 1, 1))
    frames = [a for a in all_obj_mask]
    frames_over_rgb = [p.overlaid_image(env_img, f, strength=0.7) for f in frames]
    outputs[f"all_object_mask_w"] = frames
    outputs[f"all_object_mask_w_over_rgb"] = frames_over_rgb

    return outputs


def visualize_visitation_distributions(outputs, rollout, env_img, replace=None):
    p = Presenter()
    if replace is not None:
        vdists = [v for v in replace[0]]
        suffix = replace[1]
    else:
        log_vdists = rollout[0]["model_state"].tensor_store.get_inputs_batch("log_v_dist_w")
        vdists = [v.softmax() for v in log_vdists]
        suffix = ""

    frames_bars = [v.visualize(size=env_img.shape[0]) for v in vdists]
    frames_nobars = [v.visualize(size=env_img.shape[0], nobars=True) for v in vdists]
    frames_bars_over_rgb = [p.overlaid_image(env_img, f) for f in frames_bars]
    frames_nobars_over_rgb = [p.overlaid_image(env_img, f) for f in frames_nobars]
    zerobg = np.zeros_like(env_img)
    frames_bars = [p.overlaid_image(zerobg, f, strength=1.0) for f in frames_bars]
    frames_nobars = [p.overlaid_image(zerobg, f, strength=1.0) for f in frames_nobars]
    outputs[f"vdist_{suffix}w_bars"] = frames_bars
    outputs[f"vdist_{suffix}w_nobars"] = frames_nobars
    outputs[f"vdist_{suffix}w_bars_over_rgb"] = frames_bars_over_rgb
    outputs[f"vdist_{suffix}w_nobars_over_rgb"] = frames_nobars_over_rgb


def draw_alignments(image, obj_centroids, align_strengths, align_colors, chunk_labels, obj_chunk_ids, text_chunks, text_colors, settings):
    import cairo
    if settings == "A":
        WIDTH = 560
        # Guess if we need an extra space
        HEIGHT = 464 + (30 if len(" ".join(text_chunks)) > 120 or len(text_chunks) > 9 else 0)
        scale = WIDTH
        text_size = 0.0385
    else:
        WIDTH = 900
        # Guess if we need an extra space
        HEIGHT = 500 + (30 if len(" ".join(text_chunks)) > 120 or len(text_chunks) > 9 else 0)
        scale = WIDTH
        text_size = 0.0385

    T = HEIGHT - 384
    L = (WIDTH - 512) // 2
    l = float(L) / float(WIDTH)
    text_l = 0.015

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    # First draw the image:
    buf = surface.get_data()
    ih, iw, ic = image.shape
    npbuf = np.ndarray(shape=(HEIGHT, WIDTH, 4), dtype=np.uint8, buffer=buf)
    npbuf[:, :, :] = 255
    npbuf[T:T+ih, L:L+iw, :3] = (image * 255).astype(np.uint8)[:, :, [2, 1, 0]]
    npbuf[T:T+ih, L:L+iw, 3] = 255

    context = cairo.Context(surface)
    context.scale(scale, scale)

    context.select_font_face("Liberation sans", cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_NORMAL)
    context.set_source_rgba(0, 0, 0, 1)
    context.set_font_size(text_size)
    caret_x = text_l
    caret_y = 0.05
    context.move_to(caret_x, caret_y)
    objref_anchors = []
    r = 0
    newline = True
    for c, chunk in enumerate(text_chunks):
        color = text_colors[c]
        context.set_source_rgba(color[0], color[1], color[2], 1)
        x_bearing, y_bearing, width, height, x_advance, y_advance = context.text_extents(chunk)

        # Move to next line (#TODO maybe have more than two lines)
        if caret_x + x_advance > 1.0 - text_l:
            caret_y += 0.05
            caret_x = text_l
            newline = True
            context.move_to(caret_x, caret_y)

        if newline:
            chunk = chunk.lstrip() # Strip line start

        # Find anchor point
        if chunk_labels[c] == "objref":
            anchor_x = caret_x + width / 2
            anchor_y = caret_y + 0.01
            objref_anchors.append((anchor_x, anchor_y))
            r += 1

        context.show_text(chunk)
        newline = False
        caret_x += x_advance

    def point_img_to_cairo(pt_img_px):
        pt_cairo_px = np.asarray([pt_img_px[1], pt_img_px[0]]) + np.asarray([L, T])
        pt_cairo = pt_cairo_px / np.asarray([scale, scale])
        return pt_cairo.tolist()
    obj_centroids_rel = [point_img_to_cairo(pt) for pt in obj_centroids]

    context.set_line_width(0.01)
    context.stroke()
    for r in range(len(obj_centroids)):
        sx, sy = objref_anchors[r]
        ex, ey = obj_centroids_rel[r]
        s2x, s2y = sx, 0.3 * HEIGHT / scale
        s3x, s3y = ex, 0.3 * HEIGHT / scale
        s4x, s4y = ex, 0.4 * HEIGHT / scale
        color = align_colors[r]
        strength = align_strengths[r]
        strength = strength if strength > 0.01 else 0.0
        alpha = min(strength * 5, 1.0)
        context.set_source_rgba(color[0], color[1], color[2], alpha)
        context.move_to(sx, sy)
        context.curve_to(s2x, s2y, s3x, s3y, s4x, s4y)
        context.line_to(ex, ey)
        context.stroke()

    buf = surface.get_data()
    array = np.ndarray(shape=(HEIGHT, WIDTH, 4), dtype=np.uint8, buffer=buf)[:, :, [2, 1, 0, 3]]
    return array.astype(np.float32) / 255


def compute_blob_centroid(mask, scale=1):
    mask_blur = gaussian_filter(mask.sum(2), 20)
    amax = np.argmax(mask_blur)
    amaxu = np.unravel_index(amax, mask_blur.shape)
    return (amaxu[0] * scale, amaxu[1] * scale)


def compute_blob_strength(mask):
    mask_blur = gaussian_filter(mask.sum(2), 20)
    return min(mask_blur.max(), 1.0)


def chunkate_instruction(instr, markchunks, objrefs):
    rem = instr
    outchunks = []
    chunklabels = []
    chunkindices = []
    i = 0
    for i, ref in enumerate(markchunks):
        idx = rem.find(ref)
        outchunks.append(rem[:idx])
        chunklabels.append("text")
        chunkindices.append(-1)
        outchunks.append(ref)
        if ref in objrefs:
            chunklabels.append("objref")
        else:
            chunklabels.append("spurious")
        chunkindices.append(i)
        rem = rem[idx + len(ref):]
    return outchunks, chunklabels, chunkindices


def visualize_image_text_alignment(outputs, rollout):
    p = Presenter()

    # Get text related stuff - common for all frames
    model_state = rollout[0]["model_state"]
    object_refs = model_state.get("object_references")[0]
    chunks = model_state.get("noun_chunks")[0]
    instruction_nl = rollout[0]["instruction"]
    clean = lambda r: r.replace("  ", " ")
    instruction_clean = clean(instruction_nl)
    chunks_clean = [clean(c) for c in chunks]
    objrefs_clean = [clean(c) for c in object_refs]

    frames_msk = []
    frames_bnd = []
    frames_msk_b = []
    frames_bnd_b = []
    for i in range(len(rollout)):
        s = rollout[i]
        fpv_image = p.prep_image(s["state"].image, scale=4)
        masks_over_rgb = p.prep_image(outputs["object_masks_fpv_over_rgb"][i], scale=4)
        boundaries_over_rgb = p.prep_image(outputs["object_boundaries_fpv_over_rgb"][i], scale=4)
        object_masks = [outputs[f"objects_fpv/{objref}"][i] for objref in object_refs]
        object_centroids = [compute_blob_centroid(mask, scale=4) for mask in object_masks]
        alignment_strengths = [compute_blob_strength(mask) for mask in object_masks]
        alignment_colors = [object_colors[j] for j in range(len(object_refs))]

        chunklist, chunk_labels, chunkindices = chunkate_instruction(instruction_clean, chunks_clean, objrefs_clean)
        text_colors = []
        r = 0
        for c in range(len(chunkindices)):
            if chunk_labels[c] == "text":
                text_colors.append(np.asarray([0.5, 0.5, 0.5]))
            elif chunk_labels[c] == "spurious":
                text_colors.append(np.asarray([0.2, 0.2, 0.2]))
            else:
                text_colors.append(object_colors[r])
                r += 1
        frame_msk = draw_alignments(masks_over_rgb, object_centroids, alignment_strengths, alignment_colors, chunk_labels, chunkindices, chunklist, text_colors, settings="A")
        frames_msk.append(frame_msk)
        frame_bnd = draw_alignments(boundaries_over_rgb, object_centroids, alignment_strengths, alignment_colors, chunk_labels, chunkindices, chunklist, text_colors, settings="A")
        frames_bnd.append(frame_bnd)
        frame_msk = draw_alignments(masks_over_rgb, object_centroids, alignment_strengths, alignment_colors, chunk_labels, chunkindices, chunklist, text_colors, settings="B")
        frames_msk_b.append(frame_msk)
        frame_bnd = draw_alignments(boundaries_over_rgb, object_centroids, alignment_strengths, alignment_colors, chunk_labels, chunkindices, chunklist, text_colors, settings="B")
        frames_bnd_b.append(frame_bnd)
    outputs["fpv_mask_alignments_A"] = frames_msk
    outputs["fpv_boundary_alignments_A"] = frames_bnd
    outputs["fpv_mask_alignments_B"] = frames_msk_b
    outputs["fpv_boundary_alignments_B"] = frames_bnd_b


def make_action(outputs, rollout):
    p = Presenter()
    actions = [s["action"] for s in rollout]
    frames = [(np.zeros((420, 420, 3))).astype(np.uint8) for _ in rollout]
    frames = [p.draw_action(f, (2, 335), action).astype(np.float32) / 255 for f, action in zip(frames, actions)]
    # Make it bigger!
    frames = [p.scale_image(f, scale=4) for f in frames]
    outputs["action"] = frames


def make_stage2_inputs(outputs, rollout):
    from learning.modules.structured_map_layers import StructuredMapLayers
    from learning.modules.map_transformer import MapTransformer
    from learning.models.navigation_model_component_base import NavigationModelComponentBase
    from learning.modules.pvn.pvn_stage2_rlbase import PVN_Stage2_RLBase
    from learning.inputs.pose import Pose

    # Build the tools that we need
    structured_layers = StructuredMapLayers(32)
    map_transformer_w_to_r = MapTransformer(
        source_map_size=32,
        dest_map_size=64,
        world_size_m=4.7,
        world_size_px=32
    )
    comp = NavigationModelComponentBase()
    rlbase = PVN_Stage2_RLBase(map_channels=2,
                               map_struct_channels=2,
                               crop_size=16,
                               map_size=64,
                               h1=16, h2=16, structure_h1=8, obs_dim=16, name="action")
    p = Presenter()

    # Extract data
    model_state = rollout[-1]['model_state']
    states = torch.from_numpy(np.stack([s["state"].state for s in rollout]))
    drn_poses = comp.poses_from_states(states)
    #bs = drn_poses.position.shape[0]

    accum_obs_masks_w = model_state.tensor_store.get_inputs_batch("accum_obs_masks_w")[:, 0]
    map_uncoverage_w = 1 - accum_obs_masks_w

    # This is some bug where sometimes one pose is missing?
    bs = accum_obs_masks_w.shape[0]
    if bs > drn_poses.position.shape[0]:
        posslice = drn_poses.position[-1:]
        rotslice = drn_poses.orientation[-1:]
        pos = torch.cat([drn_poses.position, posslice])
        rot = torch.cat([drn_poses.orientation, rotslice])
        drn_poses = Pose(pos, rot)

    log_v_dist_w = Partial2DDistribution.cat(model_state.tensor_store.get_inputs_batch("log_v_dist_w"))
    v_dist_w = log_v_dist_w.softmax()
    x = v_dist_w.inner_distribution
    xr, r_poses = map_transformer_w_to_r(x, None, drn_poses)
    v_dist_r = Partial2DDistribution(xr, v_dist_w.outer_prob_mass)

    # Compute things
    structured_map_info_r, map_info_w = structured_layers.build(map_uncoverage_w,
                                                                drn_poses,
                                                                map_transformer_w_to_r,
                                                                use_map_boundary=True)
    v_dist_r_crop = rlbase.crop_maps(v_dist_r)
    v_dist_r_crop_frames = []
    for frame in v_dist_r_crop:
        mx = frame.reshape([2, -1]).max(1).values
        frame = frame / mx[:, None, None]
        v_dist_r_crop_frames.append(frame)

    struct_input_frames = [f for f in structured_map_info_r]
    bw_struct_input_frames = [(1 - f.max(0, keepdim=True).values) for f in struct_input_frames]
    ego_obs_frames = [1 - f[0] for f in structured_map_info_r]
    bg = np.zeros((128, 128, 3), dtype=np.float32)
    ego_obs_frames = [p.overlaid_image(bg, f, strength=1.0, interpolation=cv2.INTER_LINEAR) for f in ego_obs_frames]
    bw_struct_input_frames = [p.overlaid_image(bg, f, strength=1.0, interpolation=cv2.INTER_LINEAR) for f in bw_struct_input_frames]

    outputs["v_dist_r_crop"] = v_dist_r_crop_frames
    outputs["stage2_struct_inputs"] = struct_input_frames
    outputs["stage2_bw_struct_inputs"] = bw_struct_input_frames
    outputs["accum_obs_mask_r"] = ego_obs_frames


def visualize_ground_truth_distributions(outputs, rollout):
    from learning.datasets import aux_data_providers as aup
    env_id = rollout[0]['env_id']
    seg_idx = rollout[0]['seg_idx']
    set_idx = 0
    seg = aup.get_instruction_segment(env_id, set_idx, seg_idx)
    start_idx = seg["start_idx"]
    end_idx = seg["end_idx"]
    frame = aup.get_top_down_ground_truth_static_global(env_id, start_idx, end_idx, 32, 32, 32, 32)
    frame = frame.repeat((len(rollout), 1, 1, 1))
    obs_masks = rollout[-1]["model_state"].tensor_store.get_inputs_batch("accum_obs_masks_w")[:, 0]

    gt_distributions = Partial2DDistribution.from_distribution_and_mask(frame, obs_masks[:, 0, :, :])
    return gt_distributions


def generate_graphics_visualizations_from_rollout(rollout, run_name):
    if len(rollout) == 0:
        return

    env_id = rollout[0]['env_id']
    seg_idx = rollout[0]['seg_idx']
    print(f"PROCESSING: {env_id}-{seg_idx}")

    RES = 864
    # TODO: Add top-down overlay variants to everything that is top-down, using this image.
    env_img = load_env_img(env_id, RES, RES, real_drone=True, origin_bottom_left=True, flipdiag=True, alpha=True)

    ONLY = [
        (7512, 4),
        (7514, 4),
        (7446, 6),
        (7136, 6),
    ]
    if (env_id, seg_idx) not in ONLY:
        return

    p = Presenter()
    outputs = {}
    make_stage2_inputs(outputs, rollout)

    make_fpv_masks(outputs, rollout)

    make_allo_masks(outputs, rollout, env_img)

    make_action(outputs, rollout)

    # WARNING: This depends on outputs containing certain previous results
    visualize_image_text_alignment(outputs, rollout)

    gt_distr = visualize_ground_truth_distributions(outputs, rollout)
    corl2019_visualizations(outputs, rollout, gt_distr, domain="real")

    visualize_instruction_chunks(outputs, rollout)

    ms = process_rollout_with_stage1_model(rollout)
    visualize_object_context_masks(outputs, rollout, ms, env_img)

    visualize_visitation_distributions(outputs, rollout, env_img)
    visualize_visitation_distributions(outputs, rollout, env_img, replace=[gt_distr, "gt_"])

    # First extract images and external inputs from the rollout
    images = np.stack([r["state"].image for r in rollout])        # Tx96x128x3 numpy.ndarray
    instruction_nl = rollout[0]['instruction']          # string
    outputs["instruction_nl"] = instruction_nl
    #argbomb["anon_instruction_nl"] = viz_data[0]["anon_instruction_nl"]
    outputs["fpv_images"] = [p.prep_image(i) for i in images]

    # Plot drone trajectory:
    viz_params = {
        "draw_topdown": True,
        "draw_trajectory": True,
        "draw_drone": True
    }
    viz = RolloutVisualizer(resolution=400)
    frames = viz.top_down_visualization(env_id, seg_idx, rollout, "sim", viz_params)
    frames = [np.flipud(np.fliplr(np.rot90(f))) for f in frames]
    outputs["top_down_frames"] = frames

    # Output the media
    base_dir = paths.get_rollout_debug_viz_dir()
    run_dir = os.path.join(base_dir, run_name)
    rollout_name = f"rollout_{env_id}_{seg_idx}"
    rollout_dir = os.path.join(run_dir, rollout_name)
    vid_dir = os.path.join(rollout_dir, "videos")
    frame_dir = os.path.join(rollout_dir, "frames")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    export_media(vid_dir, frame_dir, outputs)
