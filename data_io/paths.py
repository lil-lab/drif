import os

from parameters.parameter_server import get_current_parameters
import parameters.parameter_server as P

# Simulator
import json
import numpy as np

# Configs
# --------------------------------------------------------------------------------------------


def get_sim_executable_path():
    return get_current_parameters()["Environment"]["simulator_path"]


def get_sim_config_path_for_instance(instance_id):
    return os.path.join(get_current_parameters()["Environment"]["sim_config_dir"], "current_config", str(instance_id))


def get_sim_config_dir():
    return get_current_parameters()["Environment"]["sim_config_dir"]


def get_grounding_data_dir():
    return get_current_parameters()["Environment"]["grounding_data_dir"]


def get_grounding_dataset_dir(dataset_name):
    return os.path.join(get_grounding_data_dir(), dataset_name)


def get_object_database_dir(dataset_name):
    config_dir = get_config_base_dir()
    nod_dir = os.path.join(config_dir, "object_databases")
    if dataset_name is not None:
        nod_dir = os.path.join(nod_dir, dataset_name)
    return nod_dir


def get_dashboard_dir():
    config_dir = get_config_base_dir()
    dashboard_dir = os.path.join(config_dir, "dashboards")
    return dashboard_dir


def get_run_dashboard_dir(run_name):
    dashboard_dir = get_dashboard_dir()
    run_dashboard_dir = os.path.join(dashboard_dir, run_name)
    return run_dashboard_dir

# Configs
# --------------------------------------------------------------------------------------------
def get_env_config_path(env_id):
    return os.path.join(get_config_dir(), "configs", "random_config_%d.json" % env_id)
#    return os.path.join(get_env_config_dir(), "config_%d.json" % env_id)


def get_template_path(env_id):
    return os.path.join(get_config_dir(), "templates", "random_template_%d.json" % env_id)


def get_instructions_path(env_id):
    return os.path.join(get_config_dir(), "instructions", "instructions_%d.txt" % env_id)


def get_curve_path(env_id):
    return os.path.join(get_config_dir(), "paths", "random_curve_%d.json" % env_id)


def get_curve_plot_path(env_id):
    return os.path.join(get_config_dir(), "plots", "random_curve_%d.jpg" % env_id)


def get_anno_curve_path(env_id):
    return os.path.join(get_config_dir(), "anno_paths", "anno_curve_%d.json" % env_id)


def get_fpv_img_dir(real=True):
    subdir = "drone_img" if real else "sim_img"
    return os.path.join(get_config_dir(), subdir)


def get_fpv_img_flight_dir(env_id, real=True):
    return os.path.join(get_fpv_img_dir(real), "flight_%d" % env_id)


def get_all_poses_dir():
    return os.path.join(get_config_dir(), "poses")


def get_poses_dir(env_id):
    return os.path.join(get_all_poses_dir(), "flight_" + str(env_id))


def get_all_real_images_dir():
    return os.path.join(get_config_dir(), "drone_img")


def get_all_sim_images_dir():
    return os.path.join(get_config_dir(), "sim_img")


def get_real_images_dir(env_id):
    return os.path.join(get_all_real_images_dir(), "flight_" + str(env_id))


def get_sim_images_dir(env_id):
    return os.path.join(get_all_sim_images_dir(), "flight_" + str(env_id))


def get_env_config_dir():
    return os.path.join(get_config_dir(), "configs")


def get_pose_path(env_id, pose_id):
    return os.path.join(get_poses_dir(env_id), "pose_%d.json" % pose_id)


def get_real_img_path(env_id, pose_id):
    return os.path.join(get_real_images_dir(env_id), "usb_cam_%d.jpg" % pose_id)


def get_sim_img_path(env_id, pose_id):
    return os.path.join(get_sim_images_dir(env_id), "usb_cam_%d.png" % pose_id)


def get_plots_dir():
    return os.path.join(get_config_dir(), "plots")


def get_samples_dir():
    return os.path.join(get_config_dir(), "samples")


def get_rollout_plots_dir():
    return os.path.join(get_config_dir(), "policy_roll", "plots")


def get_rollout_samples_dir():
    return os.path.join(get_config_dir(), "policy_roll", "samples")


def get_rollout_video_dir(run_name=""):
    viddir = os.path.join(get_config_base_dir(), f"rollout_video{'/'+run_name if run_name else ''}")
    os.makedirs(viddir, exist_ok=True)
    return viddir


def get_logging_dir():
    ldir = os.path.join(get_config_base_dir(), "logs")
    os.makedirs(ldir, exist_ok=True)
    return ldir


def get_tmp_dir_for_run(run_name):
    logdir = get_logging_dir()
    rundir = os.path.join(logdir, "tmp", run_name)
    os.makedirs(rundir, exist_ok=True)
    return rundir


def get_sprites_dir():
    sdir = os.path.join(get_config_base_dir(), "sprites")
    return sdir

# Instruction Data
# --------------------------------------------------------------------------------------------
def get_instruction_annotations_path():
    return os.path.join(get_config_dir(), "annotation_results.json")


# Data and Models
# --------------------------------------------------------------------------------------------

def get_config_base_dir():
    base_dir = get_current_parameters()["Environment"]["config_dir"]
    return base_dir


def get_pretrained_weight_dir():
    return os.path.join(get_model_dir(), "pretrained_modules")


def get_wordset_path(corpus_name=None):
    wordset_corpus = corpus_name or P.get_current_parameters()["Setup"]["use_wordset_from_corpus"]
    wordset_path = os.path.join(get_corpus_dir(wordset_corpus), "wordset.json")
    return wordset_path


def get_wordfreq_path(corpus_name=None):
    wordset_path = os.path.join(get_wordset_corpus_dir(corpus_name), "wordfreq.json")
    return wordset_path


def get_corpus_dir(corpus_name=None):
    corpus_name = corpus_name or P.get_current_parameters()["Setup"]["corpus_name"]
    return os.path.join(get_config_dir(), "corpora", corpus_name)


def get_wordset_corpus_dir(corpus_name=None):
    corpus_name = corpus_name or P.get_current_parameters()["Setup"]["use_wordset_from_corpus"]
    return os.path.join(get_config_dir(), "corpora", corpus_name)


def get_model_dir():
    return os.path.join(get_config_base_dir(), "models")


def get_dataset_dir(dataset_name):
    return os.path.join(get_config_base_dir(), "data", dataset_name)


def get_rollout_viz_dir():
    return os.path.join(get_config_base_dir(), "rollout_amt_viz")


def get_rollout_debug_viz_dir():
    return os.path.join(get_config_base_dir(), "rollout_dbg_viz")


def get_config_dir():
    return os.path.join(get_config_base_dir(), "configs")


def get_instruction_cache_dir():
    return os.path.join(get_corpus_dir(), "_cache")


def get_tmp_dir():
    return os.path.join(get_config_base_dir(), "configs", "tmp_junk")


def get_supervised_data_filename(env):
    filename = "supervised_train_data_env_" + str(env)
    return filename


def get_landmark_weights_path():
    filename = os.path.join(get_corpus_dir(), "landmark_counts.txt")
    return filename


def get_noisy_pose_path():
    path = os.path.join(get_dataset_dir(), "noisy_poses")
    return path


# Results
# --------------------------------------------------------------------------------------------

def get_results_path(run_name, makedir=False):
    dir = os.path.join(get_config_base_dir(), "results")
    if makedir:
        os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, run_name + "_results.json")


def get_results_dir(run_name=None, makedir=False):
    if run_name is not None:
        dir = os.path.join(get_config_base_dir(), "results", run_name)
    else:
        dir = os.path.join(get_config_base_dir(), "results")
    if makedir:
        os.makedirs(dir, exist_ok=True)
    return dir


# Others
# --------------------------------------------------------------------------------------------

def get_landmark_image_path(landmark_name):
    path = os.path.join(get_config_base_dir(), "landmark_images", f"{landmark_name}.png")
    return path


def get_landmark_images_dir():
    basedir = os.path.join(get_config_base_dir(), "landmark_images")
    return basedir


def get_env_image_path(env_id, real_drone=False):
    config_path = get_config_dir()
    if real_drone:
        subdir = "real"
    else:
        subdir = "simulator"
    # subdir="simulator"
    img_path = os.path.join(config_path, "env_img", subdir, str(env_id) + ".png")
    return img_path


def get_current_config_folder(i=None, instance_id=None):
    current_conf_folder = "current_config"
    if instance_id is not None:
        current_conf_folder += "/" + str(instance_id)
    folder = current_conf_folder if i is None else "configs"
    return folder

# ---------------------------------------------------------------------------------
# Text Data
# ---------------------------------------------------------------------------------

def get_english_vocab_path():
    path = os.path.join(get_config_base_dir(), "english_vocabulary.json")
    return path


def get_thesaurus_path():
    path = os.path.join(get_wordset_corpus_dir(), "thesaurus.json")
    return path


def get_noun_chunk_corpus_path():
    path = os.path.join(get_corpus_dir(), "noun_chunk_corpus.json")
    return path


def get_noun_chunk_instruction_cache_path():
    path = os.path.join(get_corpus_dir(), "noun_chunk_instruction_cache.json")
    return path


def get_noun_chunk_frequencies_path():
    path = os.path.join(get_corpus_dir(), "noun_chunk_frequencies.json")
    return path


def get_chunk_landmark_alignment_path(split="train"):
    path = os.path.join(get_corpus_dir(), f"noun_chunk_landmark_alignments_{split}.json")
    return path


def get_chunk_landmark_set_path(split):
    dir = os.path.join(get_corpus_dir(), "alignment_data")
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f"chunk_landmark_sets_{split}.json")
    return path


def get_unaligned_chunk_list_path():
    path = os.path.join(get_corpus_dir(), "unaligned_noun_chunk_list.json")
    return path


def get_human_eval_envs_path():
    path = os.path.join(get_config_dir(), "human_eval_envs.json")
    return path


def get_human_eval_root_path():
    path = os.path.join(get_config_base_dir(), "human_eval")
    return path


def get_env_split_path():
    path = os.path.join(get_config_dir(), "train_env_split.json")
    return path


def get_config_metadata_path():
    path = os.path.join(get_config_dir(), "config_metadata.json")
    return path


def get_ceiling_cam_calibration_path():
    path = os.path.expanduser("~/Documents/ceiling_cam_calibration")
    return path


def get_eval_tmp_dataset_name(model_name, run_name):
    fname = f"eval/{model_name}--{run_name}"
    return fname


def get_amt_batch_dir():
    return os.path.join(get_config_base_dir(), "amt_human_eval_batch")

########### Landmark locations ################

def get_landmark_locations(conf_json):
    landmark_loc = []
    for i, x in enumerate(conf_json['xPos']):
        landmark_loc.append(np.array([x, conf_json['zPos'][i],0]))
    return landmark_loc
