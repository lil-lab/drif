from data_io.model_io import load_pytorch_model
from learning.models.model_chaplot import ModelChaplot
from learning.models.model_gs_fpv_mem import ModelGSFPV
from learning.models.model_misra2017 import ModelMisra2017
from learning.models.model_sm_action_gt import ModelTrajectoryToAction
from learning.models.model_sm_rss_global import ModelTrajectoryTopDown as ModelRSS
from learning.models.model_sm_trajectory_ratio import ModelTrajectoryTopDown as ModelTrajectoryProbRatio
import learning.models.model_sm_trajectory_ratio as mtpr
from learning.models.supervised.top_down_path_goal_predictor_pretrain_batched import \
    ModelTopDownPathGoalPredictor as ModelTopDownPathGoalPredictorBatched
from learning.utils import get_n_params, get_n_trainable_params
from parameters.parameter_server import get_current_parameters
from policies.baseline_straight import BaselineStraight
from policies.basic_carrot_planner import BasicCarrotPlanner
from policies.fancy_carrot_planner import FancyCarrotPlanner
from policies.simple_carrot_planner import SimpleCarrotPlanner
from policies.stop import BaselineStop

import parameters.parameter_server as P


def load_model(model_file_override=None, real=False, model_name_override=False):

    setup = P.get_current_parameters()["Setup"]
    model_name = model_name_override or setup["model"]
    model_file = model_file_override or setup["model_file"] or None
    cuda = setup["cuda"]
    run_name = setup["run_name"]

    model = None
    pytorch_model = False

    # -----------------------------------------------------------------------------------------------------------------
    # Oracles / baselines that ignore images
    # -----------------------------------------------------------------------------------------------------------------

    if model_name == "oracle":
        rollout_params = get_current_parameters()["Rollout"]
        if rollout_params["oracle_type"] == "SimpleCarrotPlanner":
            model = SimpleCarrotPlanner()
            print("Using simple carrot planner")
        elif rollout_params["oracle_type"] == "BasicCarrotPlanner":
            model = BasicCarrotPlanner()
            print("Using basic carrot planner")
        elif rollout_params["oracle_type"] == "FancyCarrotPlanner":
            model = FancyCarrotPlanner()
            print("Using fancy carrot planner")
        else:
            print("UNKNOWN ORACLE: ", rollout_params["OracleType"])
            exit(-1)
    elif model_name == "baseline_straight":
        model = BaselineStraight()
    elif model_name == "baseline_stop":
        model = BaselineStop()

    # -----------------------------------------------------------------------------------------------------------------
    # FASTER RSS 2018 Resubmission Model
    # -----------------------------------------------------------------------------------------------------------------

    elif model_name == "gsmn":
        model = ModelRSS(run_name, model_class=ModelRSS.MODEL_RSS,
          aux_class_features=False, aux_grounding_features=False,
          aux_class_map=True, aux_grounding_map=True, aux_goal_map=True, aux_lang=True); pytorch_model = True
    elif model_name == "gsmn_wo_jlang":
        model = ModelRSS(run_name, model_class=ModelRSS.MODEL_RSS,
          aux_class_map=True, aux_grounding_map=True, aux_goal_map=True, aux_lang=False); pytorch_model = True
    elif model_name == "gsmn_wo_jgnd":
        model = ModelRSS(run_name, model_class=ModelRSS.MODEL_RSS,
          aux_class_map=True, aux_grounding_map=False, aux_goal_map=True, aux_lang=True); pytorch_model = True
    elif model_name == "gsmn_wo_jclass":
        model = ModelRSS(run_name, model_class=ModelRSS.MODEL_RSS,
          aux_class_map=False, aux_grounding_map=True, aux_goal_map=True, aux_lang=True); pytorch_model = True
    elif model_name == "gsmn_wo_jgoal":
        model = ModelRSS(run_name, model_class=ModelRSS.MODEL_RSS,
          aux_class_map=True, aux_grounding_map=True, aux_goal_map=False, aux_lang=True); pytorch_model = True

    elif model_name == "gsmn_w_posnoise":
        model = ModelRSS(run_name, model_class=ModelRSS.MODEL_RSS,
          aux_class_features=False, aux_grounding_features=False,
          aux_class_map=True, aux_grounding_map=True, aux_goal_map=True, aux_lang=True,
                         pos_noise=True, rot_noise=False); pytorch_model = True
    elif model_name == "gsmn_w_rotnoise":
        model = ModelRSS(run_name, model_class=ModelRSS.MODEL_RSS,
          aux_class_features=False, aux_grounding_features=False,
          aux_class_map=True, aux_grounding_map=True, aux_goal_map=True, aux_lang=True,
                         pos_noise=False, rot_noise=True); pytorch_model = True
    elif model_name == "gsmn_w_bothnoise":
        model = ModelRSS(run_name, model_class=ModelRSS.MODEL_RSS,
          aux_class_features=False, aux_grounding_features=False,
          aux_class_map=True, aux_grounding_map=True, aux_goal_map=True, aux_lang=True,
                         pos_noise=True, rot_noise=True); pytorch_model = True

    elif model_name == "gs_fpv":
        model = ModelGSFPV(run_name, aux_class_features=True, aux_grounding_features=True, aux_lang=True, recurrence=False); pytorch_model = True
    elif model_name == "gs_fpv_mem":
        model = ModelGSFPV(run_name, aux_class_features=True, aux_grounding_features=True, aux_lang=True, recurrence=True); pytorch_model = True

    # -----------------------------------------------------------------------------------------------------------------
    # CoRL 2018 Model
    # -----------------------------------------------------------------------------------------------------------------

    elif model_name == "sm_traj_nav_ratio":
        model = ModelTrajectoryProbRatio(run_name, model_class=mtpr.MODEL_FPV); pytorch_model = True
    elif model_name == "sm_traj_nav_ratio_path":
        model = ModelTrajectoryProbRatio(run_name, model_class=mtpr.PVN_STAGE1_ONLY); pytorch_model = True

    elif model_name == "action_gtr":
        model = ModelTrajectoryToAction(run_name); pytorch_model = True

    # -----------------------------------------------------------------------------------------------------------------
    # CoRL 2018 Refactored
    # -----------------------------------------------------------------------------------------------------------------

    elif model_name == "pvn_full":
        model = ModelTrajectoryProbRatio(run_name, model_class=mtpr.MODEL_FPV); pytorch_model = True
    elif model_name == "pvn_stage1":
        model = ModelTrajectoryProbRatio(run_name, model_class=mtpr.PVN_STAGE1_ONLY); pytorch_model = True
    elif model_name == "pvn_stage2":
        model = ModelTrajectoryToAction(run_name); pytorch_model = True

    # -----------------------------------------------------------------------------------------------------------------
    # CoRL 2018 Top-Down Full Observability Models
    # -----------------------------------------------------------------------------------------------------------------

    elif model_name == "top_down_goal_batched":
        model = ModelTopDownPathGoalPredictorBatched(run_name); pytorch_model = True

    # -----------------------------------------------------------------------------------------------------------------
    # CoRL Baselines
    # -----------------------------------------------------------------------------------------------------------------
    elif model_name == "chaplot":
        model = ModelChaplot(run_name); pytorch_model = True

    elif model_name == "misra2017":
        model = ModelMisra2017(run_name); pytorch_model = True

    model_loaded = False
    if pytorch_model:
        n_params = get_n_params(model)
        n_params_tr = get_n_trainable_params(model)
        print("Loaded PyTorch model!")
        print("Number of model parameters: " + str(n_params))
        print("Trainable model parameters: " + str(n_params_tr))
        model.init_weights()
        model.eval()
        if model_file:
            load_pytorch_model(model, model_file)
            print("Loaded previous model: ", model_file)
            model_loaded = True
        if cuda:
            model = model.cuda()

    return model, model_loaded
