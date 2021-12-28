from data_io.model_io import load_pytorch_model
from learning.models.model_chaplot import ModelChaplot
from learning.models.model_gs_fpv_mem import ModelGSFPV
from learning.models.model_misra2017 import ModelMisra2017
from learning.models.model_sm_action_gt import ModelTrajectoryToAction
from learning.models.model_sm_rss_global import ModelTrajectoryTopDown as ModelRSS
import learning.models.model_sm_rss_global as msrg
from learning.models.model_sm_trajectory_ratio import ModelTrajectoryTopDown as ModelTrajectoryProbRatio
import learning.models.model_sm_trajectory_ratio as mtpr
from learning.models.supervised.top_down_path_goal_predictor_pretrain_batched import \
    ModelTopDownPathGoalPredictor as ModelTopDownPathGoalPredictorBatched

from learning.models.model_pvn_stage1_bidomain_originalrec import PVN_Stage1_Bidomain_Original
from learning.models.model_pvn_stage1_bidomain import PVN_Stage1_Bidomain
from learning.models.model_pvn_stage2_bidomain import PVN_Stage2_Bidomain
from learning.models.model_pvn_stage2_actor_critic import PVN_Stage2_ActorCritic
from learning.models.model_pvn_wrapper_bidomain import PVN_Wrapper_Bidomain
from learning.models.model_fspvn_wrapper_bidomain import FS_PVN_Wrapper_Bidomain
from learning.models.model_pvn_stage1_critic import PVN_Stage1_Critic
from learning.models.model_pvn_stage1_critic_big import PVN_Stage1_Critic_Big

from learning.models.model_object_recognizer import ObjectRecognizer
from learning.models.model_few_shot_instance_recognizer import ModelFewShotInstanceRecognizer
from learning.models.model_few_shot_instance_recognizer_multiscale import ModelFewShotInstanceRecognizerMultiscale
from learning.models.model_few_shot_instance_recognizer_sliding_window import \
    ModelFewShotInstanceRecognizerSlidingWindow
from learning.models.model_simple_matching_network import ModelSimpleMatchingNetwork
from learning.models.model_multi_matching_network_multi import ModelMultiMatchingNetwork
from learning.models.model_facebook_rpn_wrapper import ModelFacebookRPNWrapper
from learning.models.model_region_refinement import ModelRegionRefinementNetwork

from learning.models.model_fewshot_stage1_bidomain import FewShot_Stage1_Bidomain
from learning.models.model_rpn_fewshot_stage1_bidomain import RPN_FewShot_Stage1_Bidomain

from learning.models.model_object_reference_recognizer import ModelObjectReferenceRecognizer
from learning.models.model_object_reference_recognizer_given_database import ModelObjectReferenceRecognizerWithDb

from learning.models.model_gsmn_bidomain import ModelGSMNBiDomain
from learning.models.model_gsmn_critic import ModelGsmnCritic

from learning.utils import get_n_params, get_n_trainable_params
from parameters.parameter_server import get_current_parameters
from policies.baseline_average import BaselineAverage
from policies.basic_carrot_planner import BasicCarrotPlanner
from policies.fancy_carrot_planner import FancyCarrotPlanner
from policies.simple_carrot_planner import SimpleCarrotPlanner
from policies.stop import BaselineStop

from drones.train_real.PerceptionModel import PerceptionModel
from drones.train_real.domain_wrappers import DomainWrapperModel, DomainWrapperModel2Nets


import parameters.parameter_server as P


def load_model(model_name_override=False, model_file_override=None, domain="sim"):

    setup = P.get_current_parameters()["Setup"]
    model_name = model_name_override or setup["model"]
    model_file = model_file_override or setup["model_file"] or None
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
    elif model_name == "average":
        model = BaselineAverage()
    elif model_name == "stop":
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

    # -----------------------------------------------------------------------------------------------------------------
    # RSS Baselines
    # -----------------------------------------------------------------------------------------------------------------

    elif model_name == "gs_fpv":
        model = ModelGSFPV(run_name, aux_class_features=True, aux_grounding_features=True, aux_lang=True, recurrence=False); pytorch_model = True
    elif model_name == "gs_fpv_mem":
        model = ModelGSFPV(run_name, aux_class_features=True, aux_grounding_features=True, aux_lang=True, recurrence=True); pytorch_model = True

    # -----------------------------------------------------------------------------------------------------------------
    # RSS Model for Cage
    # -----------------------------------------------------------------------------------------------------------------

    elif model_name == "gsmn_cage":
        model = ModelRSS(run_name, model_class=msrg.MODEL_RSS,
          aux_class_features=False, aux_grounding_features=False,
          aux_class_map=True, aux_grounding_map=True, aux_goal_map=True, aux_lang=False); pytorch_model = True

    elif model_name == "gsmn_bidomain":
        model = ModelGSMNBiDomain(run_name, model_instance_name=domain); pytorch_model = True

    elif model_name == "gsmn_critic":
        model = ModelGsmnCritic(run_name); pytorch_model = True

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


    # -----------------------------------------------------------------------------------------------------------------
    # CoRL Model for cage (bidomain)
    # -----------------------------------------------------------------------------------------------------------------

    elif model_name == "pvn_original_stage1_bidomain":
        model = PVN_Stage1_Bidomain_Original(run_name, domain=domain); pytorch_model = True

    elif model_name == "pvn_stage1_bidomain":
        model = PVN_Stage1_Bidomain(run_name, domain=domain); pytorch_model = True

    elif model_name == "pvn_stage2_bidomain":
        model = PVN_Stage2_Bidomain(run_name, model_instance_name=domain); pytorch_model = True

    elif model_name == "pvn_stage2_actor_critic":
        model = PVN_Stage2_ActorCritic(run_name, model_instance_name=domain); pytorch_model = True

    elif model_name == "pvn_stage1_critic":
        model = PVN_Stage1_Critic(run_name); pytorch_model = True

    elif model_name == "pvn_stage1_critic_big":
        model = PVN_Stage1_Critic_Big(run_name); pytorch_model = True

    elif model_name == "pvn_full_bidomain":
        model = PVN_Wrapper_Bidomain(run_name, model_instance_name=domain, oracle_stage1=False); pytorch_model = True

    elif model_name == "pvn_full_bidomain_ground_truth":
        model = PVN_Wrapper_Bidomain(run_name, model_instance_name=domain, oracle_stage1=True); pytorch_model = True

    # -----------------------------------------------------------------------------------------------------------------
    # RSS 2020
    # -----------------------------------------------------------------------------------------------------------------

    elif model_name == "object_recognizer":
        model = ObjectRecognizer(run_name, domain=domain); pytorch_model = True

    elif model_name == "instance_recognizer":
        model = ModelFewShotInstanceRecognizer(run_name, domain=domain); pytorch_model = True

    elif model_name == "instance_recognizer_multiscale":
        model = ModelFewShotInstanceRecognizerMultiscale(run_name, domain=domain); pytorch_model = True

    elif model_name == "instance_recognizer_sliding_window":
        model = ModelFewShotInstanceRecognizerSlidingWindow(run_name, domain=domain); pytorch_model = True

    elif model_name == "object_reference_recognizer":
        model = ModelObjectReferenceRecognizer(run_name, domain=domain); pytorch_model = True

    elif model_name == "object_reference_recognizer_with_db":
        model = ModelObjectReferenceRecognizerWithDb(run_name, domain=domain); pytorch_model = True

    elif model_name == "simple_matching_network":
        model = ModelSimpleMatchingNetwork(run_name, domain=domain); pytorch_model = True

    elif model_name == "multi_matching_network":
        model = ModelMultiMatchingNetwork(run_name, domain=domain); pytorch_model = True

    elif model_name == "fspvn_stage1":
        model = FewShot_Stage1_Bidomain(run_name, domain=domain); pytorch_model = True

    elif model_name == "rpn_fspvn_stage1":
        model = RPN_FewShot_Stage1_Bidomain(run_name, domain=domain); pytorch_model = True

    elif model_name == "rpn_fspvn_stage2":
        # This is same as the wrapper - we use the wrapper to train stage 2
        # (because training stage 2 requires some stage 1 computations too)
        model = FS_PVN_Wrapper_Bidomain(run_name, model_instance_name=domain); pytorch_model = True

    elif model_name == "region_proposal_network":
        model = ModelFacebookRPNWrapper(run_name, domain=domain); pytorch_model = True

    elif model_name == "region_refinement_network":
        model = ModelRegionRefinementNetwork(run_name, domain=domain); pytorch_model = True

    elif model_name == "rpn_fspvn_full_bidomain":
        model = FS_PVN_Wrapper_Bidomain(run_name, model_instance_name=domain, oracle_stage1=False); pytorch_model = True

    elif model_name == "rpn_fspvn_full_bidomain_ground_truth":
        model = FS_PVN_Wrapper_Bidomain(run_name, model_instance_name=domain, oracle_stage1=True); pytorch_model = True


    # -----------------------------------------------------------------------------------------------------------------

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

        model = model.to(P.get("Setup::default_device"))

    return model, model_loaded
