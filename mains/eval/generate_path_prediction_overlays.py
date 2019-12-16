from learning.training.trainer_supervised_bidomain import TrainerBidomain
from data_io.models import load_model
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.instructions import get_restricted_env_id_lists

import parameters.parameter_server as P


# Supervised learning parameters
def train_supervised_bidomain():
    P.initialize_experiment()

    setup = P.get_current_parameters()["Setup"]
    model_sim, _ = load_model(setup["model"], setup["sim_model_file"], domain="sim")
    model_real, _ = load_model(setup["model"], setup["real_model_file"], domain="real")
    model_critic, _ = load_model(setup["critic_model"], setup["critic_model_file"])

    print("Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    env_list_name = setup.get("eval_env_set", "dev")
    if env_list_name == "dev":
        print("Using DEV envs")
        use_envs = dev_envs
    elif env_list_name == "train":
        print("Using TRAIN envs")
        use_envs = train_envs
    elif env_list_name == "test":
        print("Using TEST envs")
        use_envs = test_envs
    else:
        raise ValueError(f"Unknown env set {env_list_name}")

    env_range_start = setup.get("env_range_start")
    if env_range_start > 0:
        use_envs = [e for e in use_envs if e >= env_range_start]
    env_range_end = setup.get("env_range_end")
    if env_range_end > 0:
        use_envs = [e for e in use_envs if e < env_range_end]

    restricted_domain = "simulator"
    if restricted_domain == "simulator":
        # Load dummy model for real domain
        model_real, _ = load_model(setup["model"], setup["sim_model_file"], domain="sim")
        model_sim.set_save_path_overlays(True)
    elif restricted_domain == "real":
        # Load dummy model for sim domain
        model_sim, _ = load_model(setup["model"], setup["real_model_file"], domain="real")
        model_real.set_save_path_overlays(True)
    else:
        model_real.set_save_path_overlays(True)
        model_sim.set_save_path_overlays(True)

    trainer = TrainerBidomain(model_real, model_sim, model_critic, epoch=0)
    trainer.train_epoch(env_list=use_envs, eval=True, restricted_domain=restricted_domain)

    if restricted_domain != "simulator":
        model_real.print_metrics()
    if restricted_domain != "real":
        model_sim.print_metrics()

if __name__ == "__main__":
    train_supervised_bidomain()
