import random
from data_io.paths import get_human_eval_envs_path
from data_io.helpers import save_json
from data_io.instructions import get_restricted_env_id_lists

NUM_ENVS = 30


def sample_human_envs():
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()
    random.shuffle(test_envs)
    human_envs = test_envs[:NUM_ENVS]
    human_envs = sorted(human_envs)
    save_json(human_envs, get_human_eval_envs_path())


if __name__ == "__main__":
    sample_human_envs()