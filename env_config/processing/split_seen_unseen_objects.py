from env_config.definitions.landmarks import LANDMARK_RADII, PORTABLE_LANDMARK_RADII

from data_io.env import load_env_config
from data_io.instructions import get_all_instructions, get_restricted_env_id_lists

import parameters.parameter_server as P

UNSEEN_OBJECTS = [
    "TrafficCone",
    "Boat",
    "Box",
    "Stump",
    "Tombstone",
    "Container",
]

# All the objects that appear in simulation and not in real world are considered previously seen objects
SEEN_OBJECTS = {l: r for l, r in LANDMARK_RADII.items() if l not in UNSEEN_OBJECTS}


def split_seen_unseen_objects():
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()
    env_list = train_envs
    envs_with_unseen_objects = []
    envs_without_unseen_objects = []
    envs_with_seen_objects = []
    envs_without_seen_objects = []
    for env_id in env_list:
        config = load_env_config(env_id)
        has_unseen_object = False
        has_seen_object = False
        for lmname in config["landmarkName"]:
            if lmname in UNSEEN_OBJECTS:
                has_unseen_object = True
            else:
                has_seen_object = True
        if has_unseen_object:
            envs_with_unseen_objects.append(env_id)
        else:
            envs_without_unseen_objects.append(env_id)
        if has_seen_object:
            envs_with_seen_objects.append(env_id)
        else:
            envs_without_seen_objects.append(env_id)

    print(f"Found {len(envs_with_unseen_objects)} envs with unseen objects.")
    print(f"Found {len(envs_without_unseen_objects)} envs without unseen bjects.")
    print(f"Found {len(envs_with_seen_objects)} envs with seen objects.")
    print(f"Found {len(envs_without_seen_objects)} envs without seen objects.")

    real_envs_without_unseen_objects = [e for e in envs_without_unseen_objects if e >= 6000]
    print(f"Found {len(real_envs_without_unseen_objects)} REAL envs without unseen objects.")
    real_envs_without_seen_objects = [e for e in envs_without_seen_objects if e >= 6000]
    print(f"Found {len(real_envs_without_seen_objects)} REAL envs without seen objects.")


if __name__ == "__main__":
    P.initialize_experiment()
    split_seen_unseen_objects()