import random
from data_io.instructions import get_restricted_env_id_lists
import parameters.parameter_server as P

#env_list = "DEV"
env_list = "TEST"

num_env_groups = 10
envs_each_group = 2


def sample_real_data_subset():
    global env_list, num_env_groups
    P.initialize_experiment()
    if env_list == "DEV":
        train_i, dev_i, test_i = get_restricted_env_id_lists()
        env_list = dev_i
    elif env_list == "TEST":
        train_i, dev_i, test_i = get_restricted_env_id_lists()
        env_list = test_i

    # Each 5 subsequent environments are the same. First sample groups, then sample environments
    groups = set()
    for env in env_list:
        groups.add(int(env/5))

    groups = list(groups)
    group_envs_rel = {}
    pick_groups = random.sample(groups, num_env_groups)
    for group in pick_groups:
        group_envs_rel[group] = []
        i = 0
        while i < envs_each_group:
            rint = random.randint(0,4)
            if rint not in group_envs_rel[group]:
                group_envs_rel[group].append(rint)
                i += 1
            else:
                # Retry this loop iteration
                continue

    env_ids_out = []
    for group, env_rels in group_envs_rel.items():
        for env_rel in env_rels:
            env_id = group * 5 + env_rel
            env_ids_out.append(env_id)

    print(f"Sampled {len(env_ids_out)} envs:")
    print(list(sorted(env_ids_out)))


if __name__ == "__main__":
    sample_real_data_subset()
