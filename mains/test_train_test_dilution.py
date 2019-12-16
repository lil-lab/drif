from data_io.instructions import get_restricted_env_id_lists, get_all_instructions
import parameters.parameter_server as P

if __name__ == "__main__":
    P.initialize_experiment()
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()
    train_i, dev_i, test_i, corpus = get_all_instructions()

    train_i_envs = set([int(i) for i in train_i.keys()])

    for test_env in test_envs:
        assert test_env not in train_i_envs, "FAIL"

    for dev_env in test_envs:
        assert dev_env not in train_i_envs, "FAIL"

    print("OK")