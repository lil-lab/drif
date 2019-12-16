import multiprocessing as mp
from data_io.models import load_model
from rollout.rollout_sampler import RolloutSampler
from rollout.simple_parallel_rollout import SimpleParallelPolicyRoller

import parameters.parameter_server as P

def test_rollout_sampler():
    policy, _ = load_model("pvn_full_bidomain")
    policy_state = policy.get_policy_state()
    from visualization import Presenter

    #roller = SimplePolicyRoller(policy_factory)
    roller =  SimpleParallelPolicyRoller("pvn_full_bidomain", num_workers=4)
    rollout_sampler = RolloutSampler(roller)

    # TODO: Load some policy
    print("Sampling once")
    rollouts = rollout_sampler.sample_n_rollouts(12, policy_state)

    print("Sampling twice")
    rollouts += rollout_sampler.sample_n_rollouts(12, policy_state)

    print("Sampling thrice")
    rollouts += rollout_sampler.sample_n_rollouts(12, policy_state)

    for rollout in rollouts:
        print("Visualizing rollout")
        for sample in rollout:
            state = sample["state"]
            image = state.get_rgb_image()
            Presenter().show_image(image, "fpv", waitkey=True, scale=4)
        print("Done!")

    roller.__exit__()
    print("ding")


if __name__ == "__main__":
    P.initialize_experiment()
    mp.set_start_method('spawn')
    test_rollout_sampler()