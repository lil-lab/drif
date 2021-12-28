import itertools
from data_io.instructions import get_segs_available_for_env


class SimpleRolloutBase:
    """
    Really only a wrapper around the roll_out_policy function, which does the policy rollout in the pomdp
    It collects actions both from the user-provided policy and from the oracle (as labels) and accumulates a dataset
    """
    def __init__(self, *args, **kwargs):
        ...

    def single_segment_rollout(self, env_id, set_idx, seg_idx, do_sample, dagger_beta=0, rl_rollout=False):
        ...

    def rollout_segments(self, env_ids, seg_ids, policy_state=None, sample=False, dagger_beta=0, land_afterwards=False, rl_rollout=False):
        ...

    def rollout_envs(self, env_ids, policy_state=None, sample=False, dagger_beta=0, land_afterwards=False, rl_rollout=False):
        seg_lists = [get_segs_available_for_env(e, 0) for e in env_ids]
        env_ids = [[e] * len(segs) for e, segs in zip(env_ids, seg_lists)]
        round_segs = list(itertools.chain(*seg_lists))
        round_envs = list(itertools.chain(*env_ids))
        return self.rollout_segments(round_envs, round_segs, policy_state, sample, dagger_beta, land_afterwards, rl_rollout)