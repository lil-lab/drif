import os
import ray
import itertools

from time import sleep
from rollout.simple_rollout import SimplePolicyRoller
from rollout.simple_rollout_base import SimpleRolloutBase

import parameters.parameter_server as P


@ray.remote(num_gpus=0.1, num_cpus=1)
class RolloutActor():
    def __init__(self, params_file, instance_id, save_dataset_name, policy, oracle, device, no_reward):
        # This doesn't carry over to subprocess, so have to re-load the params from json
        P.initialize_experiment(params_file)
        self.instance_id = instance_id
        self.save_dataset_name = save_dataset_name
        self.oracle = oracle
        self.device = device
        self.no_reward = no_reward

        self.current_policy = policy
        self.policy_roller = self._make_policy_roller()

    def _make_policy_roller(self):
        return SimplePolicyRoller(instance_id=self.instance_id,
                                  policy=self.current_policy,
                                  oracle=self.oracle,
                                  dataset_save_name=self.save_dataset_name,
                                  no_reward=self.no_reward)

    # This is a hack. Try to not look at it
    def restart(self):
        del self.policy_roller
        os.system("killall -9 MyProject5-Linux-Shipping")
        sleep(2)
        self.policy_roller = self._make_policy_roller()

    def rollout(self, env_ids, seg_ids, policy_state, sample, dagger_beta, rl_rollout):
        result = self.policy_roller.rollout_segments(env_ids, seg_ids, policy_state, sample, dagger_beta, rl_rollout=rl_rollout)
        return result

    def update_static_policy_state(self, static_policy_state):
        self.current_policy.set_static_state(static_policy_state)
        self.policy_roller.set_policy(self.current_policy)


class SimpleParallelPolicyRoller(SimpleRolloutBase):

    def __init__(self, policy, num_workers,
                 oracle=None, device=None, dataset_save_name="", restart_every_n=1000,
                 no_reward=False):
        super().__init__()
        self.num_workers = num_workers
        self.actors = []
        self.shared_policy = policy
        if hasattr(self.shared_policy, "make_picklable"):
            self.shared_policy.make_picklable()
        if device:
            self.shared_policy = self.shared_policy.to(device)
        self.device = device
        self.dataset_save_name = dataset_save_name
        self.restart_every_n = restart_every_n
        self.rollout_num = 0
        self.no_reward = no_reward

        for i in range(self.num_workers):
            actor = RolloutActor.remote(P.get_setup_name(), i, dataset_save_name, self.shared_policy, oracle, device, no_reward)
            self.actors.append(actor)

    def _split_list_for_workers(self, env_ids, seg_ids):
        env_id_set = set(env_ids)
        env_ids_split = [[] for _ in range(self.num_workers)]
        seg_ids_split = [[] for _ in range(self.num_workers)]
        for i, env_id in enumerate(env_id_set):
            for e, s in zip(env_ids, seg_ids):
                if e == env_id:
                    env_ids_split[i % self.num_workers].append(e)
                    seg_ids_split[i % self.num_workers].append(s)
        return env_ids_split, seg_ids_split

    def update_static_policy_state(self, static_state):
        for actor in self.actors:
            actor.update_static_policy_state.remote(static_state)

    def rollout_segments(self, env_ids, seg_ids, policy_state=None, sample=False, dagger_beta=0, land_afterwards=False, rl_rollout=True):
        self.rollout_num += int(len(env_ids) / self.num_workers) + 1
        # Restart AirSim every N rollouts. This is needed because the workers for some reason
        # tend to get slower and slower as if there was a memory leak or something
        if self.rollout_num % self.restart_every_n == 0:
            for actor in self.actors:
                actor.restart.remote()

        # Split env_ids and seg_ids such that all seg ids on the same env are running on the same worker
        env_ids_split, seg_ids_split = self._split_list_for_workers(env_ids, seg_ids)

        # Run rollout on each worker
        result_objects = [self.actors[i].rollout.remote(env_ids_split[i], seg_ids_split[i], policy_state, sample, dagger_beta, rl_rollout)
                          for i in range(self.num_workers)]

        # Retrieve the results and join all the rollouts into a single list of rollouts
        results = ray.get(result_objects)
        results = list(itertools.chain(*results))

        return results
