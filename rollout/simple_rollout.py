from data_io.instructions import get_all_instructions
from data_io.instructions import get_word_to_token_map, merge_instruction_sets, get_instruction_segment
from data_io.train_data import save_dataset, save_metadata

from pomdp.pomdp_interface import PomdpInterface
from visualization import Presenter
import random
import types

import rollout.run_metadata as md

from utils.dict_tools import dict_merge


def prune_sample(s):
    pruned_sample = {
        "instruction": s["instruction"],
        "ref_action": s["ref_action"],
        "pol_action": s["pol_action"],
        "action": s["action"],
        "state": s["state"],
        "extrinsic_reward": s["extrinsic_reward"],
        "intrinsic_reward": s["intrinsic_reward"],
        "full_reward": s["full_reward"],
        "done": s["done"],
        "expired": s["expired"],
        "env_id": s["env_id"],
        "set_idx": s["set_idx"],
        "seg_idx": s["seg_idx"]
    }
    return pruned_sample


class SimplePolicyRoller:
    """
    Really only a wrapper around the roll_out_policy function, which does the policy rollout in the pomdp
    It collects actions both from the user-provided policy and from the oracle (as labels) and accumulates a dataset
    """
    def __init__(self, instance_id=0, real_drone=False, policy=None, oracle=None, no_reward=False):

        self.presenter = Presenter()
        self.instance_id = instance_id

        self.word2token = None
        self.all_instructions = None
        self.all_env_ids, self.all_instructions, self.corpus, self.token2term, self.word2token = self.load_all_envs()

        self.env = PomdpInterface(instance_id=self.instance_id, is_real=real_drone)
        self.policy = policy
        self.oracle = oracle
        self.no_reward = no_reward

    def load_all_envs(self):
        train_i, dev_i, test_i, corpus = get_all_instructions()
        all_instructions = merge_instruction_sets(train_i, dev_i, test_i)
        token2term, word2token = get_word_to_token_map(corpus)
        env_ids = list(all_instructions.keys())
        return env_ids, all_instructions, corpus, token2term, word2token

    def tokenize_string(self, s):
        word_list = filter(None, s.split(" "))
        token_instruction = list(map(lambda w: self.word2token[w], word_list))
        return token_instruction

    def set_policy(self, policy):
        self.policy = policy

    def save_rollouts(self, rollouts, dataset_name):
        env_rollouts = {}
        for rollout in rollouts:
            env_id = rollout[0]["env_id"]
            if env_id not in env_rollouts:
                env_rollouts[env_id] = []
            env_rollouts[env_id] += rollout

        for env_id, rollouts in env_rollouts.items():
            # This saves just a single segment per environment, as opposed to all segments that the oracle saves. Problem?
            if len(rollouts) > 0:
                #pruned_rollouts = [prune_sample(s) for s in rollouts]
                save_dataset(dataset_name, rollouts, env_id=env_id, lock=True)
                #save_metadata(dataset_name, env_id, {"seg_ids": segments})

    def choose_action(self, pol_action, ref_action, dagger_beta):
        use_expert = random.uniform(0,1) < dagger_beta
        if use_expert:
            return ref_action
        else:
            return pol_action

    def sample_to_cpu(self, sample):
        for k, v in sample.items():
            if hasattr(v, "to") and isinstance(v.to, types.MethodType):
                sample[k] = v.to("cpu")

    def single_segment_rollout(self, env_id, set_idx, seg_idx, do_sample, dagger_beta=0, rl_rollout=True):
        instruction_sets = self.all_instructions[env_id][set_idx]['instructions']
        for instruction_set in instruction_sets:
            if instruction_set["seg_idx"] == seg_idx:
                break

        # TODO: Get rid of this idiocy:
        md.IS_ROLLOUT = True

        instruction_set = get_instruction_segment(env_id, set_idx, seg_idx, all_instr=self.all_instructions)

        self.env.set_environment(env_id, instruction_set=instruction_sets, fast=True)
        self.env.set_current_segment(seg_idx)

        self.policy.start_sequence()
        if hasattr(self.policy, "start_segment_rollout"):
            self.policy.start_segment_rollout(env_id, set_idx, seg_idx)
        if self.oracle:
            self.oracle.start_segment_rollout(env_id, set_idx, seg_idx)

        string_instruction, end_idx, start_idx = instruction_set["instruction"], instruction_set["end_idx"], instruction_set["start_idx"]
        token_instruction = self.tokenize_string(string_instruction)

        # TODO: Support oracle (including setCurrentSegment, and setting the path)
        rollout_sample = []

        # Reset the drone to the segment starting position:
        state = self.env.reset(seg_idx)

        first = True
        while True:
            action, rl_stuff = self.policy.get_action(state, token_instruction, sample=do_sample, rl_rollout=rl_rollout)

            if self.oracle:
                ref_action, _ = self.oracle.get_action(state, token_instruction)
                exec_action = self.choose_action(action, ref_action, dagger_beta)
            else:
                ref_action = action
                exec_action = action

            next_state, extrinsic_reward, done, expired, oob = self.env.step(exec_action)

            # Calculate intrinsic reward (I don't like that this delays the loop)
            if hasattr(self.policy, "calc_intrinsic_rewards") and not self.no_reward:
                intrinsic_rewards = self.policy.calc_intrinsic_rewards(next_state, action, done, first)
            else:
                intrinsic_rewards = {"x": 0}
            intrinsic_reward = sum(intrinsic_rewards.values())

            sample = {
                "instruction": string_instruction,
                "ref_action": ref_action,
                "pol_action": action,
                "action": exec_action,
                "state": state,
                "extrinsic_reward": extrinsic_reward,
                "intrinsic_reward": intrinsic_reward - (1.0 if oob else 0.0),
                "full_reward": extrinsic_reward + intrinsic_reward,
                "done": done,
                "expired": expired,
                "env_id": env_id,
                "set_idx": set_idx,
                "seg_idx": seg_idx,
            }
            sample = dict_merge(sample, rl_stuff)
            if not self.no_reward:
                sample = dict_merge(sample, intrinsic_rewards)
            rollout_sample.append(sample)

            # Multiprocessing has stopped playing nice with PyTorch cuda. Move sample to cpu first.
            if rl_rollout:
                self.sample_to_cpu(sample)

            state = next_state
            first = False
            if done:
                #print(f"Done! Last action: {exec_action}")
                break

        md.IS_ROLLOUT = False
        # Add discounted returns
        return rollout_sample

    def rollout_segments(self, env_ids, seg_ids, policy_state, sample, dagger_beta=0, save_dataset_name=None, land_afterwards=False, rl_rollout=True):
        if policy_state is not None:
            self.policy.set_policy_state(policy_state)

        data = []
        for env_id, seg_idx in zip(env_ids, seg_ids):
            done = False
            while not done:
                try:
                    seg_data = self.single_segment_rollout(env_id, 0, seg_idx, sample, dagger_beta, rl_rollout)
                    done = True
                except PomdpInterface.EnvException as e:
                    continue
            data.append(seg_data)

        if save_dataset_name:
            self.save_rollouts(data, save_dataset_name)

        # Land the real drone if we have one.
        if land_afterwards:
            self.env.land()

        return data
