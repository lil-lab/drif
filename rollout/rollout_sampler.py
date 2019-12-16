import random
from data_io.instructions import get_all_instructions, get_word_to_token_map, get_restricted_env_id_lists
from rollout.simple_rollout import SimplePolicyRoller

from data_io.models import load_model

import parameters.parameter_server as P

class RolloutSampler():

    def __init__(self, simple_roller):
        self.roller = simple_roller
        self.train_i, self.dev_i, test_i, self.corpus = get_all_instructions()
        train_envs, dev_envs, test_envs = get_restricted_env_id_lists()
        self.train_i = {k:v for k,v in self.train_i.items() if k in train_envs}
        self.dev_i = {k:v for k,v in self.dev_i.items() if k in dev_envs}
        self.token2term, self.word2token = get_word_to_token_map(self.corpus)
        self.only_seg_0 = P.get_current_parameters()["Setup"].get("seg_0_only", False)
        self.train_env_and_seg_ids = self._init_env_and_seg_ids(self.train_i)
        self.dev_env_and_seg_ids = self._init_env_and_seg_ids(self.dev_i)

    def _init_env_and_seg_ids(self, instructions):
        ids = []
        num_invalid = 0
        for k, instr_sets in instructions.items():
            instr_set = instr_sets[0]
            env_id = instr_set["env"]
            for i, instr in enumerate(instr_set["instructions"]):
                valid_segment = True
                if instr["end_idx"] - instr["start_idx"] < 2:
                    valid_segment = False
                if valid_segment:
                    if self.only_seg_0:
                        ids.append([env_id, 0])
                    else:
                        ids.append([env_id, instr["seg_idx"]])
                else:
                    num_invalid += 1

        print(f"Counted {num_invalid} invalid segments and {len(ids)} valid ones")
        return ids

    def update_stage1_on_workers(self, module):
        self.roller.update_stage1_on_workers(module)

    def sample_n_rollouts(self, n, policy_state, sample=True, envs="train", dagger_beta=0):
        if n == 0:
            return []

        if envs == "train":
            use_envs = self.train_env_and_seg_ids
        elif envs == "dev":
            use_envs = self.dev_env_and_seg_ids
        else:
            raise ValueError("Unrecognized envs: {envs}")

        if len(use_envs) < n:
            use_envs = use_envs * int((n + len(use_envs) - 1) / len(use_envs))

        pick_env_and_seg_ids = random.sample(use_envs, n)
        env_ids, seg_ids = tuple(zip(*pick_env_and_seg_ids))
        return self.roller.rollout_segments(env_ids, seg_ids, policy_state, sample, dagger_beta)

