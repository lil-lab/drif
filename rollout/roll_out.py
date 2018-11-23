import random

from data_io.env import load_path
from data_io.instructions import get_all_instructions
from data_io.instructions import get_word_to_token_map, merge_instruction_sets
from pomdp.pomdp_interface import PomdpInterface
from rollout.roll_out_params import SwitchThresholdStrategy, RolloutStrategy
from visualization import Presenter
import rollout.run_metadata as run_metadata
from learning.modules.dbg_writer import DebugWriter

from time import sleep
sleepytime = 1.0


class PolicyRoller:
    """
    Really only a wrapper around the roll_out_policy function, which does the policy rollout in the pomdp
    It collects actions both from the user-provided policy and from the oracle (as labels) and accumulates a dataset
    """
    def __init__(self, instance_id=0):
        self.presenter = Presenter()
        self.instance_id = instance_id
        self.env = None

        self.word2token = None
        self.all_instructions = None

    def reset(self):
        self.__init__()

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

    def roll_out_on_segment(self, ):
        pass

    def choose_action(self, params, step, switch_thres, reference_action, policy_action):
        """
        Choose whether to perform the policy action or the reference (oracle) action based on the type of mixture
        policy that is being executed
        :param params: RolloutParams instance
        :param step: current control step number
        :param switch_thres: roll-in/roll-out control step number
        :param reference_action: action executed by oracle
        :param policy_action: action executed by policy
        :return:
        """
        if params.rollout_strategy == RolloutStrategy.POLICY:
            return policy_action
        elif params.rollout_strategy == RolloutStrategy.REFERENCE:
            return reference_action
        elif params.rollout_strategy == RolloutStrategy.POLICY_IN_REF_OUT:
            if step > switch_thres:
                return reference_action
            else:
                return policy_action
        elif params.rollout_strategy == RolloutStrategy.MIXTURE:
            if random.uniform(0, 1) < params.mixture_ref_prob:
                return reference_action
            else:
                return policy_action

    def roll_out_on_env(self, params, instructions_set, set_idx, only_seg_idx=None, custom_instr=None):

        env_dataset = []
        failed = False

        env_id = instructions_set["env"]
        self.env.set_environment(env_id, instruction_set=instructions_set['instructions'])
        path = load_path(env_id)
        params.initPolicyContext(env_id, path)

        import rollout.run_metadata as md

        segments = list(instructions_set['instructions'])

        # all segments with at least length 2
        valid_segments = [(segments[i], i) for i in range(len(segments)) if segments[i]["end_idx"] - segments[i]["start_idx"] >= 2]

        if len(valid_segments) == 0:
            print ("Ding dong!")

        first_seg = True

        # For recurrent policy, we need to explicity start a segment and reset the LSTM state
        # TODO: Make sure this still works for the older non-NL model
        params.policy.start_sequence()

        for segment, seg_idx in valid_segments:
            if only_seg_idx is not None and seg_idx != only_seg_idx:
                print("Skipping seg: " + str(seg_idx) + " as not requested")
                continue

            if params.segment_level:
                params.policy.start_sequence()

            segment_dataset = []

            # Decide when to switch policies
            switch_threshold = params.horizon + 1 # Never switch policies by default
            do_switch = random.uniform(0, 1) < params.switch_prob
            if do_switch and params.threshold_strategy == SwitchThresholdStrategy.UNIFORM:
                switch_threshold = random.uniform(0, params.horizon)

            string_instruction, end_idx, start_idx = segment["instruction"], segment["end_idx"], segment["start_idx"]

            # Manual instruction override to allow rolling out arbitrary instructions for debugging
            if custom_instr is not None:
                print("REPLACED: ", string_instruction)
                string_instruction = custom_instr
            print("INSTRUCTION:", string_instruction)

            # Set some global parameters that can be accessed by the model and other parts of the system
            md.IS_ROLLOUT = True
            md.RUN_NAME = params.run_name
            md.ENV_ID = env_id
            md.SET_IDX = set_idx
            md.SEG_IDX = seg_idx
            md.START_IDX = start_idx
            md.END_IDX = end_idx
            md.INSTRUCTION = string_instruction

            if hasattr(params.policy, "start_segment_rollout"):
                params.policy.start_segment_rollout()

            token_instruction = self.tokenize_string(string_instruction)

            # At the end of segment N, should we reset drone position to the start of segment N+1 or continue
            # rolling out seamlessly?
            if first_seg or params.shouldResetAlways() or (failed and params.shouldResetIfFailed()):
                state = self.env.reset(seg_idx)
                #instr_str = debug_untokenize_instruction(instruction)
                #Presenter().show_instruction(string_instruction.replace("  ", " "))
                failed = False
                first_seg = False
                sleep(sleepytime)

            # Tell the oracle which part of the path is currently being executed
            params.setCurrentSegment(start_idx, end_idx)

            step_num = 0
            total_reward = 0
            # If the path has been finished according to the oracle, allow rolling out STEPS_TO_KILL more steps
            # If we finish the segment, but don't stop, log the position at which we finish the segment
            oracle_finished_countdown = params.steps_to_kill

            # Finally the actual policy roll out on the path segment!
            while True:

                # Get oracle action (labels)
                ref_action = params.ref_policy.get_action(state, token_instruction)

                if ref_action is None or step_num == params.horizon:
                    failed = True # Either veered off too far, or ran out of time. Either way, we consider it a fail
                    print("Failed segment")
                    break

                # Get the policy action (actions to be rolled out)
                action = params.policy.get_action(state, token_instruction)#, env_id=env_id)

                if action is None:
                    print("POLICY PRODUCED None ACTION")
                    break

                # Choose which action to execute (reference or policy) based on the selected procedure
                exec_action = self.choose_action(params, step_num, switch_threshold, ref_action, action)

                # action = [vel_x, vel_y, vel_yaw] vel_y is unused currently. Execute the action in the pomdp
                state, reward, done = self.env.step(exec_action)

                total_reward += reward

                # Collect the data into a dataset
                sample = {
                    "instruction": string_instruction,
                    "state": state,
                    "ref_action": ref_action,
                    "reward": reward,
                    "done": done,
                    "metadata": {
                        "seg_path": path[start_idx:end_idx + 1],
                        "path": path,
                        "env_id": env_id,
                        "set_idx": set_idx,
                        "seg_idx": seg_idx,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "action": exec_action,
                        "pol_action": action,
                        "ref_action": ref_action,
                        "instruction": string_instruction,
                        "flag": params.getFlag()
                    }
                }

                segment_dataset.append(sample)
                if not params.isSegmentLevel():
                    env_dataset.append(sample)

                # Do visual feedback and logging
                if params.first_person:
                    self.presenter.show_sample(state, exec_action, reward, string_instruction)
                if params.plot:
                    self.presenter.plot_paths(segment_dataset, interactive=True)
                if params.save_samples:
                    file_path = params.getSaveSamplesPath(env_id, set_idx, seg_idx, step_num)
                    self.presenter.save_sample(file_path, state, exec_action, reward, string_instruction)
                if params.show_action:
                    self.presenter.show_action(ref_action, "ref_action")
                    self.presenter.show_action(exec_action, "exec_action")

                # If the policy is finished, we stop. Otherwise the oracle should just keep outputing
                # examples that say that the policy should output finished at this point
                if exec_action[3] > 0.5 and not params.shouldIgnorePolicyStop():
                    print("Policy stop!")
                    break
                # If oracle says we're finished, allow a number of steps before terminating.
                if ref_action[3] > 0.5:
                    if oracle_finished_countdown == params.steps_to_kill:
                        drone_pos_force_stop = state.get_pos()
                    oracle_finished_countdown -= 1
                    if oracle_finished_countdown == 0:
                        print("Oracle forced stop!")
                        break
                step_num += 1

            # Call the rollout end callback, so that the model can save any debugging information, such as feature maps
            if callable(getattr(params.policy, "on_rollout_end", None)):
                params.policy.on_rollout_end(env_id, set_idx, seg_idx)

            if params.isSegmentLevel():
                env_dataset.append(segment_dataset)

            # Plot the trajectories for error tracking
            # TODO: Plot entire envs not segment by segment
            if params.save_plots:
                if not params.isSegmentLevel():
                    self.presenter.plot_paths(env_dataset, segment_path=path[start_idx:end_idx + 1], interactive=False, bg=True)
                self.presenter.save_plot(params.getSavePlotPath(env_id, set_idx, seg_idx))

            # Calculate end of segment error
            if end_idx > len(path) - 1:
                end_idx = len(path) - 1

            # The reward is proportional to path length. Weigh it down, so that max reward is 1:
            seg_len = end_idx - start_idx
            #self.error_tracker.add_sample(not failed, drone_pos_force_stop, state.get_pos(), path[end_idx],
            #                              path[end_idx - 1], total_reward, seg_len)

            if params.first_segment_only:
                print("Only running the first segment")
                break

            sleep(sleepytime)

        return env_dataset

    def roll_out_policy(self, params):
        """
        Given the provided rollout parameters, spawn a simulator instance and execute the specified policy on all
        environments specified in params.setEnvIds.

        Awful function that really needs to be simplified.
        A lot of the code is simply checking various error conditions, because the data has issues, and logging the outcome.
        The actual rollout is a very small part of the code.
        :param params: RollOutParams instance defining the parameters of the rollout
        :return: Aggregated dataset with images, states and oracle actions.
        If params.isSegmentLevel(), the returned dataset will be a list (over environments) of samples
        otherwise it will be a list (over environments) of lists (over segments) of samples
        """

        if params.isDebug():
            run_metadata.WRITE_DEBUG_DATA = True

        dataset = []
        try:
            # Load the neural network policy from file
            # We can't just pass a neural network into this function, because it can't be pickled
            params.loadPolicy()
            assert params.hasPolicy()

            self.env = PomdpInterface(instance_id=self.instance_id)

            all_env_ids, all_instructions, corpus, token2term, self.word2token = self.load_all_envs()
            env_ids = params.envs# if params.envs is not None else all_env_ids
            seg_indices = params.seg_list
            custom_instructions = params.custom_instructions

            # Filter out the envs that are not in all_instructions (we don't have instructions available for them)
            valid_env_ids = [i for i in env_ids if i in all_instructions]

            count = 0

            # Loop through environments
            for i, env_id in enumerate(valid_env_ids):

                print ("Rolling out on env: " + str(env_id))

                # Loop through all non-empty sets of instructions for each pomdp
                instruction_sets = [s for s in all_instructions[env_id] if len(s) > 0]

                if len(instruction_sets) == 0:
                    print("No instruction sets for env: " + str(env_id))

                for j, instructions_set in enumerate(instruction_sets):
                    count += 1
                    try:
                        seg_id = seg_indices[i] if seg_indices is not None else None
                        custom_instr = custom_instructions[i] if custom_instructions is not None else None
                        import rollout.run_metadata as md
                        md.CUSTOM_INSTR_NO = i
                        dataset += self.roll_out_on_env(params, instructions_set, j, seg_id, custom_instr)
                        #log("Path finished!")
                        DebugWriter().commit()
                    except Exception as e:
                        import traceback
                        from utils.colors import print_error
                        print_error("Error encountered during policy rollout!")
                        print_error(e)
                        print_error(traceback.format_exc())
                        continue

        except Exception as e:
            import traceback
            from utils.colors import print_error
            print_error("Error encountered during policy rollout!")
            print_error(e)
            print_error(traceback.format_exc())

        return dataset
