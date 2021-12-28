import threading
import numpy as np
import torch

from pykeyboard import PyKeyboardEvent

from drones.airsim_interface.rate import Rate
from data_io.instructions import get_all_instructions, tokenize_instruction, get_word_to_token_map, debug_untokenize_instruction
from data_io.models import load_model
from pomdp.pomdp_interface import PomdpInterface
from visualization import Presenter
from learning.inputs.common import cuda_var


from mains.interactive.interact_ui import InteractAPI

from parameters.parameter_server import initialize_experiment, get_current_parameters
import parameters.parameter_server as P

import cv2

def rollout_model(model, env, env_id, set_id, seg_id, tok_instruction):
    state = env.reset(seg_id)
    while True:
        action = model.get_action(state, tok_instruction)
        state, reward, done = env.step(action)
        if action[3] > 0.5:
            print("Done!")
            break
    return True

def interactive_demo():

    P.initialize_experiment()
    InteractAPI.launch_ui()

    rate = Rate(0.1)

    env = PomdpInterface(is_real=get_current_parameters()["Setup"]["real_drone"])
    train_instructions, dev_instructions, test_instructions, corpus = get_all_instructions()
    all_instr = {**train_instructions, **dev_instructions, **train_instructions}
    token2term, word2token = get_word_to_token_map(corpus)

    # Run on dev set
    interact_instructions = dev_instructions

    env_range_start = get_current_parameters()["Setup"].get("env_range_start", 0)
    env_range_end = get_current_parameters()["Setup"].get("env_range_end", 10e10)
    interact_instructions = {k:v for k,v in interact_instructions.items() if env_range_start < k < env_range_end}

    count = 0
    stuck_count = 0

    model, _ = load_model(get_current_parameters()["Setup"]["model"])

    InteractAPI.write_empty_instruction()
    InteractAPI.write_real_instruction("None")
    instruction_str = InteractAPI.read_instruction_file()
    print("Initial instruction: ", instruction_str)

    for instruction_sets in interact_instructions.values():
        for set_idx, instruction_set in enumerate(instruction_sets):
            env_id = instruction_set['env']
            env.set_environment(env_id, instruction_set["instructions"])

            presenter = Presenter()
            cumulative_reward = 0
            for seg_idx in range(len(instruction_set["instructions"])):

                print(f"RUNNING ENV {env_id} SEG {seg_idx}")

                real_instruction_str = instruction_set["instructions"][seg_idx]["instruction"]
                InteractAPI.write_real_instruction(real_instruction_str)
                valid_segment = env.set_current_segment(seg_idx)
                if not valid_segment:
                    continue
                state = env.reset(seg_idx)

                keep_going = True
                while keep_going:
                    InteractAPI.write_real_instruction(real_instruction_str)

                    while True:
                        cv2.waitKey(200)
                        instruction = InteractAPI.read_instruction_file()
                        if instruction == "CMD: Next":
                            print("Advancing")
                            keep_going = False
                            InteractAPI.write_empty_instruction()
                            break
                        elif instruction == "CMD: Reset":
                            print("Resetting")
                            env.reset(seg_idx)
                            InteractAPI.write_empty_instruction()
                        elif len(instruction.split(" ")) > 1:
                            instruction_str = instruction
                            break

                    if not keep_going:
                        continue

                    env.override_instruction(instruction_str)
                    tok_instruction = tokenize_instruction(instruction_str, word2token)

                    state = env.reset(seg_idx)
                    print("Executing: f{instruction_str}")
                    while True:
                        rate.sleep()
                        action = model.get_action(state, tok_instruction)

                        state, reward, done, expired = env.step(action)
                        cumulative_reward += reward
                        presenter.show_sample(state, action, reward, cumulative_reward, instruction_str)
                        #show_depth(state.image)
                        if done:
                            break
                    InteractAPI.write_empty_instruction()
                    print ("Segment finished!")
        print("Env finished!")

if __name__ == "__main__":
    interactive_demo()