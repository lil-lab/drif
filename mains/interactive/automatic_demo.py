import threading
import numpy as np
import torch
import time

from pykeyboard import PyKeyboardEvent

from drones.airsim_interface.rate import Rate
from data_io.instructions import get_all_instructions, tokenize_instruction, get_word_to_token_map, debug_untokenize_instruction
from data_io.models import load_model
from pomdp.pomdp_interface import PomdpInterface
from visualization import Presenter
from learning.inputs.common import cuda_var

from mains.interactive.instruction_display import InstructionDisplay

from parameters.parameter_server import initialize_experiment, get_current_parameters
import parameters.parameter_server as P

import cv2

START_PAUSE = 5
END_PAUSE = 3


def rollout_model(model, env, env_id, set_id, seg_id, tok_instruction):
    state = env.reset(seg_id)
    while True:
        action = model.get_action(state, tok_instruction)
        state, reward, done = env.step(action)
        if action[3] > 0.5:
            print("Done!")
            break
    return True


examples = [
    [6827, 0, 4],
    [6825, 0, 8],
]


def automatic_demo():

    P.initialize_experiment()
    instruction_display = InstructionDisplay()

    rate = Rate(0.1)

    env = PomdpInterface(is_real=get_current_parameters()["Setup"]["real_drone"])
    train_instructions, dev_instructions, test_instructions, corpus = get_all_instructions()
    all_instr = {**train_instructions, **dev_instructions, **train_instructions}
    token2term, word2token = get_word_to_token_map(corpus)

    # Run on dev set
    interact_instructions = dev_instructions

    env_range_start = get_current_parameters()["Setup"].get("env_range_start", 0)
    env_range_end = get_current_parameters()["Setup"].get("env_range_end", 10e10)
    interact_instructions = {k: v for k, v in interact_instructions.items() if env_range_start < k < env_range_end}

    model, _ = load_model(get_current_parameters()["Setup"]["model"])

    # Loop over the select few examples
    while True:

        for instruction_sets in interact_instructions.values():
            for set_idx, instruction_set in enumerate(instruction_sets):
                env_id = instruction_set['env']
                found_example = None
                for example in examples:
                    if example[0] == env_id:
                        found_example = example
                if found_example is None:
                    continue
                env.set_environment(env_id, instruction_set["instructions"])

                presenter = Presenter()
                cumulative_reward = 0
                for seg_idx in range(len(instruction_set["instructions"])):
                    if seg_idx != found_example[2]:
                        continue

                    print(f"RUNNING ENV {env_id} SEG {seg_idx}")

                    real_instruction_str = instruction_set["instructions"][seg_idx]["instruction"]
                    instruction_display.show_instruction(real_instruction_str)
                    valid_segment = env.set_current_segment(seg_idx)
                    if not valid_segment:
                        continue
                    state = env.reset(seg_idx)

                    for i in range(START_PAUSE):
                        instruction_display.tick()
                        time.sleep(1)

                        tok_instruction = tokenize_instruction(real_instruction_str, word2token)

                    state = env.reset(seg_idx)
                    print("Executing: f{instruction_str}")
                    while True:
                        instruction_display.tick()
                        rate.sleep()
                        action, internals = model.get_action(state, tok_instruction)
                        state, reward, done, expired, oob = env.step(action)
                        cumulative_reward += reward
                        #presenter.show_sample(state, action, reward, cumulative_reward, real_instruction_str)
                        #show_depth(state.image)
                        if done:
                            break

                    for i in range(END_PAUSE):
                        instruction_display.tick()
                        time.sleep(1)
                        print ("Segment finished!")
                    instruction_display.show_instruction("...")

            print("Env finished!")


if __name__ == "__main__":
    automatic_demo()
