import os
import gc
from time import sleep
from data_io.models import load_model
from drones.airsim_interface.droneController import spawn_workers

import multiprocessing as mp
from rollout.simple_rollout import SimplePolicyRoller

import parameters.parameter_server as P


def worker_process(conn, instance_id, save_dataset_name, policy, oracle, device, no_reward):
    # This doesn't carry over to subprocess, so have to re-load the params from json
    P.initialize_experiment()
    policy_roller = SimplePolicyRoller(instance_id=instance_id, policy=policy, oracle=oracle, no_reward=no_reward)
    current_policy = policy
    while True:
        msg, payload = conn.recv()

        if msg == "Stop":
            break

        elif msg == "KillSim":
            print(f"  RECV: {msg}")
            del policy_roller
            os.system("killall -9 MyProject5-Linux-Shipping")
            policy_roller = None

        elif msg == "Restart":
            print(f"  RECV: {msg}")
            del policy_roller
            os.system("killall -9 MyProject5-Linux-Shipping")
            sleep(2)
            policy_roller = SimplePolicyRoller(instance_id=instance_id, policy=current_policy)

        elif msg == "ReloadStaticState":
            print(f"  RECV: {msg}")
            print(f"payload.device: {next(payload.parameters()).device}")
            # TODO: This should be general and not know anything about model details:
            current_policy.stage1_visitation_prediction = payload
            policy_roller.set_policy(current_policy)

        elif msg == "Rollout":
            if policy_roller is None:
                policy_roller = SimplePolicyRoller(instance_id=instance_id, policy=current_policy, no_reward=no_reward)
            env_ids, seg_ids, policy_state, sample, dagger_beta = payload
            result = policy_roller.rollout_segments(env_ids, seg_ids, policy_state, sample, dagger_beta, save_dataset_name)
            conn.send(result)
        else:
            raise ValueError(f"Unrecognized worker task message: {msg}")
    conn.close()


class SimpleParallelPolicyRoller():

    def __init__(self, policy_name, policy_file, num_workers,
                 oracle=None, device=None, dataset_save_name="", restart_every_n=1000,
                 no_reward=False):
        self.num_workers = num_workers
        self.processes = []
        self.connections = []
        self.policy_name = policy_name
        self.shared_policy, _ = load_model(policy_name, policy_file)
        self.shared_policy.make_picklable()
        self.shared_policy = self.shared_policy.to(device)
        self.device = device
        self.dataset_save_name = dataset_save_name
        self.restart_every_n = restart_every_n
        self.rollout_num = 0
        self.no_reward = no_reward

        for i in range(self.num_workers):
            ctx = mp.get_context("spawn")
            parent_conn, child_conn = ctx.Pipe()
            print(f"LAUNCHING WORKER {i}")
            p = ctx.Process(target=worker_process, args=(child_conn, i, dataset_save_name, self.shared_policy, oracle, device, no_reward))
            self.processes.append(p)
            self.connections.append(parent_conn)
            p.start()

    def __enter__(self):
        return self

    def __exit__(self):
        for i in range(self.num_workers):
            self.connections[i].send(["Stop", None])
        for i in range(self.num_workers):
            self.processes[i].join()

    def _split_list_for_workers(self, lst):
        n_each = int((len(lst)  + self.num_workers - 1) / self.num_workers)
        split = []
        for i in range(0, len(lst), n_each):
            split.append(lst[i:i+n_each])
        return split

    def update_stage1_on_workers(self, stage1_module):
        stage1_module.make_picklable()
        self.shared_policy.stage1_visitation_prediction = stage1_module
        for i in range(self.num_workers):
            print(f"  Reloading stage 1 on worker {i}")
            self.connections[i].send(["ReloadStaticState", stage1_module])

    def kill_airsim(self):
        for i in range(self.num_workers):
            self.connections[i].send(["KillSim", None])

    def rollout_segments(self, env_ids, seg_idx, policy_state, sample, dagger_beta=0):

        self.rollout_num += int(len(env_ids) / self.num_workers) + 1
        # Restart AirSim every N rollouts. This is needed because the workers for some reason
        # tend to get slower and slower as if there was a memory leak or something
        if self.rollout_num % self.restart_every_n == 0:
            for i in range(self.num_workers):
                self.connections[i].send(["Restart", None])

        # Split env_ids and seg_ids
        env_ids_split = self._split_list_for_workers(env_ids)
        seg_ids_split = self._split_list_for_workers(seg_idx)

        # Send to each of the worker tasks
        for i in range(self.num_workers):
            self.connections[i].send(["Rollout", [env_ids_split[i], seg_ids_split[i], policy_state, sample, dagger_beta]])

        # Collect and return results
        results = []
        for i in range(self.num_workers):
            result = self.connections[i].recv()
            results += result

        print("SimpleParallelRollout: Finished N Rollouts")
        # Keep worker tasks alive.
        return results
