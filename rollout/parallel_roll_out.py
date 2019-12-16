from collections import namedtuple
import multiprocessing as mp
from data_io.paths import get_plots_dir, get_samples_dir

#try:
#    mp.set_start_method('spawn')
#except RuntimeError:
#    pass

import itertools
import copy
from rollout.roll_out import PolicyRoller

Sample = namedtuple("Sample", ("instruction", "state", "action", "reward", "done", "metadata"))


def exec_roll_out_policy(arguments):
    worker_num = arguments[0]
    param_list = arguments[1]
    results = []
    for i, params in enumerate(param_list):
        params.loadPolicy()
        print("Staring execution for worker: " + str(worker_num) + " rollout: " + str(i))
        roller = PolicyRoller(instance_id=worker_num)
        dataset = roller.roll_out_policy(params)
        #error_tracker = roller.get_error_tracker()
        results.append(dataset)
    return results


class ParallelPolicyRoller:
    """
    Compatible with PolicyRoller below (basically the same roll_out_policy function), but runs roll_out_policy
    in parallel in multiple processes.
    """
    def __init__(self, num_workers=1, first_worker=0, reduce=True):
        self.num_workers = num_workers
        self.first_worker = first_worker
        self.reduce = reduce

    def reset(self):
        self.__init__(self.num_workers)

    def roll_out_policy(self, params):

        # Distribute the environments across workers
        env_lists = []
        for k in range(self.num_workers):
            env_lists.append([])
        for i in range(len(params.envs)):
            env_lists[i % self.num_workers].append(params.envs[i])

        worker_arglists = []
        for i in range(self.num_workers):
            worker_arglist = []
            for e in env_lists[i]:
                params = copy.deepcopy(params)
                params.setEnvList([e])
                worker_arglist.append(params)
            worker_arglists.append((self.first_worker + i, worker_arglist))

        pool = mp.Pool(processes=self.num_workers)

        # Map
        results = pool.map(exec_roll_out_policy, worker_arglists)

        pool.close()
        pool.join()

        # Eeach pool worker will return a list of lists of individual rollouts.
        # We don't care which worker each rollout came from, so chain these together in a single list
        results = list(itertools.chain.from_iterable(results))

        # Reduce
        datasets = results
        if self.reduce:
            datasets_out = list(itertools.chain.from_iterable(datasets))
        else:
            datasets_out = datasets
        return datasets_out