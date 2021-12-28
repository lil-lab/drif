import ray
from time import sleep


@ray.remote(num_gpus=0.1, num_cpus=0.1)
class KVStore:

    def __init__(self):
        self.store = {}

    def put(self, key, value):
        self.store[key] = value

    def get(self, key, halt=False):
        if halt:
            while key not in self.store:
                print(f"Waiting for <{key}> in KVStore")
                sleep(1)
        return self.store.get(key, None)