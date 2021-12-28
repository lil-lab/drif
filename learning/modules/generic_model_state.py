from learning.modules.key_tensor_store import KeyTensorStore


class DefaultDie:
    ...


class GenericModelState:

    def __init__(self):
        self.tensor_store = None
        self.kv = {}
        self.reset()

    def reset(self):
        self.tensor_store = KeyTensorStore()
        self.kv = {}

    def put(self, key, value):
        self.kv[key] = value

    def get(self, key, default=DefaultDie()):
        if isinstance(key, list):
            results = [self.get(k, default) for k in key]
            return tuple(results)
        else:
            if isinstance(default, DefaultDie):
                return self.kv[key]
            else:
                if key in self.kv:
                    return self.kv[key]
                return default
