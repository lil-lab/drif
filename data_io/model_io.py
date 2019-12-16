import os

import torch

from data_io.paths import get_model_dir


def save_pytorch_model(model, name):
    path = os.path.join(get_model_dir(), str(name) + ".pytorch")
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        try: os.makedirs(dirname)
        except Exception: pass
    torch.save(model.state_dict(), path)


def load_pytorch_model(model, name, pytorch3to4=False, namespace=None):
    path = os.path.join(get_model_dir(), str(name) + ".pytorch")

    model_dict = torch.load(path, map_location=lambda storage, loc: storage)

    # Fix for loading PyTorch 0.3 models in PyTorch 0.4
    if pytorch3to4:
        model_dict_clone = model_dict.copy()  # We can't mutate while iterating
        for key, value in model_dict_clone.items():
            if key.endswith(('running_mean', 'running_var')):
                del model_dict[key]

    if namespace is not None:
        new_dict = {}
        for k,v in model_dict.items():
            if k.startswith(namespace):
                k = k[len(namespace)+1:]
                new_dict[k] = v
        model_dict = new_dict

    if model is not None:
        model.load_state_dict(model_dict)
    else:
        return model_dict

def find_state_subdict(state_dict, prefix):
    subdict = {}
    for k,v in state_dict.items():
        if k.startswith(prefix):
            subdict[k[len(prefix)+1:]] = v
    return subdict