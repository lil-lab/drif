import os

import torch

from data_io.paths import get_model_dir
import parameters.parameter_server as P


def save_pytorch_model(model, name):
    path = os.path.join(get_model_dir(), str(name) + ".pytorch")
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        try: os.makedirs(dirname)
        except Exception: pass
    torch.save(model.state_dict(), path)


def load_model_state_dict(name):
    path = os.path.join(get_model_dir(), str(name) + ".pytorch")
    model_dict = torch.load(path, map_location=lambda storage, loc: storage)
    return model_dict


def load_pytorch_model(model, name, namespace=None, ignore_unexpected=False):
    path = os.path.join(get_model_dir(), str(name) + ".pytorch")

    model_dict = torch.load(path, map_location=lambda storage, loc: storage)

    if namespace is not None:
        new_dict = {}
        for k,v in model_dict.items():
            if k.startswith(namespace):
                k = k[len(namespace)+1:]
                new_dict[k] = v
        model_dict = new_dict

    if ignore_unexpected:
        def filter_model_dict(model, dict):
            out_dict = {}
            for k, _ in model.named_parameters():
                out_dict[k] = dict[k]
            return out_dict
        model_dict = filter_model_dict(model, model_dict)

    if model is not None:
        model.load_state_dict(model_dict)
    else:
        return model_dict


def load_pytorch_model_from_config(model, config_path):
    model_name = P.get(config_path)
    return load_pytorch_model(model, model_name)


def find_state_subdict(state_dict, prefix):
    subdict = {}
    for k,v in state_dict.items():
        if k.startswith(prefix):
            subdict[k[len(prefix)+1:]] = v
    return subdict