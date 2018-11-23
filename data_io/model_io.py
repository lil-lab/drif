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


def load_pytorch_model(model, name):
    path = os.path.join(get_model_dir(), str(name) + ".pytorch")

    model_dict = torch.load(path, map_location=lambda storage, loc: storage)

    # Fix for loading PyTorch 0.3 models in PyTorch 0.4
    model_dict_clone = model_dict.copy()  # We can't mutate while iterating
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]

    model.load_state_dict(model_dict)