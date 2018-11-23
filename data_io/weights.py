import os
import torch

from data_io.paths import get_pretrained_weight_dir


def enable_weight_saving(model, key, alwaysfreeze=False, neverfreeze=False):
    model.pretrained_weights_key = key
    model.get_weights_key = get_weights_key
    model.fix_weights = fix_weights
    model.set_pretrained_weights = set_pretrained_weights
    model.get_pretrained_weights = get_pretrained_weights

    if not hasattr(model, "dont_freeze_weights"):
        model.dont_freeze_weights = []
    if neverfreeze:
        model.dont_freeze_weights.append(key)

    if alwaysfreeze:
        fix_weights(model)
        print("Froze weights: ", key)


def restore_pretrained_weights(model, restore_run_name, fix_weights=False):
    # retrieve from models/restore_run_name
    path = os.path.join(get_pretrained_weight_dir(), restore_run_name)

    print("Restoring weights from pre-training run " + restore_run_name)
    for name, module in model.named_modules():
        if hasattr(module, "set_pretrained_weights"):
            key = module.get_weights_key(module)
            state_path = os.path.join(path, str(key) + ".weights")
            if os.path.isfile(state_path):
                print("    restoring module : " + str(key))
                state_dict = torch.load(state_path, map_location=lambda storage, loc:storage)
                module.set_pretrained_weights(module, state_dict)
                if fix_weights and key not in module.dont_freeze_weights:
                    print ("        fixing weights : " + str(key))
                    module.fix_weights(module)
                else:
                    print ("        NOT fixing loaded weights : " + str(key))
            else:
                print("ERROR: WEIGHTS NOT FOUND FOR: " + str(key))


def save_pretrained_weights(model, name):
    # store in models/run_name/
    path = os.path.join(get_pretrained_weight_dir(), name)
    os.makedirs(path, exist_ok=True)

    for name, module in model.named_modules():
        if hasattr(module, "get_pretrained_weights"):
            state_dict = module.get_pretrained_weights(module)
            key = module.get_weights_key(module)
            state_path = os.path.join(path, str(key) + ".weights")
            torch.save(state_dict, state_path)
            print("Saved pretrained weights: " + str(key))


def get_weights_key(module):
    return module.pretrained_weights_key


def fix_weights(module):
    for param in module.parameters():
        param.requires_grad = False


def set_pretrained_weights(module, weights):
    module.load_state_dict(weights)


def get_pretrained_weights(module):
    return module.state_dict()


def is_pretrainable(module):
    return hasattr(module, "set_pretrained_weights") and \
           hasattr(module, "get_pretrained_weights") and \
           hasattr(module, "fix_weights") and \
           hasattr(module, "get_weights_key")