import json
import os


def save_json(data, full_path):
    full_path = os.path.expanduser(full_path)
    dirname = os.path.dirname(full_path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)

    with open(full_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(full_path):
    full_path = os.path.expanduser(full_path)
    if not os.path.isfile(full_path):
        return None
    with open(full_path, "r") as f:
        ret = json.load(f)
    return ret