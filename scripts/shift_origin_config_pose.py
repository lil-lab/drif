"""
This script iterates through json files found in a subdirectory and adds / subtracts to values in these files
The intended use is to shift the origin of all config and pose files.
"""

import config
from data_io.paths import get_all_poses_dir, get_env_config_dir
import os
import itertools
import json
import copy

#DO_WHAT = ["conv_to_1000_part1", "conv_to_1000_part2"]
DO_WHAT = ["conv_to_1000"]
#DO_WHAT = "shift_origin"


def shift_num(x):
    return x + 0.35


def conv_to_1000(x):
    return x * 1000 / 4.7


def conv_to_1000_flip(x):
    return 1000 - conv_to_1000(x)


def identity(x):
    return x


def configure(do_what):
    global in_dirs, out_dirs, keys, key_remap, fun
    if do_what == "shift_origin":
        in_dirs = [get_all_poses_dir(), get_env_config_dir()]
        out_dirs = [get_all_poses_dir() + "_conv", get_env_config_dir() + "_conv"]
        keys = [["zPos", "$ALL"], ["xPos", "$ALL"],
                ["camera", "position", 0], ["camera", "position", 1],
                ["drone", "position", 0], ["drone", "position", 1]]
        key_remap = None
        fun = shift_num

    elif do_what == "conv_to_1000":
        in_dirs = [get_env_config_dir() + "_000"]
        out_dirs = [get_env_config_dir()]
        keys = [["zPos", "$ALL"], ["xPos", "$ALL"]]
        key_remap = None
        fun = {"zPos": conv_to_1000_flip, "xPos": conv_to_1000}

    elif do_what == "conv_to_1000_part1":
        in_dirs = [get_env_config_dir() + "_000"]
        out_dirs = [get_env_config_dir()]
        keys = [["zPos", "$ALL"], ["xPos", "$ALL"]]
        key_remap = {"zPos": "zPos_temp", "xPos": "xPos_temp"}
        fun = {"zPos": conv_to_1000_flip, "xPos": conv_to_1000}

    elif do_what == "conv_to_1000_part2":
        in_dirs = [get_env_config_dir()]
        out_dirs = [get_env_config_dir()]
        keys = [["zPos_temp"], ["xPos_temp"]]
        key_remap = {"zPos_temp": "xPos", "xPos_temp": "zPos"}
        fun = identity


def find_jsons(dir, jsons=None):
    files = os.listdir(dir)
    jsons = jsons if jsons is not None else []
    for f in files:
        pth = os.path.join(dir, f)
        if os.path.isfile(pth) and os.path.splitext(pth)[1] == ".json":
            jsons.append(pth)
        elif os.path.isdir(pth) and f not in [".", ".."]:
            find_jsons(pth, jsons)
    return jsons


def map_var(var, key, func):
    k = key[0]
    out_key = k if key_remap is None else (key_remap[k] if k in key_remap else k)
    if type(func) is dict:
        fun = func[k]
    else:
        fun = func
    print(str(k) + ":" + str(out_key))
    if len(key) == 1:
        if isinstance(var, list):
            if k == "$ALL":
                for i in range(len(var)):
                    var[i] = fun(var[i])
            else:
                var[out_key] = fun(var[k])
        elif isinstance(var, dict):
            if k in var:
                var[out_key] = fun(var[k])
    else:
        if isinstance(var, list):
            raise NotImplementedError("Lists can only appear at the leaf of the data structure")
        else:
            if k in var:
                inner_var = var[k]
                inner_key = key[1:]
                map_var(inner_var, inner_key, fun)
                var[out_key] = var[k]


def map_json(content, keys, func):
    for key in keys:
        var = content
        map_var(var, key, func)
    return content


def map_jsons(in_dirs, out_dirs, keys, func):
    for i, d in enumerate(in_dirs):
        all_jsons = find_jsons(d)
        all_jsons = list(set(all_jsons)) # remove duplicates
        all_json_subdirs = [j[len(d)+1:] for j in all_jsons]
        for j,jsonfile in enumerate(all_jsons):
            with open(jsonfile) as fp:
                content = json.load(fp)
            new_content = map_json(copy.deepcopy(content), keys, func)
            outfile = os.path.join(out_dirs[i], all_json_subdirs[j])
            os.makedirs(os.path.split(outfile)[0], exist_ok=True)
            with open(outfile, "w") as fp:
                json.dump(new_content, fp, indent=4)
            print("Mapped file: ", jsonfile)


if __name__ ==  "__main__":
    do_what = DO_WHAT.copy()
    if type(do_what) is list:
        for what in do_what:
            configure(what)
            print("Doing " + str(what))
            map_jsons(in_dirs, out_dirs, keys, fun)
    else:
        configure(do_what)
        map_jsons(in_dirs, out_dirs, keys, fun)