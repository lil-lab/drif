"""
This script iterates through json files found in a subdirectory and adds / subtracts to values in these files
The intended use is to shift the origin of all config and pose files.
"""

from data_io.paths import get_env_config_dir
import os
import shutil

FROM = "config_"
TO = "random_config_"


def rename_configs():
    confdir = get_env_config_dir()
    jsons = os.listdir(confdir)
    jsons = [j for j in jsons if j.endswith(".json")]
    jsons_out = [TO + j.split(FROM)[1] for j in jsons]

    for i, j_in in enumerate(jsons):
        j_in = os.path.join(confdir, j_in)
        j_out = os.path.join(confdir, jsons_out[i])
        shutil.move(j_in, j_out)
        print(j_in + " --> ", j_out)


if __name__ ==  "__main__":
    rename_configs()