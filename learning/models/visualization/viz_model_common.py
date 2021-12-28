import os


def has_nowrite_flag(run_dir, default_flag):
    fname = os.path.join(run_dir, "nowrite.txt")
    if os.path.exists(fname):
        with open(fname, "r") as fp:
            content = fp.read(100)
        if "true" in content.lower() or "nowrite" in content.lower():
            return True
    else:
        # Create a no-write flag with default value True or False depending on default_flag
        with open(fname, "w") as fp:
            fp.write(str(default_flag))
    return False