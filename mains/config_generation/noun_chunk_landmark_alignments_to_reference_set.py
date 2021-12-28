import json
from data_io.instructions import load_noun_chunk_landmark_alignments
from env_config.definitions.landmarks import UNSEEN_LANDMARK_RADII, ORIG_PORTABLE_LANDMARK_RADII

import parameters.parameter_server as P


ONLY_NOVEL_OBJECTS = False
ONLY_REAL_OBJECTS = True


def skip_this_landmark(landmark):
    if ONLY_NOVEL_OBJECTS:
        if landmark not in UNSEEN_LANDMARK_RADII:
            return True
    if ONLY_REAL_OBJECTS:
        if landmark not in ORIG_PORTABLE_LANDMARK_RADII:
            return True
    return False


def alingments_to_reference_sets(alignments):
    out_dict = {}
    for chunk, landmarks in alignments.items():
        for landmark in landmarks["landmarks"]:
            if skip_this_landmark(landmark):
                continue
            if landmark not in out_dict:
                out_dict[landmark] = []
            out_dict[landmark].append(chunk)
    return out_dict


if __name__ == "__main__":
    P.initialize_experiment()
    train_alignments = load_noun_chunk_landmark_alignments("train")
    dev_alingments = load_noun_chunk_landmark_alignments("dev")

    train_references = alingments_to_reference_sets(train_alignments)
    dev_references = alingments_to_reference_sets(dev_alingments)

    suffix = "_novel" if ONLY_NOVEL_OBJECTS else ""

    with open(f"train_references{suffix}.json", "w") as fp:
        json.dump(train_references, fp, indent=4)

    with open(f"dev_references{suffix}.json", "w") as fp:
        json.dump(dev_references, fp, indent=4)
