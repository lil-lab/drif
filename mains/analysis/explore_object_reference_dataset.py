from learning.datasets.object_reference_dataset import ObjectReferenceDataset

import parameters.parameter_server as P


def explore_object_reference_dataset():
    dataset = ObjectReferenceDataset(eval=False)
    for example in dataset:
        print("ding")


if __name__ == "__main__":
    P.initialize_experiment()
    explore_object_reference_dataset()