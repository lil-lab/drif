import torch
import numpy as np
from data_io.models import load_model
from learning.datasets.object_reference_dataset import ObjectReferenceDataset

import parameters.parameter_server as P

objects = [
    "calculator",
    "gray box",
    "blue cabinet",
    "pink thing",
    "sofa",
    "big plate",
    "cup",
    "pizza box",
    "tall construction crane",
    "nearby round shape"
]
spurious = [
    "left",
    "the left",
    "the right",
    "endpoint",
    "straight",
    "curve",
    "left curve",
    "clockwise trajectory",
    "stop",
    "stop near the tree"
]

THRES = 0.01


def query_object_reference_recognizer():
    dataset = ObjectReferenceDataset()
    model, _ = load_model()
    model.eval()

    inputs = objects + spurious
    labels = [0] * len(objects) + [1] * len(spurious)

    num_correct = 0
    num_total = 0

    for input, label in zip(inputs, labels):
        chunk_embedding = dataset._vectorize(input)
        chunk_embedding_t = torch.tensor(chunk_embedding)[np.newaxis, :].cuda()
        prediction = model(chunk_embedding_t)
        score = torch.sigmoid(prediction)[0].cpu().item()
        predicted_label = 0 if score < THRES else 1
        print("--------------------------------------------------------------------------")
        print("Input: ", input)
        print("Predicted score: ", score)
        print(f"Class: {'object reference' if predicted_label == 0 else 'spurious chunk'}")

        correct = predicted_label == label
        num_correct += (1 if correct else 0)
        num_total += 1

    print("--------------------------------------------------------------------------")
    print("Overall accuracy: ", float(num_correct) / float(num_total))


if __name__ == "__main__":
    P.initialize_experiment()
    query_object_reference_recognizer()