import torch
import numpy as np
from data_io.models import load_model
from learning.datasets.object_reference_dataset import ObjectReferenceDataset

import parameters.parameter_server as P


def query_object_reference_recognizer():
    dataset = ObjectReferenceDataset()
    model, _ = load_model()
    model = model.cpu()
    while True:
        x = input("Enter a noun phrase")
        x = x.strip()
        chunk_embedding = dataset._vectorize(x)
        chunk_embedding_t = torch.tensor(chunk_embedding)[np.newaxis, :]#.cuda()
        prediction = model(chunk_embedding_t)
        pred_class = model.threshold(prediction)[0].cpu().item()
        score = torch.sigmoid(prediction)[0].cpu().item()
        print("Predicted score: ", score)
        print(f"Class: {'object reference' if pred_class < 0.5 else 'spurious chunk'}")


if __name__ == "__main__":
    P.initialize_experiment()
    query_object_reference_recognizer()