import torch
import numpy as np
from sklearn import svm

from learning.datasets.object_reference_dataset import ObjectReferenceDataset
from mains.analysis.quick_test_object_reference_recognizer import spurious, objects
import parameters.parameter_server as P


def extract_data(dataset):
    data = [(x.numpy(), y.numpy()) for x, y in dataset]
    X, Y = zip(*data)
    return X, Y


def fit_object_reference_recognizer():
    dset_train = ObjectReferenceDataset(eval=False)
    dset_test = ObjectReferenceDataset(eval=True)
    X_train, Y_train = extract_data(dset_train)
    clf = svm.SVC(kernel="rbf", probability=True)
    clf.fit(X_train, Y_train)

    test_strings = spurious + objects
    X_test = [dset_train._vectorize(c) for c in test_strings]
    Y_test = [1] * len(spurious) + [0] * len(objects)

    Y_star_p = clf.predict_proba(X_test)
    Y_star = clf.predict(X_test)

    for x, y_star, y, p in zip(test_strings, Y_star, Y_test, Y_star_p):
        print(f"{x} : Predicted: {y_star}. True: {y}. Prob: {p}")
    print("ding")


if __name__ == "__main__":
    P.initialize_experiment()
    fit_object_reference_recognizer()