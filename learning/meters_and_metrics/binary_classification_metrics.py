import torch
import numpy as np
import learning.meters_and_metrics.meter_server as ms
from scipy.ndimage import gaussian_filter

INSTANCE = None


def get_instance():
    global INSTANCE
    if not INSTANCE:
        INSTANCE = BinaryClassificationMetrics()
    return INSTANCE


class BinaryClassificationMetrics:

    class Storage:
        def __init__(self):
            self.reset()

        def reset(self):
            self.log_count = 0
            self._num_true_p = 0
            self._num_false_p = 0
            self._num_true_n = 0
            self._num_false_n = 0
            self._num_predicted_p = 0
            self._num_predicted_n = 0
            self._num_label_p = 0
            self._num_label_n = 0
            self._num_correct = 0
            self._num_total = 0

    def __init__(self, reset_every_n=-1):
        self.reset_every_n = reset_every_n
        self.reset()

    def reset(self):
        self.storages = {}

    def log_predictions(self, logit_scores, labels, name="window"):
        """
        :param logit_scores: Batch of binary classification scores
        :param labels: Batch of binary classification labels
        :param name:
        :return:
        """
        if name not in self.storages:
            self.storages[name] = BinaryClassificationMetrics.Storage()
        storage = self.storages[name]

        storage.log_count += 1
        if self.reset_every_n > 0 and self.reset_every_n % storage.log_count == 0:
            storage.reset()

        batch_size = logit_scores.shape[0]
        predicted_probs = torch.sigmoid(logit_scores)

        predicted_p = predicted_probs > 0.5
        num_predicted_p = predicted_p.int().sum()
        label_p = labels > 0.5
        num_label_p = label_p.int().sum()

        correct = predicted_p == label_p
        num_correct = correct.int().sum()
        num_total = torch.ones_like(predicted_probs).sum()

        # True / False positives
        # (logical and)
        true_p = predicted_p * label_p
        num_true_p = true_p.int().sum()

        storage._num_predicted_p += int(num_predicted_p)
        storage._num_label_p += int(num_label_p)
        storage._num_true_p += int(num_true_p)
        storage._num_correct += int(num_correct)
        storage._num_total += int(num_total)

    def consolidate(self):
        for key, storage in self.storages.items():
            metric_accuracy = float(storage._num_correct) / (storage._num_total + 1e-30)
            metric_precision = float(storage._num_true_p) / (storage._num_predicted_p + 1e-30)
            metric_recall = float(storage._num_true_p) / (storage._num_label_p + 1e-30)
            metric_f1_score = float(2 * metric_precision * metric_recall) / (metric_precision + metric_recall + 1e-30)

            metric_true_rate = float(storage._num_label_p) / (storage._num_total + 1e-30)
            metric_predicted_true_rate = float(storage._num_predicted_p) / (storage._num_total + 1e-30)

            ms.log_value(f"{key}:accuracy", metric_accuracy)
            ms.log_value(f"{key}:precision", metric_precision)
            ms.log_value(f"{key}:recall", metric_recall)
            ms.log_value(f"{key}:f1_score", metric_f1_score)
            ms.log_value(f"{key}:real_true_rate", metric_true_rate)
            ms.log_value(f"{key}:predicted_true_rate", metric_predicted_true_rate)
