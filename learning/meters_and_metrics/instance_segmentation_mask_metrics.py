import numpy as np
import learning.meters_and_metrics.meter_server as ms
from scipy.ndimage import gaussian_filter

INSTANCE = None


def get_instance():
    global INSTANCE
    if not INSTANCE:
        INSTANCE = InstanceSegmentationMaskMetric()
    return INSTANCE


class InstanceSegmentationMaskMetric:
    def __init__(self, reset_every_n=-1):
        self.reset()
        self.reset_every_n = reset_every_n

    def reset(self):
        self.log_count = 0

        self._out_true_p = 0
        self._out_false_p = 0
        self._out_true_n = 0
        self._out_false_n = 0
        self._out_correct = 0
        self._out_actual_p = 0
        self._out_actual_n = 0
        self._out_predicted_p = 0
        self._out_predicted_n = 0

        self.mask_probs_if_inside = []
        self.mask_probs = []
        self.weighted_mask_probs = []
        self.successes = 0
        self.successes_if_inside = 0
        self.total_inside = 0
        self.weighted_successes = 0
        self.count = 0

    def _smooth_argmax(self, mask_np, sigma=4):
        smooth_mask_img = gaussian_filter(mask_np, sigma=sigma)
        sargmax = np.argmax(smooth_mask_img)
        sargmax = np.unravel_index(sargmax, mask_np.shape)
        return sargmax

    def log_mask(self, predicted_mask, label_mask):
        self.log_count += 1
        if self.reset_every_n > 0 and self.log_count % self.reset_every_n == 0:
            self.reset()

        batch_size = predicted_mask.inner_distribution.shape[0]
        for i in range(batch_size):
            # First log success of out-of-bounds prediction
            label_outside = label_mask.outer_prob_mass[i].item() > 0.5
            predicted_outside = predicted_mask.outer_prob_mass[i].item() > 0.5
            correct_outside = label_outside == predicted_outside

            out_tp = 1 if label_outside and predicted_outside else 0
            out_fp = 1 if not label_outside and predicted_outside else 0
            out_tn = 1 if not label_outside and not predicted_outside else 0
            out_fn = 1 if label_outside and not predicted_outside else 0
            self._out_true_n += out_tn
            self._out_false_n += out_fn
            self._out_true_p += out_tp
            self._out_false_p += out_fp
            self._out_actual_p += 1 if label_outside else 0
            self._out_actual_n += 1 if not label_outside else 0
            self._out_predicted_p += 1 if predicted_outside else 0
            self._out_predicted_n += 1 if not predicted_outside else 0
            self._out_correct += 1 if correct_outside else 0

            inner_mask_np = predicted_mask.inner_distribution[i].detach().cpu().numpy().transpose((1, 2, 0))
            inner_label_np = label_mask.inner_distribution[i].detach().cpu().numpy().transpose((1, 2, 0))
            sargmax = self._smooth_argmax(inner_mask_np)
            sargmax_prob = inner_label_np[sargmax] / (np.max(inner_label_np) + 1e-30)

            if not label_outside:
                self.mask_probs_if_inside.append(sargmax_prob)
            self.mask_probs.append(sargmax_prob)

            # Success is either predicting correctly that object is not visible, or correctly predicting the location.
            spatial_success_pred = sargmax_prob > 0.5

            if not label_outside:
                self.successes_if_inside += 1 if spatial_success_pred else 0
                self.total_inside += 1

            overall_success = 1 if predicted_outside and label_outside else spatial_success_pred if not label_outside else 0
            self.successes += overall_success
            self.count += 1

    def consolidate(self):
        out_precision = self._out_true_p / (self._out_predicted_p + 1e-30)
        out_recall = self._out_true_p / (self._out_actual_p + 1e-30)
        out_fscore = 2 * out_precision * out_recall / (out_precision + out_recall + 1e-30)

        in_precision = self._out_true_n / (self._out_predicted_n + 1e-30)
        in_recall = self._out_true_n / (self._out_actual_n + 1e-30)
        in_fscore = 2 * in_precision * in_recall / (in_precision + in_recall + 1e-30)
        in_out_accuracy = self._out_correct / (self.count + 1e-30)

        out_prediction_rate = self._out_predicted_p / (self.count + 1e-30)
        out_true_rate = self._out_actual_p / (self.count + 1e-30)

        avg_predicted_centroid_prob_if_inside = sum(self.mask_probs_if_inside) / (len(self.mask_probs_if_inside) + 1e-30)
        avg_predicted_centroid_prob = sum(self.mask_probs) / (len(self.mask_probs) + 1e-30)

        success_rate_if_inside = float(self.successes_if_inside) / float(self.total_inside + 1e-30)
        success_rate = float(self.successes) / float(self.count + 1e-30)

        ms.log_value("out_precision", out_precision)
        ms.log_value("out_recall", out_recall)
        ms.log_value("out_fscore", out_fscore)
        ms.log_value("in_out_accuracy", in_out_accuracy)
        ms.log_value("in_precision", in_precision)
        ms.log_value("in_recall", in_recall)
        ms.log_value("in_fscore", in_fscore)
        ms.log_value("avg_centroid_prob_if_inside", avg_predicted_centroid_prob_if_inside)
        ms.log_value("avg_centroid_prob", avg_predicted_centroid_prob)
        ms.log_value("success_rate_if_inside", success_rate_if_inside)
        ms.log_value("success_rate", success_rate)
        ms.log_value("out_prediction_rate", out_prediction_rate)
        ms.log_value("out_true_rate", out_true_rate)
