from typing import List, Dict
import math
import torch
import warnings
import numpy as np
from .metric import Metric
from sklearn import metrics
from overrides import overrides
warnings.filterwarnings('ignore')


class F1Metric(Metric):
    """
    Compute F1 Metric for One-Label and Multi-Label Tasks.

    Parameters
    ----------
    positive_label : `int`, optional (default = `1`)
        Positive class label for metric.
    """
    def __init__(self, positive_label: int = 1):
        self._positive_label = positive_label
        self._all_predictions = torch.FloatTensor()
        self._all_labels = torch.LongTensor()

    @overrides
    def __call__(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        """
        Save predictions and labels.

        Parameters
        ----------
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, num_classes).
        labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, num_classes). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        """
        if len(predictions.size()) == 1:
            predictions = predictions.unsqueeze(1)
            labels = labels.unsqueeze(1)
        if labels.size(-1) != predictions.size(-1):
            raise Exception(
                "Predictions and labels have different number of classes."
            )
        self._all_predictions = torch.cat([
            self._all_predictions,
            predictions.detach().float()
        ], dim=0)
        self._all_labels = torch.cat([
            self._all_labels,
            labels.detach().long()
        ], dim=0)

    @overrides
    def get_metric(self, reset: bool = False) -> List[Dict[str, float]]:
        """
        Get list of metric results for each class.
        Each metric is a dict with this keys:
        f1-score, precision, recall, threshold

        Parameters
        ----------
        reset : `bool`, optional (default = `False`)
            Whether to clear concatenated predictions and labels.
            Tensors concatenation is useful for getting metric results
            for one batch and for an epoch.
        """
        f1_metrics = []
        for idx in range(self._all_labels.size(-1)):
            prediction = self._all_predictions[:, idx]
            labels = self._all_labels[:, idx]
            # PR-Curve
            precision, recall, thr = metrics.precision_recall_curve(
                labels.numpy(),
                prediction.numpy(),
                pos_label=self._positive_label
            )
            # F1-score
            f1 = 2 * ((precision * recall) / (precision + recall))
            max_idx = np.argmax(f1)
            f1_max = f1[max_idx]
            f1_metrics.append({
                'f1-score': f1_max if not (math.isnan(f1_max) or math.isinf(f1_max)) else 0,
                'precision': precision[max_idx],
                'recall': recall[max_idx],
                'threshold': thr[max_idx]
            })
        if reset:
            self.reset()
        return f1_metrics

    @overrides
    def reset(self):
        self._all_predictions = torch.FloatTensor()
        self._all_labels = torch.LongTensor()
