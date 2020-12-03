from typing import List, Dict
import torch
import numpy as np
from sklearn import metrics
from overrides import overrides
from .metric import ClassificationMetric


class F1Metric(ClassificationMetric):
    """
    Compute F1 Metric for One-Label and Multi-Label Tasks.

    Parameters
    ----------
    positive_label : `int`, optional (default = `1`)
        Positive class label for metric.
    accum_batchs : `bool`, optional (default = `False`)
        Whether to compute metric for all accumulated batches or not.
        If reset == True use all batchs.
    """

    def __init__(self, positive_label: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._positive_label = positive_label

    @overrides
    def _get_metric(self, predictions: torch.FloatTensor, labels: torch.FloatTensor) -> List[Dict[str, int]]:
        f1_metrics = []
        for idx in range(labels.size(-1)):
            prediction = predictions[:, idx]
            target = labels[:, idx]
            # PR-Curve
            precision, recall, _ = metrics.precision_recall_curve(
                target.numpy(), prediction.numpy(), pos_label=self._positive_label
            )
            # F1-score
            f1 = 2 * ((precision * recall) / (precision + recall))
            max_idx = np.nanargmax(f1)
            f1_metrics.append(
                {"f1-score": f1[max_idx], "precision": precision[max_idx], "recall": recall[max_idx]}
            )
        return f1_metrics
