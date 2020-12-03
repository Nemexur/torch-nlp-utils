from typing import List
import math
import torch
from sklearn import metrics
from overrides import overrides
from .metric import ClassificationMetric


class AucMetric(ClassificationMetric):
    """
    Compute ROC-AUC Metric for One-Label and Multi-Label Tasks.

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
    def _get_metric(self, predictions: torch.FloatTensor, labels: torch.FloatTensor) -> List[int]:
        auc_metrics = []
        for idx in range(labels.size(-1)):
            prediction = predictions[:, idx]
            target = labels[:, idx]
            false_positive_rates, true_positive_rates, _ = metrics.roc_curve(
                target.numpy(), prediction.numpy(), pos_label=self._positive_label
            )
            auc = metrics.auc(false_positive_rates, true_positive_rates)
            auc_metrics.append(auc if not (math.isnan(auc) or math.isinf(auc)) else 0)
        return auc_metrics
