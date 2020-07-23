from typing import List, Dict, Tuple
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
    accum_batchs : `bool`, optional (default = `False`)
        Whether to compute metric for all accumulated batches or not.
        If reset == True use all batchs.
    """
    def __init__(self, positive_label: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._positive_label = positive_label
        self._all_predictions = torch.FloatTensor()
        self._all_labels = torch.LongTensor()
        self._batch_predictions = torch.FloatTensor()
        self._batch_labels = torch.FloatTensor()

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
        self._batch_predictions = predictions.detach().float()
        self._batch_labels = predictions.detach().long()
        self._all_predictions = torch.cat([
            self._all_predictions,
            self._batch_predictions
        ], dim=0)
        self._all_labels = torch.cat([
            self._all_labels,
            self._batch_labels
        ], dim=0)

    @overrides
    def _get_metric(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor
    ) -> List[Dict[str, int]]:
        f1_metrics = []
        for idx in range(labels.size(-1)):
            prediction = predictions[:, idx]
            target = labels[:, idx]
            # PR-Curve
            precision, recall, _ = metrics.precision_recall_curve(
                target.numpy(),
                prediction.numpy(),
                pos_label=self._positive_label
            )
            # F1-score
            f1 = 2 * ((precision * recall) / (precision + recall))
            max_idx = np.nanargmax(f1)
            f1_metrics.append({
                'f1-score': f1[max_idx],
                'precision': precision[max_idx],
                'recall': recall[max_idx]
            })
        return f1_metrics

    @overrides
    def get_all_batches(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all accumulated batches for computing metric.
        Returns tuple of two torch.Tensor where first one is predictions,
        second one is labels.
        """
        return self._all_predictions, self._all_labels

    @overrides
    def get_last_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get last batch for computing metric.
        Returns tuple of two torch.Tensor where first one is predictions,
        second one is labels.
        """
        return self._batch_predictions, self._batch_labels

    @overrides
    def reset(self):
        self._all_predictions = torch.FloatTensor()
        self._all_labels = torch.LongTensor()
