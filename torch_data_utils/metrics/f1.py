from typing import List
import math
import torch
import warnings
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
    def get_metric(self, reset: bool = False) -> List[float]:
        """
        Get list of metric results for each class.

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
            precision, recall, _ = metrics.precision_recall_curve(
                labels.numpy(),
                prediction.numpy(),
                pos_label=self._positive_label
            )
            f1 = max(2 * ((precision * recall) / (precision + recall)))
            f1_metrics.append(
                f1 if not (math.isnan(f1) or math.isinf(f1)) else 0
            )
        if reset:
            self.reset()
        return f1_metrics

    @overrides
    def reset(self):
        self._all_predictions = torch.FloatTensor()
        self._all_labels = torch.LongTensor()
