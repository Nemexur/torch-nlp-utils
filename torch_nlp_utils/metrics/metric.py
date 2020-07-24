"""
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Metric is slightly modified with additional functionality
for computing metric on single batch.
Copyright by the AllenNLP authors.
"""

from typing import List, Any, Tuple
import torch
import warnings
from abc import ABCMeta, abstractmethod
warnings.filterwarnings('ignore')


class Metric(metaclass=ABCMeta):
    """Default Class for any Metric for One-Label and Multi-Label Tasks."""
    def __init__(self, accum_batches: bool = False):
        self._accum_batchs = accum_batches

    def get_metric(self, reset: bool = False) -> List[Any]:
        """
        Get metric results.

        Parameters
        ----------
        reset : `bool`, optional (default = `False`)
            Whether to clear concatenated predictions and labels.
            Tensors concatenation is useful for getting metric results
            for one batch and for an epoch.
        """
        if self._accum_batchs or reset:
            predictions, labels = self.get_all_batches()
        else:
            predictions, labels = self.get_last_batch()
        metrics = self._get_metric(
            predictions=predictions,
            labels=labels
        )
        if reset:
            self.reset()
        return metrics

    @abstractmethod
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
        pass

    @abstractmethod
    def _get_metric(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> List[Any]:
        """
        Compute metric for certain labels and predictions.

        Parameters
        ----------
        predictions : `torch.Tensor`, required
            Any torch.Tensor with model predictions.
        labels : `torch.Tensor`, required
            Any torch.Tensor with true labels.

        Returns
        -------
        `List[Any]`
            List of metric results.
                - len(results) == 1 for one-label tasks
                - len(results) > 1 for multi-label tasks
        """
        pass

    @abstractmethod
    def get_all_batches(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all accumulated batches for computing metric.
        Returns tuple of two torch.Tensor where first one is predictions,
        second one is labels.
        """
        pass

    @abstractmethod
    def get_last_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get last batch for computing metric.
        Returns tuple of two torch.Tensor where first one is predictions,
        second one is labels.
        """
        pass

    @abstractmethod
    def reset(self):
        pass
