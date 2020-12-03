"""
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Metric is slightly modified with additional functionality
for computing metric on single batch.
Copyright by the AllenNLP authors.
"""

from typing import List, Any, Tuple, NamedTuple, Iterable, Dict, Union
import torch
from overrides import overrides
from abc import ABC, abstractmethod


class MetricTensors(NamedTuple):
    all_batches: torch.Tensor = torch.FloatTensor()
    last_batch: torch.Tensor = torch.FloatTensor()


class Metric(ABC):
    """Default Class for Any Metric."""

    @abstractmethod
    def __call__(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None
    ) -> None:
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        pass

    @abstractmethod
    def get_metric(
        self,
        reset: bool,
    ) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        pass

    @staticmethod
    def _prepare_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        return (
            tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor
            for tensor in tensors
        )


class ClassificationMetric(Metric):
    """Default Class for any Metric for One-Label and Multi-Label Tasks."""

    def __init__(self, accum_batches: bool = False):
        self._accum_batchs = accum_batches
        self._predictions = MetricTensors()
        self._labels = MetricTensors()

    @overrides
    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        predictions, labels = self._prepare_tensors(predictions, labels)
        if len(predictions.size()) == 1:
            predictions = predictions.unsqueeze(1)
            labels = labels.unsqueeze(1)
        if labels.size(-1) != predictions.size(-1):
            raise Exception("Predictions and labels have different number of classes.")
        self._predictions = MetricTensors(
            all_batches=torch.cat([self._predictions.all_batches, self._batch_predictions], dim=0),
            last_batch=self._batch_predictions,
        )
        self._labels = MetricTensors(
            all_batches=torch.cat([self._labels.all_batches, self._batch_labels], dim=0),
            last_batch=self._batch_labels,
        )

    @overrides
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
            predictions, labels = self._predictions.all_batches, self._labels.all_batches
        else:
            predictions, labels = self._predictions.last_batch, self._labels.last_batch
        metrics = self._get_metric(predictions=predictions, labels=labels)
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self):
        self._predictions = MetricTensors()
        self._labels = MetricTensors()

    @abstractmethod
    def _get_metric(self, predictions: torch.Tensor, labels: torch.Tensor) -> List[Any]:
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
