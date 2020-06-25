from typing import List
import torch
import warnings
from abc import ABCMeta, abstractmethod
warnings.filterwarnings('ignore')


class Metric(metaclass=ABCMeta):
    """Default Class for any Metric for One-Label and Multi-Label Tasks."""
    @abstractmethod
    def __call__(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        pass

    @abstractmethod
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

    @abstractmethod
    def reset(self):
        pass
