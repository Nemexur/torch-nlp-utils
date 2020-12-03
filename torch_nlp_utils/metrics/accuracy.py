import torch
from .metric import Metric
from overrides import overrides


class AccuracyMetric(Metric):
    """
    Compute Accuracy Metric over 3D-tensor.

    Parameters
    ----------
    top_k : `int`, optional (default = `1`)
        Top K elements to compute classification on.
    """

    def __init__(self, top_k: int = 1) -> None:
        super().__init__()
        self._top_k = top_k
        self._true_count = 0.0
        self._all_count = 0.0

    @overrides
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
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, labels, mask = self._prepare_tensors(predictions, labels, mask)
        # Some sanity checks.
        num_classes = predictions.size(-1)
        if labels.dim() != predictions.dim() - 1:
            raise Exception(
                "gold_labels must have dimension == predictions.size() - 1 but "
                "found tensor of shape: {}".format(predictions.size())
            )
        if (labels >= num_classes).any():
            raise Exception(
                "A gold label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )
        predictions = predictions.view(-1, num_classes)
        gold_labels = labels.view(-1).long()
        # Top K indexes of the predictions (or fewer, if there aren't K of them).
        # Special case topk == 1, because it's common and .max() is much faster than .topk().
        if self._top_k == 1:
            top_k = predictions.max(-1)[1].unsqueeze(-1)
        else:
            top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]
        # This is of shape (batch_size, ..., top_k).
        correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        if mask is not None:
            correct *= mask.view(-1, 1).float()
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
