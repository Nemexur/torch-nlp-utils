from typing import Dict


class EarlyStopping:
    """
    Stop model training when certain `metric` did not improve
    for `patience` number of epochs.

    Parameters
    ----------
    patience : `int`, required
        Number of epochs to wait before first improvement.
    metric : `str`, required
        Which metric to monitor in such formath: {- or +}{metric_name}.
        Example: -loss it means loss metric should decrease.
    """
    def __init__(self, patience: int, metric: str):
        if metric[0] == "-":
            self._is_better = lambda old, new: old > new
            self._old_metric = float('inf')
        elif metric[0] == "+":
            self._is_better = lambda old, new: old < new
            self._old_metric = -float('inf')
        else:
            raise Exception(
                "You should pass '-' or '+' at the beginnig of metric name."
            )
        self._patience = 0
        self._metric = metric[1:]
        self._should_stop = False
        self._best_metrics = None
        self._max_patience = patience

    @property
    def improved(self) -> bool:
        """Whether metric improved or not."""
        return self._patience == 0

    @property
    def should_stop(self) -> bool:
        """Whether to stop training or not."""
        return self._should_stop

    def __call__(self, metrics: Dict[str, float]) -> None:
        """Check results after one epoch."""
        if self._is_better(self._old_metric, metrics[self._metric]):
            self._patience = 0
            self._best_metrics = metrics
            self._old_metric = metrics[self._metric]
        else:
            self._patience += 1
        # Raise error if max_patience
        if self._patience == self._max_patience:
            self._should_stop = True
