from typing import Dict, Any
import os
import json
import torch
import shutil


class SaveCheckpoint:
    """
    Save PyTorch Model after each epoch.

    Parameters
    ----------
    model : `torch.nn.Module`, required
        Torch model to monitor.
    directory : `str`, required
        Directory to save model checkpoints per epoch.
    keep_num_checkpoints : `int`, optional (default = `None`)
        Number of checkpoints to keep. If None then keep all of them.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        directory: str,
        keep_num_checkpoints: int = None,
    ) -> None:
        os.makedirs(directory, exist_ok=False)
        if keep_num_checkpoints is not None and keep_num_checkpoints < 1:
            raise Exception("keep_num_checkpoints should be greater than 0")
        self.epoch_idx = 0
        self._model = model
        self._directory = directory
        self._keep_num_checkpoints = keep_num_checkpoints

    def __call__(self, metrics: Dict[str, Any], should_save: bool) -> None:
        """Perform saving after one epoch."""
        if not should_save:
            self.epoch_idx += 1
            return
        cur_epoch_dir = os.path.join(self._directory, f"epoch_{self.epoch_idx}")
        os.makedirs(cur_epoch_dir, exist_ok=True)
        # Save torch model
        torch.save(self._model.state_dict(), os.path.join(cur_epoch_dir, "model.pt"))
        # Save metrics
        with open(os.path.join(cur_epoch_dir, "metrics.json"), mode="w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
        self.epoch_idx += 1
        if self._keep_num_checkpoints:
            self._delete_spare_if_needed()

    def _delete_spare_if_needed(self):
        checkpoints = sorted(os.listdir(self._directory))
        if len(checkpoints) > self._keep_num_checkpoints:
            for checkpoint in checkpoints[:-self._keep_num_checkpoints]:
                shutil.rmtree(os.path.join(self._directory, checkpoint), ignore_errors=True)
