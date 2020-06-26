import os
import torch


class SaveCheckpoint:
    """
    Save PyTorch Model after each epoch.

    Parameters
    ----------
    directory : `str`, required
        Directory to save model checkpoints per epoch.
    keep_num_checkpoints : `int`, optional (default = `None`)
        Number of checkpoints to keep. If None then keep all of them.
    """
    def __init__(self, directory: str, keep_num_checkpoints: int = None) -> None:
        os.makedirs(directory, exist_ok=False)
        if keep_num_checkpoints < 1:
            raise Exception('keep_num_checkpoints should be greater than 0')
        self._epoch_idx = 0
        self._directory = directory
        self._keep_num_checkpoints = keep_num_checkpoints

    def __call__(self, model: torch.nn.Module) -> None:
        """Perform saving after one epoch."""
        torch.save(
            model.state_dict(),
            os.path.join(self._directory, f'epoch_{self._epoch_idx}.pt')
        )
        self._epoch_idx += 1
        if self._keep_num_checkpoints:
            self._delete_spare_if_needed()

    def _delete_spare_if_needed(self):
        checkpoints = os.listdir(self._directory)
        if len(checkpoints) > self._keep_num_checkpoints:
            for checkpoint in checkpoints[:-self._keep_num_checkpoints]:
                os.remove(os.path.join(self._directory, checkpoint))
