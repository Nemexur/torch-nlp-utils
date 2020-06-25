import os
import torch


class SaveCheckpoint:
    """
    Save PyTorch Model after each epoch.

    Parameters
    ----------
    directory : `str`, required
        Directory to save model checkpoints per epoch.
    """
    def __init__(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=False)
        self._directory = directory
        self._epoch_idx = 0

    def __call__(self, model: torch.nn.Module) -> None:
        """Perform saving after one epoch."""
        torch.save(
            model.state_dict(),
            os.path.join(self._directory, f'epoch_{self._epoch_idx}.pt')
        )
        self._epoch_idx += 1
