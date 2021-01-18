from typing import Dict, Any, Union
import json
import torch
import shutil
from pathlib import Path
from loguru import logger


class SaveCheckpoint:
    """
    Save PyTorch Model after each epoch.

    Parameters
    ----------
    model : `torch.nn.Module`, required
        Torch model to monitor.
    directory : `Union[str, Path]`, required
        Directory to save model checkpoints per epoch.
    keep_num_checkpoints : `int`, optional (default = `None`)
        Number of checkpoints to keep. If None then keep all of them.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        model: torch.nn.Module = None,
        keep_num_checkpoints: int = None,
    ) -> None:
        if model is None:
            logger.warning(
                "Model is not passed in init. "
                "Then you should pass dict to save with torch.save by yourself."
            )
        if keep_num_checkpoints is not None and keep_num_checkpoints < 1:
            raise Exception("keep_num_checkpoints should be greater than 0")
        self._directory = Path(directory)
        # Create directory for checkpoint
        self._directory.mkdir(exist_ok=False)
        self.epoch_idx = 0
        self._model = model
        self._directory = directory
        self._keep_num_checkpoints = keep_num_checkpoints
        self._best_model_path = None

    @property
    def best_model_path(self) -> str:
        return self._best_model_path

    def __call__(
        self,
        metrics: Dict[str, Any],
        is_best_so_far: bool,
        save_dict: Dict[str, Any] = None
    ) -> None:
        """Perform saving after one epoch."""
        if not save_dict and not self._model:
            raise Exception("You should pass save_dict on call if model is None.")
        cur_epoch_dir = self._directory / f"epoch_{self.epoch_idx}"
        cur_epoch_dir.mkdir()
        # Save torch model
        torch.save(save_dict or self._model.state_dict(), cur_epoch_dir / "model.pt")
        # Save metrics
        with (cur_epoch_dir / "metrics.json").open(mode="w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
        # Save best model
        if is_best_so_far:
            # Save best model in parent directory
            best_model_path = self._directory.parent / "best-model"
            logger.info(
                f"Best validation performance so far. Copying to `{best_model_path}`.",
            )
            # Delete directory if exists
            if best_model_path.exists():
                shutil.rmtree(best_model_path)
            shutil.copytree(cur_epoch_dir, best_model_path)
            self._best_model_path = best_model_path
        # Delete spare checkpoints
        if self._keep_num_checkpoints:
            self._delete_spare_if_needed()
        # Update epoch index
        self.epoch_idx += 1

    def _delete_spare_if_needed(self):
        checkpoints = sorted(
            [str(x) for x in self._directory.glob("epoch_*")],
            key=lambda x: (len(x), x),
        )
        if len(checkpoints) > self._keep_num_checkpoints:
            for checkpoint in checkpoints[:-self._keep_num_checkpoints]:
                shutil.rmtree(self._directory / checkpoint)
