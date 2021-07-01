from typing import Iterable, Dict, List, DefaultDict, Callable, Any, Union, T
import torch
import numpy as np
from functools import wraps
from collections import defaultdict
from torch_nlp_utils.common import Registrable
from torch_nlp_utils.common.utils import partialclass
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_nlp_utils.data.dataset_readers.datasets import (
    MemorySizedDatasetInstances, DatasetInstances
)


class Batch:
    """
    Class to construct batch from iterable of instances
    (each instance is a dictionary).

    Example
    -------
    Dictionary instance `{'tokens': ..., 'labels': ...}`
    then to access tokens you need to get an `attribute tokens` from `Batch` class instance.
    """

    def __init__(self, instances: Iterable[Dict[str, List]]) -> None:
        tensor_dict = self._as_tensor_dict_from(instances)
        self.__dict__.update(tensor_dict)

    def __repr__(self) -> str:
        cls = str(self.__class__.__name__)
        info = ", ".join(map(lambda x: f"{x[0]}={x[1]}", self.__dict__.items()))
        return f"{cls}({info})"

    @staticmethod
    def _as_tensor_dict_from(instances: Iterable[Dict[str, List]]) -> DefaultDict[str, List]:
        """
        Construct tensor from list of `instances` per namespace.

        Returns
        -------
        `DefaultDict[str, List]`
            Dict in such format:
                - key: namespace id
                - value: list of torch tensors
        """
        tensor_dict = defaultdict(list)
        for instance in instances:
            for field, tensor in instance.items():
                tensor_dict[field].append(tensor)
        return tensor_dict


def custom_collate(collate_fn: Callable) -> Callable:
    """Decorator for PyTorch collate function."""

    @wraps(collate_fn)
    def wrapper(instances: Iterable[Dict[str, List]]) -> Any:
        return collate_fn(Batch(instances))

    return wrapper


class CollateBatch(Registrable):
    """
    Default class to Collate Batch of Data for Data Iterator.
    Properties of this class are names of parameters that should be passed in forward pass.
    """

    def pin_memory(self) -> T:
        """Pin memory for fast data transfer on CUDA."""
        self.__dict__ = {
            prop: value.pin_memory() if isinstance(value, torch.Tensor) else value
            for prop, value in self.__dict__.items()
        }
        return self

    def to_device(
        self,
        device: Union[str, torch.device],
        **params
    ) -> Dict[str, torch.Tensor]:
        """Helper function to send batch to device and convert it to dict."""
        return {
            prop: value.to(device=device, **params) if isinstance(value, torch.Tensor) else value
            for prop, value in self.__dict__.items()
        }

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self.__dict__


class DataIterator:
    """
    Perform iteration over `DatasetReader` and its subclasses.
    It accepts dataset instance from `DatasetReader.read()` method and parameters for
    `torch.utils.data.DataLoader`. `collate_fn` should accept Batch instance whose arrtributes
    are namespaces seen during training.

    `P.S.`: if `max_instances_in_memory` is not None for `DatasetReader` you can pass additional
    sampling methods for `torch.utils.data.DataLoader`
    """

    def __init__(
        self, dataset: Dataset, batch_size: int, collate_fn: Callable, *args, **kwargs
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._is_memory_sized_dataset = isinstance(dataset, MemorySizedDatasetInstances)
        self._collate_fn = custom_collate(collate_fn)
        if not self._is_memory_sized_dataset:
            self._dataloader: DataLoader = DataLoader(
                dataset, batch_size=batch_size, collate_fn=self._collate_fn, *args, **kwargs
            )
        else:
            self._dataloader: DataLoader = partialclass(
                DataLoader, batch_size=batch_size, collate_fn=self._collate_fn, *args, **kwargs
            )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self):
        if not self._is_memory_sized_dataset:
            return len(self._dataloader)
        else:
            return len(self._dataset)

    def __iter__(self):
        if not self._is_memory_sized_dataset:
            yield from self._dataloader
        else:
            for dataset in self._dataset:
                yield from self._dataloader(DatasetInstances(dataset))

    def sample(self) -> Any:
        """
        Return random sample from dataset.
        It might be needed for tasks where you need to do additional updates
        like in aggressive training for VAE.
        """
        if isinstance(self._dataset, IterableDataset):
            raise Exception("Sample is not supported for Iterable dataset.")
        indices = np.random.choice(len(self._dataset), size=self._batch_size)
        sample = self._collate_fn([self._dataset[idx] for idx in indices])
        if self._dataloader.pin_memory:
            sample.pin_memory()
        return sample
