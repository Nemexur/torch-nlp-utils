from typing import (
    Iterable, Dict,
    List, DefaultDict,
    Callable, Any
)
from torch_nlp_utils.data.dataset_readers.dataset_reader import (
    MemorySizedDatasetInstances, DatasetInstances
)
from functools import wraps
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from torch_nlp_utils.common.utils import partialclass


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
        info = ', '.join(map(lambda x: f"{x[0]}={x[1]}", self.__dict__.items()))
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
        self,
        dataset: Dataset,
        collate_fn: Callable,
        *args, **kwargs
    ) -> None:
        self._dataset = dataset
        self._is_memory_sized_dataset = isinstance(
            dataset,
            MemorySizedDatasetInstances
        )
        if not self._is_memory_sized_dataset:
            self._dataloader: DataLoader = DataLoader(
                dataset, collate_fn=custom_collate(collate_fn),
                *args, **kwargs
            )
        else:
            self._dataloader: DataLoader = partialclass(
                DataLoader, collate_fn=custom_collate(collate_fn),
                *args, **kwargs
            )

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
