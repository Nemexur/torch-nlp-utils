from typing import (
    Tuple, Iterable, Type
)
from torch_data_utils.data.dataset_readers.dataset_reader import (
    _MemorySizedDatasetInstances, _DatasetInstances
)
import torch
from torch.utils.data import DataLoader, Dataset
from torch_data_utils.common.utils import partialclass


class DataIterator:
    """
    Perform iteration over `DatasetReader` and its subclasses.
    It accepts dataset instance from `DatasetReader.read()` method and parameters for
    `torch.utils.data.DataLoader`.

    `P.S.`: if `max_instances_in_memory` is not None for `DatasetReader` you can pass additional
    sampling methods for `torch.utils.data.DataLoader`
    """
    def __init__(self, dataset: Type[Dataset], *args, **kwargs) -> None:
        self._dataset = dataset
        self._is_memory_sized_dataset = isinstance(
            dataset,
            _MemorySizedDatasetInstances
        )
        if not self._is_memory_sized_dataset:
            self._dataloader: Type[DataLoader] = DataLoader(
                dataset, *args, **kwargs
            )
        else:
            self._dataloader: Type[DataLoader] = partialclass(
                DataLoader, *args, **kwargs
            )

    def __iter__(self):
        if not self._is_memory_sized_dataset:
            yield from self._iterate_normal_dataset()
        else:
            yield from self._iterate_memory_sized_dataset()

    def _iterate_normal_dataset(self) -> Iterable[Tuple[torch.Tensor]]:
        """
        Iterate over normal datasets like `_DatasetInstances`, `_LazyDatasetInstances`
        from `dataset_readers`
        """
        for instance in self._dataloader:
            yield instance

    def _iterate_memory_sized_dataset(self) -> Iterable[Tuple[torch.Tensor]]:
        """
        Iterate over memory-sized dataset which pre loads
        certain amount of instances in memory.
        It also supports additional sampling in `torch.utils.data.DataLoader`
        """
        for dataset in self._dataset:
            for instances in self._dataloader(_DatasetInstances(dataset)):
                yield instances
