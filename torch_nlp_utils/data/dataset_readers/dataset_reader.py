from typing import (
    List, Callable,
    Iterable, Iterator, Any,
    Union, Optional, Dict
)
from torch.utils.data import (
    Dataset, IterableDataset
)
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from torch_nlp_utils.data.vocabulary import Vocabulary
from torch_nlp_utils.common.utils import lazy_groups_of
from torch_nlp_utils.common.checks import ConfigurationError


class Encodable:
    def __init__(self):
        self._encoder = lambda x: x

    def encode_with(self, vocab: Vocabulary):
        self._encoder = vocab.get_encoder()

    def encode(self, token: Any) -> Callable:
        return self._encoder(token)


class DatasetInstances(Dataset, Encodable):
    def __init__(self, instances: List[List[Any]]) -> None:
        super().__init__()
        self._instances = instances

    def __getitem__(self, idx):
        return self.encode(self._instances[idx])

    def __len__(self):
        return len(self._instances)


class LazyDatasetInstances(IterableDataset, Encodable):
    """
    An `Iterable` that just wraps a thunk for generating instances and calls it for
    each call to `__iter__`.
    """
    def __init__(
        self,
        instance_generator: Callable[[], Iterable[List[Any]]]
    ) -> None:
        super().__init__()
        self._instance_generator = instance_generator

    def __iter__(self) -> Iterator[List[Any]]:
        instances = self._instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError(
                "For a lazy dataset reader, _read() must return a generator."
            )
        for instance in instances:
            yield self.encode(instance)

    def __len__(self):
        """
        Return one as we yield only one instance
        and can not guess the whole number of instances.
        """
        return 1


class MemorySizedDatasetInstances(IterableDataset, Encodable):
    """
    Breaks the dataset into "memory-sized" lists of instances,
    which it yields up one at a time until it gets through a full epoch.
    For instance, if the dataset is lazily read from disk and we've specified to
    load 1000 instances at a time, then it yields lists of 1000 instances each.
    """
    def __init__(
        self,
        instance_generator: Callable[[], Iterable[List[Any]]],
        max_instances_in_memory: int
    ) -> None:
        super().__init__()
        self._instance_generator = instance_generator
        self._max_instances_in_memory = max_instances_in_memory

    def __iter__(self):
        instances = self._instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError(
                "For a lazy dataset reader, _read() must return a generator"
            )
        for instance in lazy_groups_of(instances, self._max_instances_in_memory):
            yield self.encode(instance)

    def __len__(self):
        """
        Return max_instances_in_memory for IterableDataset
        as we keep in-memory only this many instances at a time
        """
        return self._max_instances_in_memory


class DatasetReader(metaclass=ABCMeta):
    """
    A `DatasetReader` perform data reading from certain file.
    To implement your own, just override the `_read(file_path)` method to return an `Iterable` of the instances.
    This could be a list containing the instances
    or a lazy generator that returns them one at a time.
    All parameters necessary to _read the data apart from the filepath should be passed
    to the constructor of the `DatasetReader`.

    Parameters
    ----------
    lazy : `bool`, optional (default = `False`)
        Whether to read dataset in lazy format or not.
    max_instances_in_memory : `Optional[int]`, optional (default = `None`)
        If specified, dataset reader will load this many instances at a time into an
        in-memory list and then produce batches from one such list at a time. This
        could be useful if your instances are read lazily from disk and you want
        to perform some kind of additional sampling.
    """
    def __init__(
        self,
        lazy: bool = True,
        max_instances_in_memory: Optional[int] = None
    ) -> None:
        self._lazy = lazy
        self._max_instances_in_memory = max_instances_in_memory

    def read(self, file_path: str) -> Union[Iterable[Any], Dataset]:
        """
        Returns an `Iterable` containing all the instances
        in the specified dataset.

        If `lazy` is False, this calls `self._read()`,
        ensures that the result is a list, then returns the resulting list.

        If `lazy` is True, this returns an object whose
        `__iter__` method calls `self._read()` each iteration.
        In this case your implementation of `_read()` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a `ConfigurationError`.

        If `lazy` is True and max_instances_in_memory is not None this
        returns an iterable which will load this many instances at a time into an
        in-memory list which could be splitted into batches.

        In either case, the returned `Iterable` can be iterated
        over multiple times.
        """
        if self._lazy:
            if self._max_instances_in_memory is not None and self._max_instances_in_memory > 0:
                instances: IterableDataset = MemorySizedDatasetInstances(
                    lambda: self._read(file_path),
                    max_instances_in_memory=self._max_instances_in_memory
                )
            else:
                instances: IterableDataset = LazyDatasetInstances(
                    lambda: self._read(file_path)
                )
        else:
            instances = self._read(file_path)

            # Then some validation.
            if not isinstance(instances, list):
                instances = [
                    instance for instance in tqdm(instances, desc='Reading dataset in memory')
                ]
            if not instances:
                raise ConfigurationError(
                    "No instances were read from the given filepath {}. "
                    "Is the path correct?".format(file_path)
                )
            instances: Dataset = DatasetInstances(instances)
        return instances

    @abstractmethod
    def _read(self, file_path: str) -> Iterable[Dict[str, Any]]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator) of dicts.
        You are strongly encouraged to use a generator, so that users can
        read a dataset in a lazy way, if they so choose.
        """
        pass
