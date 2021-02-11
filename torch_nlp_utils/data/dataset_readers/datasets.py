from typing import List, Callable, Iterable, Iterator, Any, Dict, Union
from overrides import overrides
from torch_nlp_utils.data.vocabulary import Vocabulary
from torch_nlp_utils.common.utils import lazy_groups_of
from torch_nlp_utils.common.checks import ConfigurationError
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class Encodable:
    def __init__(self):
        self._vocab: Vocabulary = None

    def encode_with(self, vocab: Vocabulary) -> None:
        self._vocab = vocab

    def encode(self, token: Dict[str, Union[Any, List[Any]]]) -> Dict[str, Union[int, List[int]]]:
        return self._vocab.encode(token) if self._vocab is not None else token


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

    def __init__(self, instance_generator: Callable[[], Iterable[List[Any]]]) -> None:
        super().__init__()
        self._instance_generator = instance_generator

    def __iter__(self) -> Iterator[List[Any]]:
        instances = self._instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError("For a lazy dataset reader, _read() must return a generator.")
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
        self, instance_generator: Callable[[], Iterable[List[Any]]], max_instances_in_memory: int
    ) -> None:
        super().__init__()
        self._instance_generator = instance_generator
        self._max_instances_in_memory = max_instances_in_memory

    def __iter__(self):
        instances = self._instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError("For a lazy dataset reader, _read() must return a generator.")
        for instance in lazy_groups_of(instances, self._max_instances_in_memory):
            # Returns list of instances.
            yield [self.encode(x) for x in instance]

    def __len__(self):
        """
        Return max_instances_in_memory for IterableDataset
        as we keep in-memory only this many instances at a time.
        """
        return self._max_instances_in_memory

# TODO: Work in progress
class ShardedDatasetInstances(IterableDataset, Encodable):
    def __init__(self, shards: List[IterableDataset]) -> None:
        super().__init__()
        self._shards = shards

    @overrides
    def encode_with(self, vocab: Vocabulary) -> None:
        for shard in self._shards:
            # Shard is an Iterable Dataset of type
            # LazyDatasetInstances or MemorySizedDatasetInstances
            shard.encode_with(vocab)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            for shard in self._shards:
                yield from shard
        else:
            if worker_info.num_workers != len(self._shards):
                raise Exception("Number of workers should be equal to the number of datasets.")
            rank = worker_info.id
            return iter(self._shards[rank])
