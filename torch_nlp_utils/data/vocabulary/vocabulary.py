from typing import (
    Dict, Callable, Any,
    Type, T, List, Iterable
)
import os
from tqdm import tqdm
from copy import deepcopy
from .namespace import Namespace
from torch.utils.data import Dataset


class Vocabulary:
    """
    A Vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token if needed.
    Vocabulary are fit to a particular dataset and works with `Namespaces`
    which we use to decide which tokens are in-vocabulary.

    Parameters
    ----------
    datasets : `Dict[str, Dataset]`, optional (default = `{}`)
        Datasets from which to construct vocabulary.
        `MemorySizedDataset is not supported`.
    namespaces : `Dict[str, Namespace]`, optional (default = `{}`)
        Namespace defines all the necessary processing for data from DatasetReader.
        DatasetReader yields dictionaries so namespace is a key of such dictionary.
        You need to pass all the namespace that particular DatasetReader can yield.
    dependent_namespaces : `List[List[str]]`, optional (default = `[]`)
        Set namespaces that should share its vocabulary.
    """
    def __init__(
        self,
        datasets: Dict[str, Dataset] = {},
        namespaces: Dict[str, Namespace] = {},
        dependent_namespaces: List[List[str]] = []
    ) -> None:
        if (datasets and not namespaces) or (not datasets and namespaces):
            raise Exception(
                'You need to define both datasets and namespaces '
                'or set them empty.'
            )
        self._namespaces = deepcopy(namespaces)
        for d_namespaces in dependent_namespaces:
            self._set_dependent_namespaces([self._namespaces[x] for x in d_namespaces])
        self._generate_vocab_from(datasets)
        # Set eval mode
        for namespace in self._namespaces.values():
            namespace.eval()

    def _set_dependent_namespaces(self, namespaces: List[Namespace]) -> None:
        """Setup Namespaces that should have shared dicts."""
        token_to_index, index_to_token = namespaces[0].token_to_index, namespaces[0].index_to_token
        for namespace in namespaces:
            namespace._token_to_index, namespace._index_to_token = token_to_index, index_to_token

    def _generate_vocab_from(self, datasets: Dict[str, Dataset]) -> None:
        """Generate vocab from `datasets`."""
        for type, dataset in datasets.items():
            for sample in tqdm(dataset, desc=f'Iterating {type}'):
                self._add_sample_to_namespace(sample)

    def _add_sample_to_namespace(self, sample: Dict[str, Any]) -> None:
        """Add one `sample` to certain namespaces defined by data."""
        for namespace, data in sample.items():
            if namespace not in self._namespaces:
                raise Exception('Unrecognized namespace occurred.')
            tokens = data if isinstance(data, list) else [data]
            self._namespaces[namespace].add_tokens(tokens)

    def save(self, path: str = 'vocabulary') -> None:
        """
        Save data at `path`.
        You should pass a directory title
        in which directory for Vocabulary
        will be saved (default = `'vocabulary'`).
        """
        for namespace, namespace_cls in self._namespaces.items():
            namespace_dir = os.path.join(path, namespace)
            namespace_cls.save(namespace_dir)

    @classmethod
    def from_files(cls: Type[T], path: str = 'vocabulary') -> T:
        """Load `Vocabulary` instance from files in directory at `path`."""
        vocab = cls()
        namespaces = os.listdir(path)
        for namespace in namespaces:
            namespace_dir = os.path.join(path, namespace)
            vocab._namespaces[namespace] = Namespace.load(namespace_dir)
        return vocab

    def get_encoder(self) -> Callable:
        """Get encoder for tokens (token -> index)."""
        def encoder(sample: Dict):
            return {
                namespace: [self.token_to_index(token, namespace) for token in tokens]
                if isinstance(tokens, Iterable) else self.token_to_index(tokens, namespace)
                for namespace, tokens in sample.items()
            }
        return encoder

    def get_decoder(self) -> Callable:
        """Get decoder for tokens (index -> token)."""
        def decoder(sample: Dict):
            return {
                namespace: [self.index_to_token(index, namespace) for index in indexes]
                if isinstance(indexes, Iterable) else self.index_to_token(indexes, namespace)
                for namespace, indexes in sample.items()
            }
        return decoder

    def get_vocab_size(self, namespace: str = 'tokens') -> int:
        """Get size of vocabulary for a `namespace`."""
        return self._namespaces[namespace].get_size()

    def token_to_index(self, token: Any, namespace: str = 'tokens') -> int:
        """Get index for `token` in a `namespace`."""
        try:
            return self._namespaces[namespace].token_to_index(token)
        except Exception as err:
            raise Exception(
                f'\nUnexpected error occurred: {err}\n'
                f'Namespace: {namespace} '
                f'Token: {token}'
            )

    def index_to_token(self, index: int, namespace: str = 'tokens') -> Any:
        """Get token for `index` in a `namespace`."""
        try:
            return self._namespaces[namespace].index_to_token(index)
        except Exception as err:
            raise Exception(
                f'\nUnexpected error occurred: {err}.\n'
                f'Namespace: {namespace} '
                f'Index: {index}'
            )
