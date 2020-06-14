from typing import (
    Dict, NamedTuple,
    Callable, Any, Type,
    List, Union
)
import os
import json
from tqdm import tqdm
from enum import Enum
from loguru import logger
from copy import deepcopy
from overrides import overrides
from torch.utils.data import Dataset
from torch_data_utils.common import T
from .dataset_readers import DatasetReader
from torch_data_utils.common.utils import save_params
from .dataset_readers.dataset_reader import (
    _MemorySizedDatasetInstances
)


DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"


class _DictModule:
    """
    Module for dictionary that adds:
    loading, saving and evaluation
    for particular dict subclass.
    """
    @classmethod
    def load(cls: Type['_DictModule'], path: str) -> '_DictModule':
        """Load class from `path`."""
        pass

    def save(self, path: str) -> None:
        """Save data at `path`."""
        pass

    def eval(self) -> None:
        """Set evaluation mode."""
        pass


class _PassThroughDict(dict, _DictModule):
    """
    Dict that returns key as a value on each call.
    If eval is called, it would raise an error
    incase you try to get an item
    that has not been in train data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unique = set()
        self._eval_mode = False

    def __getitem__(self, key: Any):
        if self._eval_mode:
            if key in self._unique:
                return key
            else:
                raise Exception(
                    'Invalid key has been passed. It was not in train.'
                )
        else:
            return key

    def __setitem__(self, key: Any, value: Any):
        if not self._eval_mode:
            self._unique.add(key)

    def __len__(self):
        return len(self._unique)

    @overrides
    def save(self, path: str) -> None:
        """Save data at `path`."""
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(list(self._unique), file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> '_PassThroughDict':
        """Load class from `path`."""
        cls_instance = cls()
        with open(path, 'r', encoding='utf-8') as file:
            cls_instance._unique = set(json.load(file))
        return cls_instance

    @overrides
    def eval(self):
        """Set evaluation mode."""
        self._eval_mode = True


class _VocabularyDictOOV(dict, _DictModule):
    """
    Dict that works just like `defaultdict`
    which returns certain value for unseen key
    but this implementation is much simpler to save
    and it doesn't need default value to be a function.
    """
    def __init__(self, default: Any = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default = default

    @property
    def oov_value(self):
        return self._default

    @oov_value.setter
    def oov_value(self, value: Any):
        self._default = value

    def __missing__(self, key: Any):
        return self._default

    @overrides
    def save(self, path: str) -> None:
        """Save data at `path`."""
        params = {'oov_value': self._default, 'dict': dict(self)}
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(params, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls: Type['_VocabularyDictOOV'], path: str) -> '_VocabularyDictOOV':
        """Load class from `path`."""
        cls_instance = cls()
        with open(path, 'r', encoding='utf-8') as file:
            params = json.load(file)
            cls_instance.oov_value = params['oov_value']
            cls_instance.update(params['dict'])
        return cls_instance


class _ProcessingTypeProperties(NamedTuple):
    """Properties for `_ProcessingType` enum cases."""
    dict_type: Type[T]
    dicts: List[Dict]


class _ProcessingType(Enum):
    """
    Enum of supported cases for namespace processing.
    - padding_oov - set padding as 0 and out-of-vocabulary as 1.
    - oov - do not set padding and set out-of-vocabulary as 0.
    - pass_through - skip any processing and return passed value on each call.
    """
    __add_token_func__ = '__add_token_func__'

    padding_oov = _ProcessingTypeProperties(
        dict_type=_VocabularyDictOOV,
        dicts=[
            _VocabularyDictOOV(1, {DEFAULT_PADDING_TOKEN: 0, DEFAULT_OOV_TOKEN: 1}),
            _VocabularyDictOOV(DEFAULT_OOV_TOKEN, {0: DEFAULT_PADDING_TOKEN, 1: DEFAULT_OOV_TOKEN})
        ]
    )
    oov = _ProcessingTypeProperties(
        dict_type=_VocabularyDictOOV,
        dicts=[
            _VocabularyDictOOV(0, {DEFAULT_OOV_TOKEN: 0}),
            _VocabularyDictOOV(DEFAULT_OOV_TOKEN, {0: DEFAULT_OOV_TOKEN})
        ]
    )
    pass_through = _ProcessingTypeProperties(
        dict_type=_PassThroughDict,
        dicts=[_PassThroughDict(), _PassThroughDict()]
    )

    @property
    def dict_type(self) -> Type[T]:
        """Dict type associated with _ProcessingType case."""
        return self.value.dict_type

    def get_dicts(self) -> List[Dict]:
        """Get dicts for encoding data."""
        return deepcopy(self.value.dicts)

    @staticmethod
    def register_for(
        case: Union[Type['_ProcessingType'], List[Type['_ProcessingType']]]
    ) -> Callable:
        """
        Register certain function for processing _ProcessingType case/cases.
        We need this because it allows to get rid of switch case or if/else.
        """
        case = case if isinstance(case, list) else [case]
        def inner_decorator(func: Callable) -> Callable:  # noqa: E301
            for c in case:
                setattr(c, _ProcessingType.__add_token_func__, func)
            return func
        return inner_decorator


class Namespace:
    """
    Namespace defines all the necessary processing for data from DatasetReader.
    DatasetReader yields dictionaries so namespace is a key of such dictionary.

    Parameters
    ----------
    processing_type : `str`, required
        Type of processing associated with namespace.
        It supports such cases:
            - padding_oov - set padding as 0 and treat out-of-vocabulary tokens.
            - oov - do not set padding and treat out-of-vocabulary tokens.
            - pass_through - skip any processing and return passed value on each call.
    """
    @save_params
    def __init__(self, processing_type: str) -> None:
        if not hasattr(_ProcessingType, processing_type):
            raise ValueError('Invalid processing type has been passed.')
        self._processing_type = getattr(_ProcessingType, processing_type)
        self._token_to_index, self._index_to_token = self._processing_type.get_dicts()

    def add_tokens(self, tokens: List[Any]) -> None:
        """
        Add tokens to namespace.

        Parameters
        ----------
        tokens : `List[Any]`, required
            Tokens to add.
        """
        for token in tokens:
            if not hasattr(self._processing_type, _ProcessingType.__add_token_func__):
                raise Exception(
                    f'You need to register func to add tokens for {self._processing_type.name}'
                )
            # We need to pass self as well
            getattr(self._processing_type, _ProcessingType.__add_token_func__)(self, token)

    @_ProcessingType.register_for(case=_ProcessingType.pass_through)
    def _pass_through(self, token: Any) -> None:
        """
        Function for adding one token in case of processing_type=pass_through is set.
        """
        if not isinstance(token, int):
            return Exception(
                'processing_type=skip is only supported for int values.'
            )
        self._token_to_index[token] = token
        self._index_to_token[token] = token

    @_ProcessingType.register_for(case=[_ProcessingType.padding_oov, _ProcessingType.oov])
    def _add_token(self, token: Any) -> None:
        """
        Function for adding one token in case of processing_type=padding_oov/oov is set.
        """
        if token not in self._token_to_index:
            index = len(self._token_to_index)
            self._token_to_index[token] = index
            self._index_to_token[index] = token

    def save(self, path: str) -> None:
        """
        Save data at `path`.
        You should pass a directory title.
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as file:
            json.dump(self.__init_params__, file, ensure_ascii=False, indent=2)
        self._token_to_index.save(os.path.join(path, 'token_to_index.json'))
        self._index_to_token.save(os.path.join(path, 'index_to_token.json'))

    @classmethod
    def load(cls: Type['Namespace'], path: str) -> 'Namespace':
        """
        Load class from `path`.
        Path is a directory title.
        """
        with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as file:
            namespace = cls(**json.load(file))
        dict_type = namespace._processing_type.dict_type
        namespace._token_to_index = dict_type.load(os.path.join(path, 'token_to_index.json'))
        namespace._index_to_token = dict_type.load(os.path.join(path, 'index_to_token.json'))
        namespace.eval()
        return namespace

    def eval(self) -> None:
        """Set evaluation mode."""
        self._token_to_index.eval()
        self._index_to_token.eval()

    def get_size(self) -> int:
        """Return the number of unique tokens registered for namespace."""
        return len(self._token_to_index)

    def token_to_index(self, token: Any) -> int:
        """Get index for `token`."""
        return self._token_to_index[token]

    def index_to_token(self, index: int) -> Any:
        """Get token for `index`."""
        return self._index_to_token[index]


class Vocabulary:
    """
    A Vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token if needed.
    Vocabulary are fit to a particular dataset and works with `Namespaces`
    which we use to decide which tokens are in-vocabulary.

    Parameters
    ----------
    datasets : `Dict[str, DatasetReader]`, optional (default = `{}`)
        Datasets from which to construct vocabulary.
    namespaces : `Dict[str, Namespace]`, optional (default = `{}`)
        Namespace defines all the necessary processing for data from DatasetReader.
        DatasetReader yields dictionaries so namespace is a key of such dictionary.
        You need to pass all the namespace that particular DatasetReader can yield.
    """
    def __init__(
        self,
        datasets: Dict[str, DatasetReader] = {},
        namespaces: Dict[str, Namespace] = {}
    ) -> None:
        if any([isinstance(datasets, _MemorySizedDatasetInstances) for datasets in datasets.values()]):
            raise Exception('Vocabulary does not support MemorySized Datasets.')
        self._namespaces = deepcopy(namespaces)
        self._generate_vocab_from(datasets)
        # Set eval mode
        for namespace in self._namespaces.values():
            namespace.eval()

    def _generate_vocab_from(self, datasets: Dict[str, Dataset]) -> None:
        """Generate vocab for `datasets`."""
        for type, dataset in datasets.items():
            iterator = tqdm(dataset, desc=f'Iterating {type}')
            for sample in iterator:
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
        will be saved.
        """
        for namespace, namespace_cls in self._namespaces.items():
            namespace_dir = os.path.join(path, namespace)
            namespace_cls.save(namespace_dir)

    @classmethod
    def from_files(cls: Type['Vocabulary'], path: str = 'vocabulary') -> 'Vocabulary':
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
                if isinstance(tokens, list) else self.token_to_index(tokens, namespace)
                for namespace, tokens in sample.items()
            }
        return encoder

    def get_decoder(self) -> Callable:
        """Get decoder for indexes (index -> token)."""
        def decoder(sample: Dict):
            return {
                namespace: [self.index_to_token(index, namespace) for index in indexes]
                if isinstance(indexes, list) else self.index_to_token(indexes, namespace)
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
            logger.error(
                f'\nUnexpected error occurred: {err}.\n'
                f'Namespace: {namespace} '
                f'Token: {token}'
            )
            raise

    def index_to_token(self, index: int, namespace: str = 'tokens') -> Any:
        """Get token for `index` in a `namespace`."""
        try:
            return self._namespaces[namespace].index_to_token(index)
        except Exception as err:
            logger.error(
                f'\nUnexpected error occurred: {err}.\n'
                f'Namespace: {namespace} '
                f'Index: {index}'
            )
            raise
