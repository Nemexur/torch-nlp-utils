from typing import (
    Type, T, Dict, NamedTuple,
    List, Callable, Union, Any
)
import os
import json
from enum import Enum
from copy import deepcopy
from .dicts import PassThroughDict, NamespaceDict
from torch_nlp_utils.common.utils import save_params


DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"


class ProcessingTypeProperties(NamedTuple):
    """Properties for `ProcessingType` enum cases."""
    dict_type: Type[T]
    dicts: List[Dict]


class ProcessingType(Enum):
    """
    Enum of supported cases for namespace processing.
    - padding_oov - set padding as 0 and out-of-vocabulary as 1.
    - padding - do not set out-of-vocabulary token and set padding as 0.
    - just_encode - encode value starting from 0 and raise error for out-of-vocabulary tokens.
    - pass_through - skip any processing and return passed value on each call.
    """
    __add_token_func__ = '__add_token_func__'

    padding_oov = ProcessingTypeProperties(
        dict_type=NamespaceDict,
        dicts=[
            NamespaceDict(1, {DEFAULT_PADDING_TOKEN: 0, DEFAULT_OOV_TOKEN: 1}),
            NamespaceDict(DEFAULT_OOV_TOKEN, {0: DEFAULT_PADDING_TOKEN, 1: DEFAULT_OOV_TOKEN})
        ]
    )
    padding = ProcessingTypeProperties(
        dict_type=NamespaceDict,
        dicts=[
            NamespaceDict(0, {DEFAULT_PADDING_TOKEN: 0}),
            NamespaceDict(DEFAULT_PADDING_TOKEN, {0: DEFAULT_PADDING_TOKEN})
        ]
    )
    just_encode = ProcessingTypeProperties(
        dict_type=NamespaceDict,
        dicts=[NamespaceDict(), NamespaceDict()]
    )
    pass_through = ProcessingTypeProperties(
        dict_type=PassThroughDict,
        dicts=[PassThroughDict(), PassThroughDict()]
    )

    @property
    def dict_type(self) -> Type[T]:
        """Dict type associated with ProcessingType case."""
        return self.value.dict_type

    def get_dicts(self) -> List[Dict]:
        """Get dicts for encoding data."""
        return deepcopy(self.value.dicts)

    @staticmethod
    def register_for(
        case: Union[Type[T], List[Type[T]]]
    ) -> Callable:
        """
        Register certain function for processing ProcessingType case/cases.
        We need this because it simplifies if/else for this enum.
        """
        case = case if isinstance(case, list) else [case]
        def inner_decorator(func: Callable) -> Callable:  # noqa: E301
            for c in case:
                setattr(c, ProcessingType.__add_token_func__, func)
            return func
        return inner_decorator


# Simple shortcut for ProcessingType Enum
_PT = ProcessingType


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
            - padding - do not set out-of-vocabulary token and set padding as 0.
            - just_encode - encode value starting from 0 and raise error for out-of-vocabulary tokens.
            - pass_through - skip any processing and return passed value on each call.
    max_size : `int`, optional (default = `None`)
        Max size of vocabulary for a namespace.
    """
    @save_params
    def __init__(
        self,
        processing_type: str,
        max_size: int = None,
    ) -> None:
        if not hasattr(_PT, processing_type):
            raise ValueError('Invalid processing type has been passed.')
        if max_size is not None and max_size <= 0:
            raise Exception('Max size can not be less or equal 0.')
        self._from_saved = False
        self._max_size = max_size
        self._processing_type = getattr(_PT, processing_type)
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
            if not hasattr(self._processing_type, _PT.__add_token_func__):
                raise Exception(
                    f'You need to register func to add tokens for {self._processing_type.name}.'
                )
            # We need to pass self as well
            getattr(self._processing_type, _PT.__add_token_func__)(self, token)

    @ProcessingType.register_for(case=_PT.pass_through)
    def _pass_through(self, token: Any) -> None:
        """Function for adding token in case of `processing_type=pass_through`."""
        if not isinstance(token, int):
            return Exception(
                'processing_type=pass_through is only supported for int values.'
            )
        # json saves only strings as keys
        self._token_to_index[token] = token
        self._index_to_token[token] = token

    @ProcessingType.register_for(case=[_PT.padding_oov, _PT.padding, _PT.just_encode])
    def _add_token(self, token: Any) -> None:
        """
        Function for adding token in case of
        `processing_type=padding_oov/padding/just_encode`.
        """
        # max_size is None then we will never reach maximum capacity.
        not_reached_max = len(self._token_to_index) <= self._max_size if self._max_size else True
        if token not in self._token_to_index and not_reached_max:
            index = len(self._token_to_index)
            # json saves only strings as keys
            self._token_to_index[str(token)] = index
            self._index_to_token[str(index)] = token

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """Load class from `path`. Path is a directory title."""
        with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as file:
            namespace = cls(**json.load(file))
        namespace._from_saved = True
        dict_type = namespace._processing_type.dict_type
        namespace._token_to_index = dict_type.load(os.path.join(path, 'token_to_index.json'))
        namespace._index_to_token = dict_type.load(os.path.join(path, 'index_to_token.json'))
        namespace.eval()
        return namespace

    def save(self, path: str) -> None:
        """Save data at `path`. You should pass a directory title."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as file:
            json.dump(self.__func_params__['__init__'], file, ensure_ascii=False, indent=2)
        self._token_to_index.save(os.path.join(path, 'token_to_index.json'))
        self._index_to_token.save(os.path.join(path, 'index_to_token.json'))

    def eval(self) -> None:
        """Set evaluation mode."""
        self._token_to_index.eval()
        self._index_to_token.eval()

    def get_size(self) -> int:
        """Return the number of unique tokens registered for namespace."""
        return len(self._token_to_index)

    def token_to_index(self, token: Any) -> int:
        """Get index for `token`."""
        # Convert to string as json stores only string keys.
        return self._token_to_index[str(token)]

    def index_to_token(self, index: int) -> Any:
        """Get token for `index`."""
        # Convert to string as json stores only string keys.
        return self._index_to_token[str(index)]
