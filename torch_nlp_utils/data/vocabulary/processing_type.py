from typing import (
    Type, T, Dict, NamedTuple,
    List, Callable, Union
)
from enum import Enum
from copy import deepcopy
from .metadata import TokensMetadata
from .dicts import PassThroughDict, NamespaceDict


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
            NamespaceDict(1, {TokensMetadata.PAD: 0, TokensMetadata.OOV: 1}),
            NamespaceDict(TokensMetadata.OOV, {0: TokensMetadata.PAD, 1: TokensMetadata.OOV})
        ]
    )
    padding = ProcessingTypeProperties(
        dict_type=NamespaceDict,
        dicts=[
            NamespaceDict(0, {TokensMetadata.PAD: 0}),
            NamespaceDict(TokensMetadata.PAD, {0: TokensMetadata.PAD})
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
