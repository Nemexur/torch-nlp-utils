from typing import (
    Type, T, Dict,
    List, Any, Callable,
    NamedTuple, Tuple
)
import os
import json
from .dicts import DictModule
from .statistics import Statistics
from .processing_type import ProcessingType as PT
from torch_nlp_utils.common.utils import save_params


TARGET_TYPES: Dict[str, Callable[[], Statistics]] = {
    'oneclass': lambda: Statistics.from_params(type='target'),
    'multiclass': lambda: Statistics.from_params(type='target'),
    'multilabel': lambda: Statistics.from_params(type='multilabel_target'),
    'regression': lambda: Statistics.from_params(type='regression_target')
}


class Encoders(NamedTuple):
    """NamedTuple of Encoder Dictionaries for a Namespace."""
    token_to_index: DictModule
    index_to_token: DictModule


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
    target : `str`, optional (default = `None`)
        Target type if namespace is associated with target variable.
        Supported types:
            - oneclass - binary classification.
            - multiclass - multiple classes classification.
            - multilabel - classification with multilabels.
            - regression - ordinary regression.
    """
    @save_params
    def __init__(
        self,
        processing_type: str,
        max_size: int = None,
        target: str = None
    ) -> None:
        if not hasattr(PT, processing_type):
            raise ValueError('Invalid processing type has been passed.')
        if max_size is not None and max_size <= 0:
            raise Exception('Max size can not be less or equal 0.')
        if target is not None and target not in TARGET_TYPES:
            raise Exception('Invalid target type.')
        self._max_size = max_size
        self._processing_type = getattr(PT, processing_type)
        self._encoders = Encoders(*self._processing_type.get_dicts())
        self.statistics = TARGET_TYPES[target]() if target else Statistics()

    @property
    def encoders(self) -> Tuple[DictModule, DictModule]:
        return self._encoders

    @encoders.setter
    def encoders(self, encoders: Tuple[DictModule, DictModule]) -> None:
        self._encoders = encoders

    def add_tokens(self, tokens: List[Any]) -> None:
        """Add list of `tokens` to namespace."""
        for token in tokens:
            if not hasattr(self._processing_type, PT.__add_token_func__):
                raise Exception(
                    f'You need to register func to add tokens for {self._processing_type.name}.'
                )
            # We need to pass self as well
            getattr(self._processing_type, PT.__add_token_func__)(self, token)
        # Update statistics at last.
        self.statistics.update_stats(tokens)

    @PT.register_for(case=PT.pass_through)
    def _pass_through(self, token: Any) -> None:
        """Function for adding token in case of `processing_type=pass_through`."""
        if not (isinstance(token, int) or str(token).isdigit()):
            raise Exception(
                'processing_type=pass_through is only supported for int values.'
            )
        self._encoders.token_to_index[token] = token
        self._encoders.index_to_token[token] = token

    @PT.register_for(case=[PT.padding_oov, PT.padding, PT.just_encode])
    def _add_token(self, token: Any) -> None:
        """
        Function for adding token in case of
        `processing_type=padding_oov/padding/just_encode`.
        """
        # max_size is None then we will never reach maximum capacity.
        not_reached_max = len(self._encoders.token_to_index) <= self._max_size if self._max_size else True
        if token not in self._encoders.token_to_index and not_reached_max:
            index = len(self._encoders.token_to_index)
            # json saves only strings as keys
            self._encoders.token_to_index[str(token)] = index
            self._encoders.index_to_token[str(index)] = token

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """Load class from `path`. Path is a directory title."""
        with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as file:
            namespace = cls(**json.load(file))
        dict_type = namespace._processing_type.dict_type
        namespace.encoders = Encoders(
            token_to_index=dict_type.load(os.path.join(path, 'token_to_index.json')),
            index_to_token=dict_type.load(os.path.join(path, 'index_to_token.json'))
        )
        namespace.statistics = namespace.statistics.__class__.load(os.path.join(path, 'statistics.json'))
        namespace.eval()
        return namespace

    def save(self, path: str) -> None:
        """Save data at `path`. You should pass a directory title."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as file:
            json.dump(self.__func_params__['__init__'], file, ensure_ascii=False, indent=2)
        self._encoders.token_to_index.save(os.path.join(path, 'token_to_index.json'))
        self._encoders.index_to_token.save(os.path.join(path, 'index_to_token.json'))
        self.statistics.save(os.path.join(path, 'statistics.json'))

    def eval(self) -> None:
        """Set evaluation mode."""
        self._encoders.token_to_index.eval()
        self._encoders.index_to_token.eval()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for the namespace."""
        return self.statistics.get_statistics()

    def get_size(self) -> int:
        """Return the number of unique tokens registered for namespace."""
        return len(self._encoders.token_to_index)

    def token_to_index(self, token: Any) -> int:
        """Get index for `token`."""
        # Convert to string as json stores only string keys.
        return self._encoders.token_to_index[str(token)]

    def index_to_token(self, index: int) -> Any:
        """Get token for `index`."""
        # Convert to string as json stores only string keys.
        return self._encoders.index_to_token[str(index)]
