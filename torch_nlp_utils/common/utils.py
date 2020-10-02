from typing import Iterable, List, Any, Iterator, Type, Callable, T
import os
import inspect
from itertools import islice
from functools import partialmethod
from torch_nlp_utils.settings import ROOT_DIR

# Constants
CACHE_DIRECTORY = os.path.join(ROOT_DIR, ".torch_nlp_utils_cache")


def partialclass(cls: Type[T], *args, **kwargs) -> Type[T]:
    """
    Just like `partialmethod` from `functools`
    it performs partial init for classes.
    `Args` and `kwargs` are parameters that needs to be fixed
    for class init.
    """

    class PartialCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialCls


def save_params(func: Callable) -> Callable:
    """
    Decorator to save parameters of function call for the class.

    It works only for functions
    that doesn't have `*args` in its arguments list.
    This decorator only saves parameters for the latest call.
    """

    def wrapper(cls: Type[T], *args, **kwargs) -> Callable:
        cls.__func_params__ = getattr(cls, "__func_params__", {})
        call_params = {arg: value for arg, value in zip(inspect.getfullargspec(func).args[1:], args)}
        call_params.update(kwargs)
        cls.__func_params__[func.__name__] = call_params
        return func(cls, *args, **kwargs)

    return wrapper


def lazy_groups_of(iterable: Iterable[List[Any]], group_size: int) -> Iterator[List[List[Any]]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    # lazy_groups_of adopted from:
    # https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break
