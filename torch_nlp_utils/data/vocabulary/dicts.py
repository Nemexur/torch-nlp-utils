from typing import Type, Any, T
import json
from overrides import overrides


class DictModule(dict):
    """
    Module for dictionary that adds:
    loading, saving and evaluation for particular dict subclass.
    """

    def eval(self) -> None:
        """Set evaluation mode."""
        pass

    def save(self, path: str) -> None:
        """Save data at `path`."""
        raise NotImplementedError()

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """Load class from `path`."""
        raise NotImplementedError()


class PassThroughDict(DictModule):
    """
    Dict that returns key as a value on each call.
    If eval is called, it would raise an error
    incase you try to get an item
    that has not been in train data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._unique = set()
        self._eval_mode = False

    def __getitem__(self, key: int):
        # Convert to int as it works only with int values.
        key = int(key)
        if self._eval_mode:
            if key in self._unique:
                return key
            else:
                raise Exception("Invalid key has been passed. It was not in train.")
        else:
            return key

    def __setitem__(self, key: int, value: Any):
        # Convert to int as it works only with int values.
        key = int(key)
        if not self._eval_mode:
            self._unique.add(key)

    def __len__(self):
        return len(self._unique)

    @overrides
    def eval(self):
        """Set evaluation mode."""
        self._eval_mode = True

    @overrides
    def save(self, path: str) -> None:
        """Save data at `path`."""
        with open(path, "w", encoding="utf-8") as file:
            json.dump(list(self._unique), file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """Load class from `path`."""
        cls_instance = cls()
        with open(path, "r", encoding="utf-8") as file:
            cls_instance._unique = set(json.load(file))
        return cls_instance


class NamespaceDict(DictModule):
    """
    Dict that works just like `defaultdict`
    which returns certain value for unseen key
    but this implementation is much simpler to save
    and it doesn't need default value to be a function.

    If `oov` is None then we do not consider that
    there could be out-of-vocabulary tokens
    """

    def __init__(self, oov: Any = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._oov = oov

    @property
    def oov_value(self):
        return self._oov

    @oov_value.setter
    def oov_value(self, value: Any):
        self._oov = value

    def __missing__(self, key: Any):
        if self._oov is None:
            raise Exception("Invalid key.")
        return self._oov

    @overrides
    def save(self, path: str) -> None:
        """Save data at `path`."""
        params = {"oov_value": self._oov, "dict": dict(self)}
        with open(path, "w", encoding="utf-8") as file:
            json.dump(params, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """Load class from `path`."""
        cls_instance = cls()
        with open(path, "r", encoding="utf-8") as file:
            params = json.load(file)
            cls_instance.oov_value = params["oov_value"]
            cls_instance.update(params["dict"])
        return cls_instance
