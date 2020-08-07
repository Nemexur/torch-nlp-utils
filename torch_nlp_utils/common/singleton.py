from typing import Dict, Type, T


class Singleton:
    """
    `Decorator` that turns your class into a `Singleton`.
    By definition of a `Singleton` pattern this class must not accept any arguments
    during initialization so this `decorator` works only with such classes.
    """
    _registry: Dict[str, Type[T]] = {}

    def __init__(self, cls: Type[T]) -> None:
        self._cls: Type[T] = cls
        self._shared: Type[T] = None

    @property
    def shared(self) -> Type[T]:
        """Get `Singleton` instance."""
        if not self._shared:
            self._shared = self._init_class()
            return self._shared
        else:
            return self._shared

    def _init_class(self) -> Type[T]:
        """Initialize an instance of class and save it into dictionary for mapping."""
        title = self._cls.__name__.lower()
        if title not in Singleton._registry:
            Singleton._registry[title] = self._cls()
        return Singleton._registry[title]
