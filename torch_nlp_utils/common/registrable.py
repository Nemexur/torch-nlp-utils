from typing import (
    Type, Dict, Optional, List
)
from loguru import logger
from .extra_typing import T
from collections import defaultdict
from .from_params import FromParams
from .checks import ConfigurationError


class Registrable(FromParams):
    """
    Any class that inherits from `Registrable` gains access to a named registry for its
    subclasses. To register them, just decorate them with the classmethod
    `@BaseClass.register(name)`.
    After which you can call `BaseClass.by_name(name)` to get the corresponding subclass.
    Note that the registry stores the subclasses themselves; not class instances.
    In most cases you would then call `from_params(params)` on the returned subclass.
    """
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)

    @classmethod
    def register(cls: Type[T], name: str, exist_ok=False):
        """
        Register a class under a particular name.

        Parameters
        ----------
        name: `str`
            The name to register the class under.
        exist_ok: `bool`, optional (default=False)
            If True, overwrites any existing models registered under `name`. Else,
            throws an error if a model is already registered under `name`.
        """
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                if exist_ok:
                    message = (
                        f"{name} has already been registered as {registry[name].__name__}, but "
                        f"exist_ok=True, so overwriting with {cls.__name__}."
                    )
                    logger.info(message)
                else:
                    message = (
                        f"Cannot register {name} as {cls.__name__}; "
                        f"name already in use for {registry[name].__name__}."
                    )
                    raise ValueError(message)
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: Optional[str]) -> Optional[Type[T]]:
        """
        Return subclass by its name.

        Parameters
        ----------
        name : `Optional[str]`, required
            Name of registered subclass.

        Returns
        -------
        `Optional[Type[T]]`
            Returns type registered to this name.

        Raises
        ------
        `ConfigurationError`
            Raises an error if subclass is not registered.
        """
        if name is None:
            # if passed name as None probably because we tried to
            # initialize a subclass of registried class
            return None
        logger.debug(f"Instantiating registered subclass {name} of {cls}.")
        if name in Registrable._registry[cls]:
            return Registrable._registry[cls].get(name)
        else:
            # is not a qualified class name
            raise ConfigurationError(
                f"{name} is not a registered name for {cls.__name__}."
            )

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List registered subclass of a certain class.

        Returns
        -------
        `List[str]`
            List of subclasses.
        """
        return list(Registrable._registry[cls].keys())
