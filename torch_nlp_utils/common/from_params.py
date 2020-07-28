"""
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
This implementation of FromParams is a bit simplified by getting rid of some special functionality.
Copyright by the AllenNLP authors.
"""

from typing import Type, T
from loguru import logger
from .checks import ConfigurationError


class FromParams:
    """
    Mixin to give a from_params method to classes. We create a distinct base class for this
    because sometimes we want non-Registrable classes to be instantiatable from_params.
    """

    @classmethod
    def from_params(cls: Type[T], **params) -> T:
        """
        This is the automatic implementation of `from_params`. Any class that subclasses `FromParams`
        (or `Registrable`, which itself subclasses `FromParams`) gets this implementation for free.
        If you want your class to be instantiated from params in the "obvious" way -- pop off parameters
        and hand them to your constructor with the same names -- this provides that functionality.
        """

        from .registrable import Registrable  # import here to avoid circular imports

        logger.info(
            f"Instantiating class {cls} from params.",
            feature='f-strings'
        )

        if params is None:
            raise ConfigurationError(
                "We cannot instantiate any class with params as None."
            )
        registered_subclasses = Registrable._registry.get(cls)
        if registered_subclasses is not None:
            subclass_type = params.pop('type', None)
            if subclass_type is None:
                raise ConfigurationError(
                    "We cannot instantiate subclass without its type."
                )
            subclass = registered_subclasses[subclass_type]
            logger.info(
                f'Instantiating class {subclass} inherited from {cls}.',
                feature='f-strings'
            )
            return subclass(**params)
        else:
            logger.warning(
                f"""
                {cls} is not registered so we can not instantiate any subclass from it.
                Probably it is a subclass, so we try to instantiate it.
                """,
                feature='f-strings'
            )
            return cls(**params)
