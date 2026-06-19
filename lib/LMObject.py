'''Base class and shared type aliases for pylorenzmie objects.'''

from abc import ABC, abstractmethod
import json
import logging
import numpy as np
import pandas as pd
from .meshgrid import meshgrid
from .types import (Property, Properties,
                    Image, Images,
                    Coordinates, Coefficients, Field,
                    Result, Results)


class LMObject(ABC):
    '''Base class for pylorenzmie objects.

    Provides the ``properties`` protocol (used for both serialization and
    optimization), JSON and pandas I/O, equality comparison, and a
    class-scoped logger.

    Attributes
    ----------
    properties : dict
        Dictionary of adjustable object properties.  Concrete subclasses
        must override the getter; the base-class setter applies any key
        that matches an existing attribute and logs a debug message for
        unknown keys.

    Notes
    -----
    ``LMObject`` instances are mutable and therefore unhashable
    (``__hash__`` is explicitly ``None``).

    The type aliases below are re-exported at class scope for backward
    compatibility.  Prefer importing them directly from
    ``pylorenzmie.lib``.
    '''

    # Type aliases re-exported at class scope for backward compatibility.
    Property = Property
    Properties = Properties
    Image = Image
    Images = Images
    Coordinates = Coordinates
    Coefficients = Coefficients
    Field = Field
    Result = Result
    Results = Results

    # Module-level meshgrid re-exported at class scope for backward
    # compatibility.  Import from pylorenzmie.lib directly instead.
    meshgrid = staticmethod(meshgrid)

    # Mutable objects should not be hashable.
    __hash__ = None

    @property
    def logger(self) -> logging.Logger:
        '''Logger named after the concrete class.'''
        return logging.getLogger(
            f'{self.__class__.__module__}.{self.__class__.__qualname__}')

    def __eq__(self, other: object) -> bool:
        '''Compare by properties dict rather than identity.

        Returns ``NotImplemented`` for objects of a different type so
        that Python can try the reflected operation; this is preferable
        to returning ``False`` outright.
        '''
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.properties == other.properties

    @property
    @abstractmethod
    def properties(self) -> Properties:
        '''Adjustable parameters of this object.

        Returns a flat ``dict`` mapping parameter names to their current
        values.  Only parameters included here are visible to the
        serialization methods and to ``Optimizer`` during fitting.

        Subclasses must override the getter using::

            @ParentClass.properties.getter
            def properties(self) -> Properties:
                props = super().properties
                props.update(...)
                return props

        The base-class getter returns an empty dict; the base-class
        setter iterates over the supplied dict and calls ``setattr`` for
        every key that already exists as an attribute.  Unknown keys are
        silently ignored and logged at DEBUG level.
        '''
        return dict()

    @properties.setter
    def properties(self, properties: Properties) -> None:
        for name, value in properties.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                self.logger.debug('Ignoring unknown property: %s', name)

    def to_json(self, **kwargs) -> str:
        '''Serialize properties to a JSON string.

        NumPy scalars are automatically converted to native Python types
        before serialization.

        Parameters
        ----------
        **kwargs
            Passed through to ``json.dumps``.

        Returns
        -------
        str
            JSON-encoded properties.
        '''
        def np_encoder(obj):
            if isinstance(obj, np.generic):
                return obj.item()

        return json.dumps(self.properties, default=np_encoder, **kwargs)

    def from_json(self, s: str) -> None:
        '''Update properties from a JSON string.

        Mutates the object in place by assigning to ``self.properties``.

        Parameters
        ----------
        s : str
            JSON-encoded properties, as produced by ``to_json``.
        '''
        self.properties = json.loads(s)

    def to_pandas(self, **kwargs) -> pd.Series:
        '''Serialize properties to a pandas Series.

        Parameters
        ----------
        **kwargs
            Passed through to ``pandas.Series``.

        Returns
        -------
        pandas.Series
            Index is the property names; values are the property values.
        '''
        return pd.Series(self.properties, **kwargs)

    def from_pandas(self, series: pd.Series) -> None:
        '''Update properties from a pandas Series.

        Mutates the object in place by assigning to ``self.properties``.

        Parameters
        ----------
        series : pandas.Series
            As produced by ``to_pandas``.
        '''
        self.properties = series.to_dict()

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        a = cls()
        print(a)
