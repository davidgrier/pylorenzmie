'''Base class and shared type aliases for pylorenzmie objects.'''

from abc import ABC, abstractmethod
import json
import logging
import numpy as np
import pandas as pd
from numpy.typing import NDArray


# Module-level type aliases — importable without going through LMObject.
Property = bool | int | float | str | None
Properties = dict[str, Property]
Image = NDArray[float] | NDArray[int]
Images = Image | list[Image]
Coordinates = NDArray[float]
Coefficients = NDArray[complex]
Field = NDArray[complex]
Result = pd.Series | pd.DataFrame
Results = Result | list[Result]


def meshgrid(shape: tuple[int, int],
             corner: tuple[float, float] = (0., 0.),
             flatten: bool = True,
             dtype: type = float) -> Coordinates:
    '''Pixel coordinate grid for holographic microscopy images.

    Parameters
    ----------
    shape : tuple[int, int]
        (ny, nx) dimensions of the grid.
    corner : tuple[float, float]
        (left, top) origin of the coordinate system in pixels.
        Default: (0., 0.).
    flatten : bool
        If True (default), return shape (2, ny*nx).
        If False, return shape (2, ny, nx).
    dtype : type
        Numeric type for the coordinate arrays.
        Default: float.

    Returns
    -------
    xy : numpy.ndarray
        Coordinate grid.
    '''
    ny, nx = shape
    left, top = corner
    x = np.arange(left, left + nx, dtype=dtype)
    y = np.arange(top, top + ny, dtype=dtype)
    xy = np.array(np.meshgrid(x, y))
    return xy.reshape((2, -1)) if flatten else xy


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
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.properties == other.properties

    @property
    @abstractmethod
    def properties(self) -> Properties:
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
        '''Load properties from a JSON string.

        Parameters
        ----------
        s : str
            JSON-encoded properties.
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
        '''
        return pd.Series(self.properties, **kwargs)

    def from_pandas(self, series: pd.Series) -> None:
        '''Load properties from a pandas Series.

        Parameters
        ----------
        series : pandas.Series
        '''
        self.properties = series.to_dict()

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        a = cls()
        print(a)
