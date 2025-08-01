from abc import (ABC, abstractmethod)
import json
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from pathlib import Path


class LMObject(ABC):
    '''
    Base class for pylorenzmie objects

    ...

    Attributes
    ----------
    properties: dict
        Dictionary of object properties

    directory: str
        Fully resolved directory to object definition

    Methods
    -------
    to_json(**kwargs): str
        Returns JSON string of object properties and values
        Accepts keywords for json.dumps

    from_json(s: str): None
        Load JSON string of properties

    to_pandas(**kwargs): pandas.Series
        Returns pandas Series of object properties and values.
        Accepts keywords for pandas.Series.

    from_pandas(s: pandas.Series): None
        Loads properties from pandas Series

    meshgrid(shape, corner, flatten, dtype): numpy.ndarray
        Returns coordinate system for Lorenz-Mie microscopy images

    '''

    Property = bool | int | float
    Properties = dict[str, Property]
    Image = NDArray[float] | NDArray[int]
    Coordinates = NDArray[float]

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return False
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

    def to_json(self, **kwargs) -> str:
        '''Returns JSON string of adjustable properties

        Parameters
        ----------
        Accepts all keywords of json.dumps()

        Returns
        -------
        str : string
            JSON-encoded string of properties
        '''
        def np_encoder(obj):
            if isinstance(obj, np.generic):
                return obj.item()

        return json.dumps(self.properties, default=np_encoder, **kwargs)

    def from_json(self, s: str) -> None:
        '''Loads JSON string of adjustable properties

        Parameters
        ----------
        s : str
            JSON-encoded string of properties
        '''
        self.properties = json.loads(s)

    def to_pandas(self, **kwargs) -> pd.Series:
        '''Returns pandas Series of adjustable properties

        Parameters
        ----------
        Accepts all keywords of pandas.Series

        Returns
        -------
        series: pandas Series
        '''
        return pd.Series(self.properties, **kwargs)

    def from_pandas(self, series: pd.Series) -> None:
        '''Loads adjustable properties from pandas Series

        Parameters
        ----------
        series: pandas Series
        '''
        self.properties = series.to_dict()

    @property
    def directory(self) -> Path:
        '''Returns fully-qualified path to source file'''
        return Path(__file__).parent.resolve()

    @staticmethod
    def meshgrid(shape: tuple[int, int],
                 corner: tuple[int, int] = (0., 0.),
                 flatten: bool = True,
                 dtype=float) -> Coordinates:
        '''Returns coordinate system for Lorenz-Mie microscopy images

        Parameters
        ----------
        shape : tuple
            (nx, ny) shape of the coordinate system

        Keywords
        --------
        corner : tuple
            (left, top) starting coordinates for x and y, respectively
        flatten : bool
            If False, coordinates shape is (2, nx, ny)
            If True, coordinates are flattened to (2, nx*ny)
            Default: True
        dtype : type
            Data type.
            Default: float

        Returns
        -------
        xy : numpy.ndarray
            Coordinate system
        '''
        ny, nx = shape
        left, top = corner
        x = np.arange(left, left + nx, dtype=dtype)
        y = np.arange(top, top + ny, dtype=dtype)
        xy = np.array(np.meshgrid(x, y))
        return xy.reshape((2, -1)) if flatten else xy
