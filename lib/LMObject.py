from abc import (ABC, abstractmethod)
from typing import (Optional, Dict, Any)
import json
import pandas as pd
import numpy as np
from pathlib import Path


Properties = Dict[str, float]


class LMObject(ABC):

    '''
    Base class for pylorenzmie objects

    ...

    Attributes
    ----------
    properties: dict
        Dictionary of object properties

    Methods
    -------
    to_json(**kwargs): str
        Returns JSON string of object properties and values
        Accepts keywords for json.dumps

    from_json(s: str): None
        Load JSON string of properties

    '''

    @property
    @abstractmethod
    def properties(self) -> Properties:
        return dict()

    @properties.setter
    def properties(self, properties: Properties) -> None:
        for name, value in properties.items():
            if hasattr(self, name):
                setattr(self, name, value)

    def to_json(self, **kwargs: Optional[Any]) -> str:
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

    def to_pandas(self, **kwargs: Optional[Any]) -> pd.Series:
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
    def directory(self):
        return Path(__file__).parent.resolve()
