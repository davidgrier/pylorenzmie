from abc import ABC, abstractmethod
from typing import Optional, Any
import json


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
    def properties(self) -> dict:
        return dict()

    @properties.setter
    def properties(self, properties: dict) -> None:
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
        return json.dumps(self.properties, **kwargs)

    def from_json(self, s: str) -> None:
        '''Loads JSON string of adjustable properties

        Parameters
        ----------
        s : str
            JSON-encoded string of properties
        '''
        self.properties = json.loads(s)
