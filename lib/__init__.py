from typing import Any
from .lmtypes import Coordinates
from .meshgrid import meshgrid
from .LMObject import LMObject
from .CircleTransform import (CircleTransform, circletransform)
from . import Azimuthal


def coordinates(*args: Any, **kwargs: Any) -> Coordinates:
    '''Deprecated alias of :func:`meshgrid`.'''
    from warnings import warn
    warn('coordinates() is deprecated; use meshgrid() from pylorenzmie.lib.',
         DeprecationWarning,
         stacklevel=2)
    return meshgrid(*args, **kwargs)


__all__ = ['LMObject', 'meshgrid',
           'Azimuthal', 'CircleTransform', 'circletransform',
           'coordinates']
