from .meshgrid import meshgrid
from .LMObject import LMObject
from .CircleTransform import (CircleTransform, circletransform)
from . import Azimuthal
from warnings import warn


def coordinates(*args, **kwargs):
    warn('''
    coordinates() is deprecated and will be removed in a future release.
    Use meshgrid() from pylorenzmie.lib instead.
    ''', DeprecationWarning, 2)
    return meshgrid(*args, **kwargs)


__all__ = ['LMObject', 'meshgrid',
           'Azimuthal', 'CircleTransform', 'circletransform',
           'coordinates']
