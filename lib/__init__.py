from .meshgrid import meshgrid
from .LMObject import (LMObject,
                       Property, Properties,
                       Image, Images,
                       Coordinates, Coefficients, Field,
                       Result, Results)
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
           'Property', 'Properties',
           'Image', 'Images',
           'Coordinates', 'Coefficients', 'Field',
           'Result', 'Results',
           'Azimuthal', 'CircleTransform', 'circletransform',
           'coordinates']
