from .LMObject import LMObject
from .CircleTransform import (CircleTransform, circletransform)
from warnings import warn


def coordinates(*args, **kwargs):
    warn('''
    coordinates() is deprecated and will be removed in a future release.
    Use LMObject.meshgrid() instead.
    ''', DeprecationWarning, 2)
    return LMObject.meshgrid(*args, **kwargs)


__all__ = 'LMObject Azimuthal CircleTransform circletransform'.split()
