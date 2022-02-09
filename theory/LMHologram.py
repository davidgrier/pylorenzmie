from pylorenzmie.theory.Aberrations import Aberrations
from pylorenzmie.theory import LorenzMie
import numpy as np


class LMHologram(object):
    '''
    Compute in-line hologram of a sphere

    ...

    Properties
    ----------
    alpha : float
        Relative amplitude of scattered field.
        Default: 1
    aberrations : Aberrations
    lorenzmie : LorenzMie
    properties : dict
        Properties that can be accessed and set

    NOTE: Properties of aberrations and lorenzmie can
    be accessed directly for convenience

    Methods
    -------
    hologram() : numpy.ndarray
        Computed hologram of sphere
    '''

    def __init__(self,
                 alpha=1.,
                 **kwargs):
        super().__setattr__('alpha', alpha)
        super().__setattr__('aberrations', Aberrations(**kwargs))
        super().__setattr__('lorenzmie', LorenzMie(**kwargs))

    def __str__(self):
        return f'<{self.__class__.__name__}(alpha={self.alpha})>'

    def __repr__(self):
        return str(self)

    def __getattr__(self, key):
        if hasattr(self.lorenzmie, key):
            return getattr(self.lorenzmie, key)
        elif hasattr(self.aberrations, key):
            return getattr(self.aberrations, key)

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        for component in [self.lorenzmie, self.aberrations]:
            if hasattr(component, key):
                setattr(component, key, value)

    @property
    def properties(self):
        p = self.lorenzmie.properties
        p['alpha'] = self.alpha
        p.update(self.aberrations.properties)
        return p

    @properties.setter
    def properties(self, properties):
        for name, value in properties.items():
            if hasattr(self, name):
                setattr(self, name, value)
        self.lorenzmie.properties = properties
        self.aberrations.properties = properties

    def hologram(self):
        '''Return hologram of sphere

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        try:
            field = self.alpha * self.lorenzmie.field()
        except TypeError:
            return None
        field[0, :] += self.aberrations.field()
        hologram = np.sum(np.real(field * np.conj(field)), axis=0)
        return hologram
