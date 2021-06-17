from . import (Aberrations, LorenzMie)
import numpy as np


class LMHologram(object):
    '''
    Compute in-line hologram of a sphere

    ...

    Properties
    ----------
    aberrations : Aberrations
    lorenzmie : LorenzMie
    coordinates : numpy.ndarray

    Methods
    -------
    hologram() : numpy.ndarray
        Computed hologram of sphere
    '''

    def __init__(self,
                 alpha=1.,
                 coordinates=None,
                 **kwargs):
        self.alpha = alpha
        self.aberrations = Aberrations(**kwargs)
        self.lorenzmie = LorenzMie(**kwargs)
        self.coordinates = coordinates
        self.particle = self.lorenzmie.particle
        self.instrument = self.lorenzmie.instrument
        self.method = self.lorenzmie.method

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def coordinates(self):
        return self.lorenzmie.coordinates
    
    @coordinates.setter
    def coordinates(self, coordinates):
        self.aberrations.coordinates = coordinates
        self.lorenzmie.coordinates = coordinates

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
