from pylorenzmie.theory import LorenzMie
import numpy as np


class LMHologram(LorenzMie):
    '''
    Compute in-line hologram of a sphere

    ...

    Properties
    ----------
    alpha : float
        Relative amplitude of scattered field.
        Default: 1

    Methods
    -------
    hologram() : numpy.ndarray
        Computed hologram of sphere
    '''

    def __init__(self,
                 alpha=1.,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha or 1.

    def __str__(self):
        fmt = '<{}(alpha={})>'
        return fmt.format(self.__class__.__name__, self.alpha)

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

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
