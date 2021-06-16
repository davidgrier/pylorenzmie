from . import LorenzMie
from .Aberrations import Aberrations
import numpy as np


class LMHologram(LorenzMie):
    '''
    Compute in-line hologram of a sphere

    ...

    Properties
    ----------
    alpha : float, optional
        weight of scattered field in superposition
    coefficients : numpy.ndarray
        coefficients of the first 8 Zernike polynomials
        describing geometric aberrations

    Methods
    -------
    hologram() : numpy.ndarray
        Computed hologram of sphere
    '''

    def __init__(self, *args,
                 alpha=1.,
                 **kwargs):
        self.aberrations = Aberrations(**kwargs)
        super(ALMHologram, self).__init__(*args, **kwargs)
        self.coefficients = self.aberrations.coefficients
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)

    @LorenzMie.coordinates.setter
    def coordinates(self, coordinates):
        LorenzMie.coordinates.fset(self, coordinates)
        self.aberrations.coordinates = coordinates
        
    @LorenzMie.properties.getter
    def properties(self):
        p = LorenzMie.properties.fget(self)
        p['alpha'] = self.alpha
        return p

    def hologram(self):
        '''Return hologram of sphere

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        try:
            field = self.alpha * self.field()
        except TypeError:
            return None
        field[0, :] += self.aberrations.field()
        hologram = np.sum(np.real(field * np.conj(field)), axis=0)
        return hologram
