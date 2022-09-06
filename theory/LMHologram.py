from pylorenzmie.theory import LorenzMie
from typing import Optional, Any
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

    def __init__(self, alpha: float = 1., **kwargs: Optional[Any]) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha or 1.

    def __str__(self) -> str:
        fmt = '<{}(alpha={})>'
        return fmt.format(self.__class__.__name__, self.alpha)

    def __repr__(self) -> str:
        return str(self)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value

    @LorenzMie.properties.getter
    def properties(self) -> dict:
        p = LorenzMie.properties.fget(self)
        p['alpha'] = self.alpha
        return p

    def hologram(self) -> np.ndarray:
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


def main():
    a = LMHologram()
    print(a)
