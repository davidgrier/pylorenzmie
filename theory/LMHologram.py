#!/usr/bin/env python
# -*- coding:utf-8 -*-

from . import GeneralizedLorenzMie
import numpy as np

class LMHologram(GeneralizedLorenzMie):

    '''
    Compute in-line holograms of spheres

    ...

    Properties
    ----------
    alpha : float, optional
        weight of scattered field in superposition

    Methods
    -------
    hologram() : numpy.ndarray
        Computed hologram of sphere
    '''

    def __init__(self, *args, alpha=1., **kwargs):
        super(LMHologram, self).__init__(*args, **kwargs)
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)

    @property
    def properties(self):
        p = dict()
        p.update(self.particle.properties)
        p.update(self.instrument.properties)
        p.update({'alpha': self.alpha})
        for k in p.keys():
            if type(p[k]) is np.ndarray:
                p[k] = p[k].tolist()
        return p

    @properties.setter
    def properties(self, properties):
        for prop in properties.keys():
            if hasattr(self.particle, prop):
                setattr(self.particle, prop, properties[prop])
            elif hasattr(self.instrument, prop):
                setattr(self.instrument, prop, properties[prop])
            elif hasattr(self, prop):
                setattr(self, prop, properties[prop])
            else:
                msg = "{} is not a property of LMHologram"
                raise ValueError(msg.format(prop))

    def hologram(self):
        '''Return hologram of sphere

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        field = self.alpha * self.field()
        field[0, :] += 1.
        hologram = np.sum(np.real(field * np.conj(field)), axis=0)
        return hologram


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Instrument import coordinates
    from time import time

    shape = [201, 201]
    h = LMHologram(coordinates=coordinates(shape))
    h.particle.r_p = [125, 75, 100]
    h.particle.a_p = 0.9
    h.particle.n_p = 1.45
    h.instrument.wavelength = 0.447
    h.hologram()
    start = time()
    hol = h.hologram()
    print("Time to calculate {}".format(time() - start))
    plt.imshow(hol.reshape(shape), cmap='gray')
    plt.show()
