# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def gaussian(x, mu, sig):
    '''Gaussian function for radial distribution'''
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class Mask(object):
    '''
    Stores information about an algorithm's general and
    parameter specific options during fitting.

    ...

    Properties
    ----------
    coordinates : ndarray (3, npix)

    percentpix : float
        percentage of pixels to sample

    distribution : str
        probability distribution for random sampling

    index : ndarray (nsampled)

    exclude : ndarray
    '''

    def __init__(self, coordinates,
                 percentpix=0.1,
                 distribution='fast',
                 exclude=None):
        self.d_map = {'uniform': self.uniform_distribution,
                      'radial': self.radial_distribution,
                      'donut': self.donut_distribution,
                      'fast': self.fast_distribution}
        
        self._coordinates = coordinates
        self._percentpix = percentpix
        self._distribution = distribution
        self._exclude = exclude or []
        self.update()

    @property
    def percentpix(self):
        return self._percentpix

    @percentpix.setter
    def percentpix(self, value):
        self._percentpix = np.clip(float(value), 0, 1)
        self.update()

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, name):
        if name in self.d_map:
            self._distribution = name
        else:
            self._distribution = 'fast'
        self.update()

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates
        self.update()

    @property
    def index(self):
        return self._index

    @property
    def exclude(self):
        return self._exclude

    @exclude.setter
    def exclude(self, exclude):
        self._exclude = exclude

    # Various sampling probability distributions

    def uniform_distribution(self):
        npts = self.coordinates[0].size
        rho = np.ones(npts)
        return rho

    def radial_distribution(self):
        npts = self.coordinates[0].size
        ext_size = int(np.sqrt(npts))
        x0, x1 = self.coordinates[0, [0,-1]]
        y0, y1 = self.coordinates[1, [0,-1]]
        nrows = y1 - y0
        ncols = x1 - x0
        center = np.array([int(ncols/2.) + x0, int(nrows/2.) + y0])

        # mean and stdev of gaussian as percentages of max radius
        mu = 0.6 * ext_size/2.
        sigma = 0.2 * ext_size/2.

        pixels = self.coordinates[:2, :]
        dist = np.linalg.norm(pixels.T - center, axis=1)
        rho = gaussian(dist, mu, sigma)
        return rho

    def donut_distribution(self):
        npts = self.coordinates[0].size
        ext_size = int(np.sqrt(npts))
        x0, x1 = self.coordinates[0, [0,-1]]
        y0, y1 = self.coordinates[1, [0,-1]]
        nrows = y1 - y0
        ncols = x1 - x0
        center = np.array([int(ncols/2.) + x0, int(nrows/2.) + y0])

        # outer concentric circle lies at 0% of edge
        outer = 0.0
        # inner concentric circle lies at 100% of edge
        inner = 1.

        radius1 = ext_size * (1/2 - outer)
        radius2 = ext_size * (1/2 - inner)

        pixels = self.coordinates[:2, :]
        dist = np.linalg.norm(pixels.T - center, axis=1)
        rho = np.where((dist > radius2) & (dist < radius1), 10., 1.)
        return rho

    def fast_distribution(self):
        return None

    def get_distribution(self):
        return self.d_map[self.distribution]()

    def update(self):
        if self.coordinates is None:
            self._index = None
            return
        npts = self.coordinates[0].size
        if self.percentpix == 1.:
            index = np.delete(np.arange(npts), self.exclude)
        else:
            nchosen = int(npts * self.percentpix)
            rho = self.get_distribution()
            if rho is not None:
                rho[self.exclude] = 0.
                rho /= np.sum(rho)
            index = np.random.choice(npts, nchosen,
                                     p=rho, replace=False)
        self._index = index

    # Return coordinates array from sampled indices
    def masked_coords(self):
        return np.take(self.coordinates, self.index, axis=1)


if __name__ == '__main__':
    from pylorenzmie.theory.Instrument import coordinates

    shape = (201, 201)
    corner = (350, 300)
    m = Mask(coordinates(shape, corner=corner))
    m.settings['percentpix'] = 0.4
    m.settings['distribution'] = 'radial_gaussian'
    m.exclude = np.arange(10000, 12000)
    m.initialize_sample()
    m.draw_mask()
