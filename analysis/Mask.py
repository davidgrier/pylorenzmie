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
    coordinates : numpy.ndarray
        [ndim, npix] pixel coordinates
    percentpix : float
        percentage of pixels to sample
    distribution : str
        name of the probability distribution for random sampling
    exclude : numpy.ndarray
        indexes of pixels to exclude from mask
    mask : numpy.ndarray

    '''

    def __init__(self,
                 coordinates=None,
                 percentpix=0.1,
                 distribution='fast',
                 exclude=None,
                 **kwargs):

        self.d_map = {'uniform': self._uniform_distribution,
                      'radial': self._radial_distribution,
                      'donut': self._donut_distribution,
                      'fast': self._fast_distribution}

        self._percentpix = percentpix
        self._distribution = distribution
        self._exclude = exclude or []
        self.coordinates = coordinates

    @property
    def coordinates(self):
        '''Distance of each pixel from center of feature'''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates
        if coordinates is not None:
            center = np.mean(coordinates, axis=1)
            self._distance = np.linalg.norm(coordinates.T - center, axis=1)
        self._update()
        
    @property
    def percentpix(self):
        '''Percentage of pixels to sample, expressed as a fraction'''
        return self._percentpix

    @percentpix.setter
    def percentpix(self, value):
        self._percentpix = np.clip(float(value), 0, 1)
        self._update()

    @property
    def distribution(self):
        '''Name of the probability distribution'''
        return self._distribution

    @distribution.setter
    def distribution(self, name):
        if name in self.d_map:
            self._distribution = name
        else:
            self._distribution = 'fast'
        self._update()

    @property
    def exclude(self):
        '''indexes of pixels to exclude from fits'''
        return self._exclude

    @exclude.setter
    def exclude(self, exclude):
        self._exclude = exclude

    @property
    def selected(self):
        return self._selected

    # Various sampling probability distributions

    def _uniform_distribution(self):
        npts = len(self._distance)
        return np.ones(npts)

    def _radial_distribution(self):
        # mean and stdev of gaussian as percentages of max radius
        extent = np.max(self._distance)
        mu = 0.2 * extent
        sigma = 0.3 * extent
        return gaussian(self._distance, mu, sigma)

    def _donut_distribution(self):
        outer = 0.0
        inner = 1.
        dist = self._distance
        radius1 = np.max(dist) * (1/2 - outer)
        radius2 = np.max(dist) * (1/2 - inner)
        return np.where((dist > radius2) & (dist < radius1), 10., 1.)

    def _fast_distribution(self):
        return None

    def _get_distribution(self):
        return self.d_map[self.distribution]()

    def _update(self):
        if self._coordinates is None:
            self._selected = None
            return
        npts = self._distance.size
        if self.percentpix >= 1.:
            index = np.delete(np.arange(npts), self.exclude)
        else:
            nchosen = int(npts * self.percentpix)
            rho = self._get_distribution()
            if rho is not None:
                rho[self.exclude] = 0.
                rho /= np.sum(rho)
            index = np.random.choice(npts, nchosen,
                                     p=rho, replace=False)
        self._selected = np.full(npts, False)
        self._selected[index] = True


if __name__ == '__main__':  # pragma: no cover
    from pylorenzmie.utilities import coordinates

    shape = (201, 201)
    corner = (350, 300)
    m = Mask(coordinates(shape, corner=corner))
    m.settings['percentpix'] = 0.4
    m.settings['distribution'] = 'radial_gaussian'
    m.exclude = np.arange(10000, 12000)
    m.initialize_sample()
    m.draw_mask()
