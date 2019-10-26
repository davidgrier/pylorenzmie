# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# Rescale distribution so sum = 1.
def normalize(distribution):
    total = np.sum(distribution)
    normed = distribution / total
    return normed

# Gaussian function for radial gaussian distribution


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class Mask(object):
    '''
    Stores information about an algorithm's general and
    parameter specific options during fitting.

    ...

    Attributes
    ----------
    coordinates: ndarray (3, npix)

    settings: dict
              'percentpix': percent of pixels to sample
              'distribution': probability distribution for random sampling

    sampled_index: ndarray (nsampled)

    exclude: ndarray
    '''

    def __init__(self, coordinates, exclude=[]):
        self.coordinates = coordinates
        self.settings = {'percentpix': 0.1,        # default settings
                         'distribution': 'radial_gaussian'}
        self._exclude = exclude
        if coordinates is not None:
            img_size = coordinates[0].size
            self._sampled_index = np.arange(int(0.1*img_size))
        else:
            self._sampled_index = None

    @property
    def sampled_index(self):
        return self._sampled_index

    @sampled_index.setter
    def sampled_index(self, sample):
        self._sampled_index = sample

    @property
    def exclude(self):
        return self._exclude

    @exclude.setter
    def exclude(self, exclude):
        self._exclude = exclude

    # Various sampling probability distributions

    def uniform_distribution(self):
        img_size = self.coordinates[0].size
        distribution = np.ones(img_size)
        return distribution

    def radial_gaussian(self):  # it's like a donut, but ~ hazier ~
        img_size = self.coordinates[0].size
        ext_size = int(np.sqrt(img_size))
        distribution = np.ones(img_size)
        leftcorner, rightcorner, topcorner, botcorner = [int(x) for sublist in [list(
            coord[::len(coord)-1]) for coord in self.coordinates[:2]] for x in sublist]
        numrows = botcorner - topcorner
        numcols = rightcorner - leftcorner
        center = (int(numcols/2.)+leftcorner, int(numrows/2.)+topcorner)

        # mean and stdev of gaussian as percentages of max radius
        mu_ = 0.6
        sigma_ = 0.2

        mu = ext_size*1/2 * mu_
        sigma = ext_size*1/2*sigma_

        pixels = self.coordinates[:2, :]
        dist = np.linalg.norm(pixels.T - center, axis=1)
        distribution *= gaussian(dist, mu, sigma)

        return distribution

    def donut_distribution(self):
        img_size = self.coordinates[0].size
        ext_size = int(np.sqrt(img_size))
        distribution = np.ones(img_size)
        leftcorner, rightcorner, topcorner, botcorner = [int(x) for sublist in [list(
            coord[::len(coord)-1]) for coord in self.coordinates[:2]] for x in sublist]
        numrows = botcorner - topcorner
        numcols = rightcorner - leftcorner
        center = np.array([int(numcols/2.)+leftcorner,
                           int(numrows/2.)+topcorner])

        # outer concetric circle lies at 0% of edge
        outer = 0.0
        # inner concentric circle lies at 100% of edge
        inner = 1.

        radius1 = ext_size * (1/2 - outer)
        radius2 = ext_size * (1/2 - inner)

        pixels = self.coordinates[:2, :]
        dist = np.linalg.norm(pixels.T - center, axis=1)
        distribution = np.where((dist > radius2) & (dist < radius1),
                                distribution * 10,
                                distribution)

        return distribution

    def get_distribution(self):
        d_name = self.settings['distribution']
        if d_name == 'uniform':
            distribution = self.uniform_distribution()
        elif d_name == 'donut':
            distribution = self.donut_distribution()
        elif d_name == 'radial_gaussian':
            distribution = self.radial_gaussian()
        else:
            raise ValueError(
                "Invalid distribution name")

        distribution[self.exclude] = 0.
        distribution = normalize(distribution)
        return distribution

    # Get new pixels to sample
    def initialize_sample(self):
        totalpix = int(self.coordinates[0].size)
        percentpix = float(self.settings['percentpix'])
        if percentpix == 1.:
            sampled_index = np.delete(np.arange(totalpix),
                                      self.exclude)
        elif percentpix <= 0 or percentpix > 1:
            raise ValueError(
                "percent of pixels must be a value between 0 and 1.")
        else:
            p_dist = self.get_distribution()
            numpixels = int(totalpix*percentpix)
            sampled_index = np.random.choice(
                totalpix, numpixels, p=p_dist, replace=False)
        self.sampled_index = sampled_index

    # Draw sampled and excluded pixels
    def draw_mask(self):
        maskcoords = self.masked_coords()
        maskx, masky = maskcoords[:2]
        excluded = self.exclude
        excludex = self.coordinates[0][excluded]
        excludey = self.coordinates[1][excluded]
        plt.scatter(excludex, excludey, color='blue', alpha=1, s=1, lw=0)
        plt.scatter(maskx, masky, color='red', alpha=1, s=1, lw=0)
        plt.title('sampled pixels')
        plt.show()

    # Return coordinates array from sampled indices
    def masked_coords(self):
        return np.take(self.coordinates, self._sampled_index, axis=1)


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
