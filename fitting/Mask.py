# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Various sampling probability distributions


def normalize(distribution):
    total = np.sum(distribution)
    normed = np.array([float(i)/total for i in distribution])
    return normed


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
        self.settings = {'percentpix':0.1,
                         'distribution': 'donut'}
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


    def uniform_distribution(self):
        img_size = self.coordinates[0].size
        distribution = np.ones(img_size)
        distribution[self.exclude] = 0.
        distribution = normalize(distribution)
        return distribution


    def donut_distribution(self):
        img_size = self.coordinates[0].size
        ext_size = int(np.sqrt(img_size))
        distribution = np.ones(img_size)
        numrows = np.amax(self.coordinates[1])
        numcols = np.amax(self.coordinates[0])
        leftcorner = int(np.amin(self.coordinates[0]))
        topcorner = int(np.amin(self.coordinates[1]))
        center = (int(numcols/2.)+leftcorner, int(numrows/2.)+topcorner)
        #outer concetric circle lies at 10% of edge
        outer = 0.1
        #inner concentric circle lies at 30% of edge
        inner = 0.3
    
        radius1 = ext_size* (1/2 - outer)
        radius2 = ext_size* (1/2 - inner)
        for i in range(img_size):
            pixel = self.coordinates[:2,i]
            dist = np.linalg.norm(pixel-center)
            if dist > radius2 and dist < radius1:
                distribution[i] *= 10
            #elif dist < radius2:
            #    distribution[i] *= 3
        distribution[self.exclude] = 0.
        distribution = normalize(distribution)
        return distribution

    def get_distribution(self):
        d_name = self.settings['distribution']
        if d_name=='uniform':
            return self.uniform_distribution()
        elif d_name=='donut':
            return self.donut_distribution()
    
    def initialize_sample(self):
        totalpix = int(self.coordinates[0].size)
        percentpix = float(self.settings['percentpix'])
        if percentpix == 1.:
            self.sampled_index = [x for x in np.arange(totalpix) if x not in self.exclude]
        elif percentpix <= 0 or percentpix > 1:
            raise ValueError(
                "percent of pixels must be a value between 0 and 1.")
        else:
            p_dist = self.get_distribution()
            numpixels = int(totalpix*percentpix)
            sampled_index = np.sort(np.random.choice(totalpix, numpixels, p=p_dist, replace=False))
            self._sampled_index = sampled_index
        #check that none of the excluded pixels are in sampled_index
        wrong = [x for x in self._sampled_index if x in self.exclude]
        if len(wrong) != 0:
            print('Wrong!')

    def draw_mask(self):
        maskcoords = self.masked_coords()
        maskx, masky = maskcoords[:2]
        excluded = self.exclude
        excludex = [self.coordinates[0][x] for x in excluded]
        excludey = [self.coordinates[1][x] for x in excluded]
        plt.scatter(excludex, excludey, color='blue', alpha=1, s=1, lw=0)
        plt.scatter(maskx, masky, color='red', alpha=1, s=1, lw=0)
        plt.title('sampled pixels')
        plt.show()

        
    def masked_coords(self):
        original_coords = self.coordinates
        directions = len(original_coords)
        new_coords = []
        for i in range(directions):
            localc = np.array([original_coords[i][x] for x in self._sampled_index])
            new_coords.append(localc)
        new_coords = np.array(new_coords)
        return new_coords

        
    

if __name__ == '__main__':
    from pylorenzmie.theory.Instrument import coordinates

    shape = (11, 11)
    m = Mask(coordinates(shape))
    m.settings['percentpix'] = 0.5
    m.exclude = [15]
    m.initialize_sample()
    m.draw_mask()
    
