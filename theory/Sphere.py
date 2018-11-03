#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Particle import Particle


class Sphere(Particle):
    '''Abstraction of a spherical particle for Lorenz-Mie micrsocopy'''

    def __init__(self,
                 ap=None,
                 mp=None,
                 **kwargs):
        super(Sphere, self).__init__(**kwargs)
        if ap is None:
            self.ap = 1.
        else:
            self.ap = ap

        if mp is None:
            self.mp = 1.
        else:
            self.mp = mp

    @property
    def ap(self):
        return self._ap

    @ap.setter
    def ap(self, ap):
        self._ap = ap

    @property
    def mp(self):
        return self._mp

    @mp.setter
    def mp(self, mp):
        self._mp = mp


if __name__ == '__main__':
    s = Sphere()
    print(s.rp)
