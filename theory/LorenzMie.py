#!/usr/bin/env python
# -*- coding: utf-8 -*-

from GeneralizedLorenzMie import GeneralizedLorenzMie
from Sphere import Sphere


class LorenzMie(GeneralizedLorenzMie):

    def __init__(self,
                 a_p=None,
                 n_p=None,
                 particle=None,
                 **kwargs):
        super(LorenzMie, self).__init__(**kwargs)
        self.particle = Sphere()
        if a_p is not None:
            self.particle.a_p = a_p
        if n_p is not None:
            self.particle.n_p = n_p
