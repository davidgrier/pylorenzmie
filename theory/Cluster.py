from pylorenzmie.lib import LMObject
from typing import Optional
import numpy as np


class Cluster(LMObject):

    def __init__(self,
                 x_p: Optional[float] = None,
                 y_p: Optional[float] = None,
                 z_p: Optional[float] = None) -> None:
        self._block = False
        self.x_p = x_p or 0.
        self.y_p = y_p or 0.
        self.z_p = z_p or 100.
        self.particles = list()

    @property
    def x_p(self) -> float:
        return self._x_p

    @x_p.setter
    def x_p(self, x_p: float) -> None:
        self._x_p = x_p
        self._update()

    @property
    def y_p(self) -> float:
        return self._y_p

    @y_p.setter
    def y_p(self, y_p: float) -> None:
        self._y_p = y_p
        self._update()

    @property
    def z_p(self) -> float:
        return self._z_p

    @z_p.setter
    def z_p(self, z_p: float) -> None:
        self._z_p = z_p
        self._update()

    @property
    def r_p(self) -> np.array:
        return np.array([self.x_p, self.y_p, self.z_p])

    @r_p.setter
    def r_p(self, r_p: np.array) -> None:
        self._block = True
        self.x_p, self.y_p, self.z_p = r_p
        self._block = False
        self._update()

    def _update(self) -> None:
        if self._block:
            return
        for particle in self.particles:
            particle.r_0 = self.r_p

