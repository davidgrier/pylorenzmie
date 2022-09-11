from pylorenzmie.theory import Particle
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Cluster(Particle):

    particles: list = field(repr=False, default_factory=list)

    def __setattr__(self, key: str, value: Any) -> None:
        super().__setattr__(key, value)
        self._update()

    def __iter__(self) -> iter:
        return iter(self.particles)

    def __len__(self) -> int:
        return len(self.particles)

    def __getitem__(self, index: int) -> Particle:
        return self.particles[index]

    def _update(self) -> None:
        try:
            for particle in self.particles:
                particle.r_0 = self.r_p
        except AttributeError:
            pass
