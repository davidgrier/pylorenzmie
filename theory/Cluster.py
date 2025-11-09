from pylorenzmie.theory import Particle
from dataclasses import dataclass, field


@dataclass
class Cluster(Particle):

    particles: list[Particle] = field(repr=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.update()

    def __setattr__(self, key: str, value: Particle.Property) -> None:
        super().__setattr__(key, value)
        if key in ['x_p', 'y_p', 'z_p', 'particles']:
            self._update()
            return

    def __len__(self) -> int:
        return len(self.particles)

    def __next__(self) -> Particle:
        if self._index < len(self.particles):
            result = self.particles[self._index]
            self._index += 1
            return result
        else:
            self._index = 0
            raise StopIteration

    def __getitem__(self, index: int) -> Particle:
        if index < 0 or index >= len(self):
            raise IndexError('Particle index out of range')
        return self.particles[index]

    def _update(self) -> None:
        try:
            for particle in self.particles:
                particle.r_0 = self.r_p
        except AttributeError:
            pass

    def update(self) -> None:
        self._update()
