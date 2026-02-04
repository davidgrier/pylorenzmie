import unittest
import numpy as np
from pylorenzmie.theory import Cluster, Particle


class TestCluster(unittest.TestCase):

    def setUp(self) -> None:
        self.p1 = Particle(x_p=10., y_p=20., z_p=30.)
        self.p2 = Particle(x_p=40., y_p=50., z_p=60.)
        self.cluster = Cluster(particles=[self.p1, self.p2])

    def test_len(self) -> None:
        self.assertEqual(len(self.cluster), 2)

    def test_getitem(self) -> None:
        self.assertEqual(self.cluster[0], self.p1)
        self.assertEqual(self.cluster[1], self.p2)
        with self.assertRaises(IndexError):
            _ = self.cluster[2]

    def test_iteration(self) -> None:
        particles = list(self.cluster)
        self.assertEqual(particles[0], self.p1)
        self.assertEqual(particles[1], self.p2)

    def test_update_positions(self) -> None:
        new_x = 100.
        new_y = 200.
        new_z = 300.
        self.cluster.x_p = new_x
        self.cluster.y_p = new_y
        self.cluster.z_p = new_z
        for particle in self.cluster:
            self.assertEqual(particle.r_0[0], new_x)
            self.assertEqual(particle.r_0[1], new_y)
            self.assertEqual(particle.r_0[2], new_z)

    def test_setattr_particles(self) -> None:
        new_particles = [Particle(x_p=1., y_p=2., z_p=3.),
                         Particle(x_p=4., y_p=5., z_p=6.)]
        self.cluster.particles = new_particles
        self.assertEqual(len(self.cluster), 2)
        self.assertEqual(self.cluster[0], new_particles[0])
        self.assertEqual(self.cluster[1], new_particles[1])


if __name__ == '__main__':
    unittest.main()
