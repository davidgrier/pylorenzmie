import unittest
import numpy as np
from pylorenzmie.theory import Dimer, Sphere


class TestDimer(unittest.TestCase):

    def setUp(self) -> None:
        self.dimer = Dimer()

    def test_initialization(self) -> None:
        self.assertEqual(len(self.dimer), 2)
        for particle in self.dimer:
            self.assertIsInstance(particle, Sphere)
            self.assertEqual(particle.a_p, self.dimer.a_p)
            self.assertEqual(particle.n_p, self.dimer.n_p)
            self.assertEqual(particle.k_p, self.dimer.k_p)

    def test_update_positions(self) -> None:
        a_p = 1.0
        theta = np.pi / 3
        phi = np.pi / 4
        magnification = 0.05

        self.dimer.a_p = a_p
        self.dimer.theta = theta
        self.dimer.phi = phi
        self.dimer.magnification = magnification

        x_p = np.cos(theta) * np.cos(phi)
        y_p = np.cos(theta) * np.sin(phi)
        z_p = np.sin(theta)
        r_p = (a_p / magnification) * np.array([x_p, y_p, z_p])

        self.assertTrue(np.allclose(self.dimer.particles[0].r_p, r_p))
        self.assertTrue(np.allclose(self.dimer.particles[1].r_p, -r_p))

    def test_setattr_particles(self) -> None:
        new_a_p = 1.5
        new_n_p = 1.55
        new_k_p = 0.1

        self.dimer.a_p = new_a_p
        self.dimer.n_p = new_n_p
        self.dimer.k_p = new_k_p

        for particle in self.dimer:
            self.assertEqual(particle.a_p, new_a_p)
            self.assertEqual(particle.n_p, new_n_p)
            self.assertEqual(particle.k_p, new_k_p)

    def test_invalid_particle_count(self) -> None:
        # Manually set particles to an invalid count
        self.dimer.particles = [Sphere(a_p=0.75, n_p=1.45, k_p=0.)]
        self.dimer.update_positions()  # Should not raise an error
        self.assertEqual(len(self.dimer), 1)


if __name__ == '__main__':
    unittest.main()
