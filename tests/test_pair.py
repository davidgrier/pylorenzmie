import unittest
import numpy as np
from pylorenzmie.theory import Pair, Sphere


class TestPair(unittest.TestCase):

    def setUp(self) -> None:
        self.pair = Pair()

    def test_initialization(self) -> None:
        self.assertEqual(len(self.pair), 2)
        for particle in self.pair:
            self.assertIsInstance(particle, Sphere)

    def test_independent_positions(self) -> None:
        r_p1 = [self.pair.x_p1, self.pair.y_p1, self.pair.z_p1]
        r_p2 = [self.pair.x_p2, self.pair.y_p2, self.pair.z_p2]
        self.assertTrue(np.allclose(self.pair.particles[0].r_p, r_p1))
        self.assertTrue(np.allclose(self.pair.particles[1].r_p, r_p2))
        self.assertTrue(np.allclose(self.pair.particles[0].r_0, [0., 0., 0.]))
        self.assertTrue(np.allclose(self.pair.particles[1].r_0, [0., 0., 0.]))

    def test_setattr_syncs_particles(self) -> None:
        self.pair.x_p1 = 5.
        self.pair.y_p1 = 6.
        self.pair.z_p1 = 7.
        self.pair.a_p1 = 0.6
        self.pair.n_p1 = 1.6
        self.pair.k_p1 = 0.1
        self.pair.x_p2 = -5.
        self.pair.y_p2 = -6.
        self.pair.z_p2 = -7.
        self.pair.a_p2 = 0.7
        self.pair.n_p2 = 1.4
        self.pair.k_p2 = 0.2

        p1, p2 = self.pair.particles
        self.assertTrue(np.allclose(p1.r_p, [5., 6., 7.]))
        self.assertEqual(p1.a_p, 0.6)
        self.assertEqual(p1.n_p, 1.6)
        self.assertEqual(p1.k_p, 0.1)
        self.assertTrue(np.allclose(p2.r_p, [-5., -6., -7.]))
        self.assertEqual(p2.a_p, 0.7)
        self.assertEqual(p2.n_p, 1.4)
        self.assertEqual(p2.k_p, 0.2)

    def test_particles_are_independent(self) -> None:
        self.pair.a_p1 = 0.3
        self.pair.a_p2 = 0.9
        self.assertNotEqual(self.pair.particles[0].a_p,
                            self.pair.particles[1].a_p)

    def test_properties(self) -> None:
        props = self.pair.properties
        expected = {'x_p1', 'y_p1', 'z_p1', 'a_p1', 'n_p1', 'k_p1',
                   'x_p2', 'y_p2', 'z_p2', 'a_p2', 'n_p2', 'k_p2'}
        self.assertEqual(set(props.keys()), expected)
        self.assertEqual(props['x_p1'], self.pair.x_p1)
        self.assertEqual(props['a_p2'], self.pair.a_p2)

    def test_properties_roundtrip(self) -> None:
        new_values = {'x_p1': 1., 'y_p1': 2., 'z_p1': 3.,
                     'a_p1': 0.4, 'n_p1': 1.4, 'k_p1': 0.01,
                     'x_p2': -1., 'y_p2': -2., 'z_p2': -3.,
                     'a_p2': 0.5, 'n_p2': 1.5, 'k_p2': 0.02}
        self.pair.properties = new_values
        self.assertEqual(self.pair.properties, new_values)
        self.assertTrue(np.allclose(self.pair.particles[0].r_p,
                                    [1., 2., 3.]))
        self.assertTrue(np.allclose(self.pair.particles[1].r_p,
                                    [-1., -2., -3.]))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
