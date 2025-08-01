from pylorenzmie.theory.LorenzMie import LorenzMie
import unittest
import numpy as np
import sys


mods = ['cupy', 'LorenzMie']
for mod in mods:
    if mod in sys.modules:
        sys.modules.pop(mod)


class TestLorenzMie(unittest.TestCase):

    def setUp(self) -> None:
        self.method = LorenzMie()
        self.shape = [256, 256]

    def test_repr(self) -> None:
        r = repr(self.method)
        self.assertIsInstance(r, str)

    def test_method(self) -> None:
        self.assertEqual(self.method.method, 'numpy')

    def test_meshgrid(self) -> None:
        nx = 32
        ny = 24
        xy = self.method.meshgrid((nx, ny), flatten=False)
        self.assertTupleEqual(xy.shape, (2, nx, ny))
        xy = self.method.meshgrid((nx, ny), flatten=True)
        self.assertTupleEqual(xy.shape, (2, nx*ny))

    def test_coordinates_none(self) -> None:
        self.method.coordinates = None
        self.assertIsInstance(self.method.coordinates, np.ndarray)

    def test_coordinates_point(self) -> None:
        point = np.array([1, 2, 3]).reshape((3, 1))
        self.method.coordinates = point
        self.assertTrue(np.allclose(self.method.coordinates, point))

    def test_coordinates_1d(self) -> None:
        c = np.arange(self.shape[0])
        self.method.coordinates = c
        self.assertEqual(self.method.coordinates.shape[0], 3)

    def test_coordinates_2d(self) -> None:
        c = self.method.meshgrid(self.shape)
        self.method.coordinates = c
        self.assertEqual(self.method.coordinates.shape[0], 3)
        self.assertTrue(np.allclose(self.method.coordinates[0:2, :], c))

    def test_coordinates_3dlist(self) -> None:
        c = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        self.method.coordinates = c
        self.assertTrue(np.allclose(self.method.coordinates, np.array(c)))

    def test_properties(self) -> None:
        '''Get properties, change one, and set properties'''
        value = -42
        p = dict(x_p=value)
        self.method.properties = p
        self.assertEqual(self.method.particle.x_p, value)

    def test_serialize(self) -> None:
        n_0 = 1.5
        n_1 = 1.4
        self.method.particle.n_p = n_0
        s = self.method.to_json()
        self.method.particle.n_p = n_1
        self.method.from_json(s)
        self.assertEqual(self.method.particle.n_p, n_0)

    def test_field_nocoordinates(self) -> None:
        self.method.coordinates = None
        field = self.method.field()
        self.assertIsInstance(field, np.ndarray)

    def test_field(self, bohren=False, cartesian=False) -> None:
        p = self.method.particle
        p.a_p = 1.
        p.n_p = 1.4
        p.r_p = [64, 64, 100]
        c = self.method.meshgrid([128, 128])
        self.method.coordinates = c
        field = self.method.field(bohren=bohren, cartesian=cartesian)
        self.assertEqual(field.shape[1], c.shape[1])

    def test_field_bohren(self) -> None:
        self.test_field(bohren=True)

    def test_field_cartesian(self) -> None:
        self.test_field(cartesian=True)

    def test_field_both(self) -> None:
        self.test_field(bohren=True, cartesian=True)

    def test_hologram(self) -> None:
        p = self.method.particle
        p.a_p = 1.
        p.n_p = 1.4
        p.r_p = [64, 64, 100]
        c = self.method.meshgrid([128, 128])
        self.method.coordinates = c
        holo = self.method.hologram()
        self.assertEqual(holo.size, c.shape[1])


if __name__ == '__main__':
    unittest.main()
