import unittest
from pylorenzmie.theory import Particle


class TestParticle(unittest.TestCase):

    def setUp(self) -> None:
        self.particle = Particle()

    def test_setcoordinates(self) -> None:
        value = 100.
        self.particle.x_p = value
        self.assertEqual(self.particle.r_p[0], value)
        value += 1.
        self.particle.y_p = value
        self.assertEqual(self.particle.r_p[1], value)
        value += 1.
        self.particle.z_p = value
        self.assertEqual(self.particle.r_p[2], value)

    def test_setposition(self) -> None:
        value = [100., 200., 300.]
        self.particle.r_p = value
        self.assertEqual(self.particle.x_p, value[0])
        self.assertEqual(self.particle.y_p, value[1])
        self.assertEqual(self.particle.z_p, value[2])

    def test_properties(self) -> None:
        props = self.particle.properties
        value = props['x_p'] + 1.
        props['x_p'] = value
        self.particle.properties = props
        self.assertEqual(self.particle.x_p, value)

    def test_serialize(self) -> None:
        ser = self.particle.to_json()
        b = Particle()
        b.from_json(ser)
        self.assertEqual(self.particle.x_p, b.x_p)

    def test_pandas(self) -> None:
        ser = self.particle.to_pandas()
        b = Particle()
        b.from_pandas(ser)
        self.assertEqual(self.particle.x_p, b.x_p)

    def test_ab(self) -> None:
        ab = self.particle.ab()
        self.assertEqual(ab[0], 1.)

    def test_repr(self) -> None:
        s = repr(self.particle)
        self.assertTrue(isinstance(s, str))

    def test_meshgrid(self) -> None:
        nx = 32
        ny = 24
        xy = self.particle.meshgrid((nx, ny), flatten=False)
        self.assertTupleEqual(xy.shape, (2, nx, ny))
        xy = self.particle.meshgrid((nx, ny), flatten=True)
        self.assertTupleEqual(xy.shape, (2, nx*ny))


if __name__ == '__main__':
    unittest.main()
