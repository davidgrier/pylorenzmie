import unittest

import numpy as np

from pylorenzmie.theory import AberratedLorenzMie
from pylorenzmie.theory.LorenzMie import LorenzMie


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
        nrows = 32
        ncols = 24
        xy = self.method.meshgrid((nrows, ncols), flatten=False)
        self.assertTupleEqual(xy.shape, (2, nrows, ncols))
        xy = self.method.meshgrid((nrows, ncols), flatten=True)
        self.assertTupleEqual(xy.shape, (2, nrows * ncols))

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

    def test_field_on_axis_pixel_is_finite(self) -> None:
        # kx = ky = 0: the azimuth is undefined (0/0), but the field is not
        p = self.method.particle
        p.a_p = 1.
        p.n_p = 1.4
        p.r_p = [0., 0., 100.]
        self.method.coordinates = np.array([[0., 3., 0.],
                                            [0., 0., 4.],
                                            [0., 0., 0.]])
        for cartesian in (False, True):
            field = self.method.field(cartesian=cartesian)
            self.assertTrue(np.isfinite(field).all())
        # on the axis the field is x-polarized: no y or z component
        field = self.method.field(cartesian=True)
        self.assertAlmostEqual(abs(field[1, 0]), 0.)
        self.assertAlmostEqual(abs(field[2, 0]), 0.)
        self.assertGreater(abs(field[0, 0]), 0.)

    def test_hologram(self) -> None:
        p = self.method.particle
        p.a_p = 1.
        p.n_p = 1.4
        p.r_p = [64, 64, 100]
        c = self.method.meshgrid([128, 128])
        self.method.coordinates = c
        holo = self.method.hologram()
        self.assertEqual(holo.size, c.shape[1])


class TestPropertiesAreStrict(unittest.TestCase):
    '''A name that the model does not have is an error, not a no-op.'''

    def setUp(self) -> None:
        self.model = LorenzMie()

    def test_a_known_name_is_set(self) -> None:
        self.model.properties = {'a_p': 0.9, 'wavelength': 0.5}
        self.assertEqual(self.model.single().a_p, 0.9)
        self.assertEqual(self.model.instrument.wavelength, 0.5)

    def test_a_misspelled_name_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "'a_pp'"):
            self.model.properties = {'a_pp': 0.9}

    def test_every_unknown_name_is_named(self) -> None:
        with self.assertRaisesRegex(ValueError, "'b_p'.*'c_p'"):
            self.model.properties = {'a_p': 0.9, 'b_p': 1., 'c_p': 2.}

    def test_nothing_is_set_if_a_name_is_unknown(self) -> None:
        before = self.model.properties
        with self.assertRaises(ValueError):
            self.model.properties = {'a_p': 0.9, 'a_pp': 0.9}
        self.assertEqual(self.model.properties, before)

    def test_a_method_is_not_a_property(self) -> None:
        '''hasattr is not the test: a method would have been replaced.'''
        with self.assertRaises(ValueError):
            self.model.properties = {'hologram': 1.}
        self.assertTrue(callable(self.model.hologram))

    def test_the_aberrations_of_an_aberrated_model_are_properties(self) -> None:
        model = AberratedLorenzMie()
        model.properties = {'defocus': 0.1, 'piston': 0.2}
        self.assertEqual(model.defocus, 0.1)
        self.assertEqual(model.piston, 0.2)

    def test_an_aberration_is_not_a_property_of_a_plain_model(self) -> None:
        with self.assertRaises(ValueError):
            self.model.properties = {'defocus': 0.1}


class TestWhatABackendDeclares(unittest.TestCase):
    '''The base class declares what the backends and the wrappers rely on,
    so that none of them has to probe for it.'''

    def setUp(self) -> None:
        self.model = LorenzMie()

    def test_defaults(self) -> None:
        self.assertFalse(self.model.parallel)
        self.assertEqual(self.model.parallel_threshold, 2048)
        self.assertTrue(self.model.fork_safe)
        self.assertEqual(self.model.jac_params, frozenset())
        self.assertEqual(self.model.piston, 0.)
        self.assertIsNone(self.model._aberration)
        self.assertFalse(self.model._aberrations_in_kernel)

    def test_an_array_is_a_field_as_it_is(self) -> None:
        array = np.ones((3, 4), dtype=complex)
        self.assertIs(self.model._as_field(array), array)

    def test_there_is_no_analytical_jacobian_to_ask_for(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, 'LorenzMie'):
            self.model.jac()

    def test_an_aberrated_model_provides_its_phase_mask(self) -> None:
        model = AberratedLorenzMie()
        self.assertTrue(callable(model._aberration))
        self.assertEqual(model._aberration(np.array([0., 0., 100.])), 1.)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
