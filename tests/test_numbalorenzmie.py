import unittest

import numpy as np

from pylorenzmie.theory import Sphere
from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.Aberrated import Aberrated

try:
    from pylorenzmie.theory.numbaLorenzMie import (numbaLorenzMie,
                                                   compute_field_jit)
except ImportError:                                  # pragma: no cover
    numbaLorenzMie = None


@unittest.skipIf(numbaLorenzMie is None, 'numba is not available')
class TestNumbaLorenzMie(unittest.TestCase):

    def setUp(self):
        self.n = 41
        self.coordinates = LorenzMie.meshgrid((self.n, self.n))
        self.particle = Sphere(a_p=0.75, n_p=1.45, x_p=19.3, y_p=22.1,
                               z_p=180.)

    def models(self, **kwargs):
        numpy = LorenzMie(coordinates=self.coordinates)
        numba = numbaLorenzMie(coordinates=self.coordinates, **kwargs)
        for m in (numpy, numba):
            m.particle = self.particle
        return numpy, numba

    def kdr(self, model):
        k = model.instrument.wavenumber()
        r_p = self.particle.r_p + self.particle.r_0
        return k * (model.coordinates - r_p[:, None])

    def ab(self, model):
        inst = model.instrument
        return self.particle.ab(inst.n_m, inst.wavelength)

    def test_method(self):
        self.assertEqual(numbaLorenzMie.method, 'numba numpy')

    def test_field_matches_numpy(self):
        numpy, numba = self.models()
        kdr, ab = self.kdr(numpy), self.ab(numpy)
        for cartesian in (True, False):
            for bohren in (True, False):
                ref = numpy.lorenzmie(ab, kdr, cartesian=cartesian,
                                      bohren=bohren).copy()
                new = numba.lorenzmie(ab, kdr, cartesian=cartesian,
                                      bohren=bohren)
                scale = np.abs(ref).max()
                self.assertLess(np.abs(new - ref).max(), 1e-10 * scale,
                                (cartesian, bohren))

    def test_large_particle_matches_numpy(self):
        self.particle = Sphere(a_p=3., n_p=1.6, x_p=19.3, y_p=22.1,
                               z_p=300.)
        numpy, numba = self.models()
        kdr, ab = self.kdr(numpy), self.ab(numpy)
        ref = numpy.lorenzmie(ab, kdr).copy()
        new = numba.lorenzmie(ab, kdr)
        self.assertLess(np.abs(new - ref).max(), 1e-10 * np.abs(ref).max())

    def test_special_points(self):
        # on the axis (k rho = 0), above and below, and in the plane of
        # the particle (k z = 0); all outside the sphere, where the
        # partial-wave series applies (k a = 14)
        numpy, numba = self.models()
        ab = self.ab(numpy)
        kdr = np.array([[0., 0., 25., -20., 35.],
                        [0., 0., 0., 30., -10.],
                        [-30., 40., 0., 0., 0.]])
        numpy.coordinates = numba.coordinates = np.zeros((3, 5))
        ref = numpy.lorenzmie(ab, kdr).copy()
        new = numba.lorenzmie(ab, kdr)
        self.assertTrue(np.all(np.isfinite(new)))
        self.assertLess(np.abs(new - ref).max(), 1e-10 * np.abs(ref).max())

    def test_point_at_the_particle_center_gives_nan_not_an_exception(self):
        # k r = 0 divides by zero; Numba's default error model would
        # raise ZeroDivisionError, whereas NumPy returns NaN
        numpy, numba = self.models()
        ab = self.ab(numpy)
        kdr = np.array([[0., 3., 0.],
                        [0., 4., 0.],
                        [0., -60., 25.]])
        numpy.coordinates = numba.coordinates = np.zeros((3, 3))
        with np.errstate(all='ignore'):
            ref = numpy.lorenzmie(ab, kdr).copy()
        new = numba.lorenzmie(ab, kdr)
        self.assertTrue(np.all(np.isnan(new[:, 0])))
        self.assertTrue(np.all(np.isnan(ref[:, 0])))
        for i in (1, 2):
            self.assertLess(np.abs(new[:, i] - ref[:, i]).max(),
                            1e-10 * np.abs(ref[:, i]).max())

    def test_hologram_matches_numpy(self):
        numpy, numba = self.models()
        self.assertTrue(np.allclose(numpy.hologram(), numba.hologram(),
                                    rtol=0, atol=1e-10))

    def test_cluster_hologram_matches_numpy(self):
        second = Sphere(a_p=0.5, n_p=1.58, x_p=10., y_p=12.,
                        z_p=210.)
        numpy, numba = self.models()
        numpy.particle = numba.particle = [self.particle, second]
        self.assertTrue(np.allclose(numpy.hologram(), numba.hologram(),
                                    rtol=0, atol=1e-10))

    def test_parallel_matches_serial_exactly(self):
        _, serial = self.models()
        _, parallel = self.models()
        serial.parallel = False
        parallel.parallel = True
        parallel.parallel_threshold = 1
        self.assertTrue(np.array_equal(serial.hologram(),
                                       parallel.hologram()))

    def test_threshold_does_not_change_the_result(self):
        _, a = self.models()
        _, b = self.models()
        a.parallel_threshold = 1
        b.parallel_threshold = 10**9
        self.assertTrue(np.array_equal(a.hologram(), b.hologram()))

    def test_concurrent_calls_from_threads_are_safe(self):
        # Numba's workqueue threading layer aborts the process if two
        # Python threads enter a parallel region at the same time
        from concurrent.futures import ThreadPoolExecutor
        _, reference = self.models()
        reference.parallel_threshold = 1
        expected = reference.hologram().copy()

        def work(_):
            _, m = self.models()
            m.parallel_threshold = 1
            for _ in range(10):
                h = m.hologram()
            return np.array_equal(h, expected)

        with ThreadPoolExecutor(6) as pool:
            self.assertTrue(all(pool.map(work, range(12))))

    def test_parallel_is_the_default(self):
        self.assertTrue(numbaLorenzMie.parallel)
        self.assertGreater(numbaLorenzMie.parallel_threshold, 0)

    def test_parallel_flag_can_be_set_per_instance_and_per_class(self):
        _, a = self.models()
        _, b = self.models()
        a.parallel = False
        self.assertTrue(b.parallel)
        self.assertFalse(a.parallel)

    def test_kdr_with_a_different_number_of_points(self):
        # the buffers belong to the coordinates; other sizes are allowed
        numpy, numba = self.models()
        subset = self.coordinates[:, ::7]
        reference = LorenzMie(coordinates=subset)
        reference.particle = self.particle
        kdr, ab = self.kdr(reference), self.ab(reference)
        ref = reference.lorenzmie(ab, kdr).copy()
        new = numba.lorenzmie(ab, kdr)
        self.assertEqual(new.shape, kdr.shape)
        self.assertLess(np.abs(new - ref).max(), 1e-10 * np.abs(ref).max())

    def test_noncontiguous_kdr(self):
        numpy, numba = self.models()
        ab = self.ab(numpy)
        big = np.asfortranarray(self.kdr(numpy))
        ref = numpy.lorenzmie(ab, np.ascontiguousarray(big)).copy()
        new = numba.lorenzmie(ab, big)
        self.assertLess(np.abs(new - ref).max(), 1e-10 * np.abs(ref).max())

    def test_unused_buffers_are_released(self):
        _, numba = self.models()
        self.assertEqual(numba.buffers[0].size, 0)
        self.assertEqual(numba.buffers[1].size, 0)
        self.assertEqual(numba.buffers[2].shape, (3, self.n * self.n))
        self.assertEqual(numba.buffers[3].shape, (3, self.n * self.n))

    def test_aberrated_numba_matches_aberrated_numpy(self):
        numpy = Aberrated(LorenzMie)(coordinates=self.coordinates,
                                     spherical=4., defocus=2.)
        numba = Aberrated(numbaLorenzMie)(coordinates=self.coordinates,
                                          spherical=4., defocus=2.)
        for m in (numpy, numba):
            m.particle = self.particle
        self.assertTrue(np.allclose(numpy.hologram(), numba.hologram(),
                                    rtol=0, atol=1e-10))

    def test_compute_field_jit_wrapper(self):
        numpy, numba = self.models()
        kdr, ab = self.kdr(numpy), self.ab(numpy)
        buffers = [np.empty(kdr.shape, dtype=complex) for _ in range(4)]
        new = compute_field_jit(kdr, buffers, None, ab, True, True)
        ref = numpy.lorenzmie(ab, kdr)
        self.assertLess(np.abs(new - ref).max(), 1e-10 * np.abs(ref).max())


if __name__ == '__main__':
    unittest.main()
