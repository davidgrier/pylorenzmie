from pylorenzmie.theory.LorenzMie import LorenzMie
import unittest
import numpy as np

try:
    from pylorenzmie.theory.torchLorenzMie import TorchLorenzMie
    from pylorenzmie.theory.torchLorenzMieBatch import TorchLorenzMieBatch
    import torch as _torch
    _torch_available = _torch.cuda.is_available()
except Exception:
    _torch_available = False


@unittest.skipUnless(_torch_available, 'torch/triton/CUDA not available')
class TestTorchLorenzMie(unittest.TestCase):

    def setUp(self) -> None:
        self.shape = (64, 64)
        self.model = TorchLorenzMie()
        self.model.particle.r_p = [32., 32., 150.]
        self.model.particle.a_p = 0.75
        self.model.particle.n_p = 1.45
        self.model.coordinates = self.model.meshgrid(self.shape)

        self.ref = LorenzMie()
        self.ref.particle.r_p = [32., 32., 150.]
        self.ref.particle.a_p = 0.75
        self.ref.particle.n_p = 1.45
        self.ref.coordinates = self.model.coordinates

    def test_method(self) -> None:
        self.assertIn('torch', self.model.method)
        self.assertIn('numpy', self.model.method)

    def test_hologram_shape(self) -> None:
        h = self.model.hologram()
        npts = self.shape[0] * self.shape[1]
        self.assertEqual(h.shape, (npts,))

    def test_hologram_dtype(self) -> None:
        h = self.model.hologram()
        self.assertEqual(h.dtype, np.float64)

    def test_hologram_positive(self) -> None:
        h = self.model.hologram()
        self.assertTrue(np.all(h > 0.))

    def test_hologram_is_numpy_array(self) -> None:
        h = self.model.hologram()
        self.assertIsInstance(h, np.ndarray)

    def test_hologram_matches_numpy(self) -> None:
        h_torch = self.model.hologram()
        h_np = self.ref.hologram()
        np.testing.assert_allclose(h_torch, h_np, rtol=1e-4, atol=1e-6)

    def test_repr(self) -> None:
        r = repr(self.model)
        self.assertIsInstance(r, str)


@unittest.skipUnless(_torch_available, 'torch/triton/CUDA not available')
class TestTorchLorenzMieBatch(unittest.TestCase):

    def setUp(self) -> None:
        self.shape = (32, 32)
        self.model = TorchLorenzMieBatch()
        self.model.coordinates = self.model.meshgrid(self.shape)

        def sphere(x, y, z, a_p, n_p):
            from pylorenzmie.theory import Sphere
            p = Sphere()
            p.r_p = [x, y, z]
            p.a_p = a_p
            p.n_p = n_p
            return p

        self.particle_lists = [
            [sphere(16., 16., 150., 0.75, 1.45)],
            [sphere(10., 20., 200., 1.0,  1.40)],
            [sphere(20., 10., 100., 0.5,  1.50),
             sphere(24., 24., 180., 0.8,  1.42)],
        ]

    def test_batch_hologram_shape(self) -> None:
        B = len(self.particle_lists)
        npts = self.shape[0] * self.shape[1]
        h = self.model.batch_hologram(self.particle_lists)
        self.assertEqual(h.shape, (B, npts))

    def test_batch_hologram_dtype(self) -> None:
        h = self.model.batch_hologram(self.particle_lists)
        self.assertEqual(h.dtype, np.float64)

    def test_batch_hologram_positive(self) -> None:
        h = self.model.batch_hologram(self.particle_lists)
        self.assertTrue(np.all(h > 0.))

    def test_batch_hologram_is_numpy_array(self) -> None:
        h = self.model.batch_hologram(self.particle_lists)
        self.assertIsInstance(h, np.ndarray)

    def test_batch_matches_sequential(self) -> None:
        '''Each row of batch_hologram should match the single-hologram result.'''
        h_batch = self.model.batch_hologram(self.particle_lists)
        for i, plist in enumerate(self.particle_lists):
            self.model.particle = plist
            h_seq = self.model.hologram()
            with self.subTest(i=i):
                np.testing.assert_allclose(h_batch[i], h_seq, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
