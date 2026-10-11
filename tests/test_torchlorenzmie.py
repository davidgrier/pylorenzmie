from pylorenzmie.theory.LorenzMie import LorenzMie
import unittest
import numpy as np

try:
    import torch as _torch
    from pylorenzmie.theory.torchLorenzMie import TorchLorenzMie, get_device
    _device = get_device().type          # 'cuda' or 'mps'
except Exception:
    _device = None

try:
    from pylorenzmie.theory.torchLorenzMieBatch import TorchLorenzMieBatch
    _batch_available = _device in ('cuda', 'xpu')
except Exception:
    _batch_available = False

# float32 on the GPU against float64 on the CPU
ATOL = 1e-4


@unittest.skipUnless(_device, 'torch with a CUDA or Apple (MPS) GPU '
                              'not available')
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
        np.testing.assert_allclose(h_torch, h_np, rtol=1e-4, atol=ATOL)

    def test_device(self) -> None:
        self.assertEqual(self.model.device.type, _device)

    def test_field_conventions_match_numpy(self) -> None:
        for kwargs in (dict(), dict(cartesian=False), dict(bohren=False),
                       dict(cartesian=False, bohren=False)):
            with self.subTest(**kwargs):
                f = self.model.field(**kwargs).cpu().numpy()
                np.testing.assert_allclose(f, self.ref.field(**kwargs),
                                           atol=ATOL)

    def test_particle_below_focal_plane(self) -> None:
        for model in (self.model, self.ref):
            model.particle.z_p = -150.
        np.testing.assert_allclose(self.model.hologram(),
                                   self.ref.hologram(), atol=ATOL)

    def test_cluster_matches_numpy(self) -> None:
        from pylorenzmie.theory import Cluster, Sphere
        particles = [Sphere(a_p=.5, n_p=1.5, x_p=20., y_p=32., z_p=200.),
                     Sphere(a_p=.7, n_p=1.4, x_p=44., y_p=32., z_p=210.)]
        for model in (self.model, self.ref):
            model.particle = Cluster(particles=list(particles))
        np.testing.assert_allclose(self.model.hologram(),
                                   self.ref.hologram(), atol=ATOL)

    def test_repeated_calls_follow_the_particle(self) -> None:
        # nothing stale may survive from one call to the next
        rng = np.random.default_rng(0)
        for _ in range(20):
            for model in (self.model, self.ref):
                model.particle.properties = dict(
                    a_p=rng.uniform(.3, 1.2), n_p=rng.uniform(1.4, 1.7))
            self.ref.particle.properties = self.model.particle.properties
            np.testing.assert_allclose(self.model.hologram(),
                                       self.ref.hologram(), atol=1e-3)

    def test_aberrated_matches_numpy(self) -> None:
        from pylorenzmie.theory import Aberrated
        kwargs = dict(spherical=2., defocus=.5, coma=.3, x_c=40., y_c=20.)
        torch_model = Aberrated(TorchLorenzMie)(
            coordinates=self.model.coordinates, **kwargs)
        numpy_model = Aberrated(LorenzMie)(
            coordinates=self.model.coordinates, **kwargs)
        for model in (torch_model, numpy_model):
            model.particle.properties = self.model.particle.properties
        np.testing.assert_allclose(torch_model.hologram(),
                                   numpy_model.hologram(), atol=ATOL)

    def test_single_particle_fast_path_matches_the_field_route(self) -> None:
        # on Apple GPUs a single particle's hologram is formed in the kernel
        for kwargs in (dict(), dict(cartesian=False), dict(bohren=False)):
            fast = self.model.hologram(**kwargs)
            field = self.model.field(**kwargs).cpu()
            slow = (field.real**2 + field.imag**2).sum(0) + 2*field[0].real + 1
            np.testing.assert_allclose(fast, slow.double().numpy(),
                                       atol=1e-5, err_msg=str(kwargs))

    def test_unsupported_device_raises(self) -> None:
        with self.assertRaises(ValueError):
            TorchLorenzMie(device='cpu')

    def test_get_device_rejects_unknown_preference(self) -> None:
        with self.assertRaises(ValueError):
            get_device('tpu')

    def test_device_preference_is_honored(self) -> None:
        self.assertEqual(get_device(_device).type, _device)
        self.assertEqual(TorchLorenzMie(device=_device).device.type, _device)
        for other in {'cuda', 'xpu', 'mps'} - {_device}:
            with self.assertRaises(RuntimeError):
                get_device(other)

    def test_repr(self) -> None:
        r = repr(self.model)
        self.assertIsInstance(r, str)


@unittest.skipUnless(_batch_available, 'torch/triton/CUDA not available')
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
