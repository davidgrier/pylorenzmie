'''The scripts in tools/ make files that the package ships.

They are not part of the package, so they are loaded from their files.  The
tests check that each script can be imported without doing its work, that
the pieces work on small inputs, and that what a script made last time
still agrees with what it makes now.
'''
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / 'tools'


def load(name: str):
    '''Import a script of tools/ by its file name.'''
    spec = importlib.util.spec_from_file_location(
        f'tools_{name}', TOOLS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path: Path) -> np.ndarray:
    '''Read an image as RGB.'''
    return cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1]


class TestMakeSplash(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tool = load('make_splash')

    def test_importing_does_not_write_the_image(self):
        before = self.tool.PATH.stat().st_mtime_ns
        load('make_splash')
        self.assertEqual(self.tool.PATH.stat().st_mtime_ns, before)

    def test_the_image_has_the_size_of_the_splash_screen(self):
        image = self.tool.render()
        self.assertEqual(image.shape,
                         (self.tool.HEIGHT, self.tool.WIDTH, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_the_shipped_image_is_what_the_script_makes(self):
        shipped = read(self.tool.PATH)
        made = self.tool.render()
        self.assertEqual(shipped.shape, made.shape)
        difference = np.abs(shipped.astype(int) - made.astype(int))
        # a pixel may differ by a level between platforms, not by more
        self.assertLessEqual(
            difference.max(),
            2,
            'run python tools/make_splash.py',
        )

    def test_main_saves_the_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'splash.png'
            with patch.object(self.tool, 'PATH', path), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                self.tool.main()
            saved = read(path)
        self.assertEqual(saved.shape, (self.tool.HEIGHT, self.tool.WIDTH, 3))
        self.assertIn('bytes', out.getvalue())


class TestTrainMLPEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tool = load('train_mlp_estimator')

    def test_the_profile_length_is_that_of_the_estimator(self):
        from pylorenzmie.analysis.MLPEstimator import MLPEstimator
        self.assertEqual(self.tool.N_FEATURES, MLPEstimator.n_features)

    def test_the_output_is_the_weights_that_the_estimator_loads(self):
        module = importlib.import_module('pylorenzmie.analysis.MLPEstimator')
        self.assertEqual(self.tool.OUTPUT.resolve(),
                         module._default_weights().resolve())

    def test_the_model_is_evaluated_along_a_radius(self):
        model = self.tool._make_model()
        radii = np.arange(self.tool.N_FEATURES)
        np.testing.assert_array_equal(model.coordinates[0], radii)
        np.testing.assert_array_equal(model.coordinates[1], 0.)
        self.assertEqual(model.instrument.wavelength,
                         self.tool.INSTRUMENT['wavelength'])

    def test_generated_data_has_the_shape_of_the_training_set(self):
        model = self.tool._make_model()
        rng = np.random.default_rng(0)
        with patch.object(self.tool, 'N_TRAIN', 20), \
                contextlib.redirect_stdout(io.StringIO()):
            X, y = self.tool.generate_data(model, rng)
        self.assertEqual(X.shape[1], self.tool.N_FEATURES)
        self.assertEqual(y.shape[1], 3)
        self.assertEqual(len(X), len(y))
        self.assertTrue(np.isfinite(X).all())
        low, high = self.tool.BOUNDS['z_p']
        self.assertTrue(((y[:, 0] >= low) & (y[:, 0] <= high)).all())

    def test_the_pipeline_fits_and_predicts_in_the_original_scale(self):
        model = self.tool._make_model()
        rng = np.random.default_rng(1)
        with patch.object(self.tool, 'N_TRAIN', 40), \
                contextlib.redirect_stdout(io.StringIO()):
            X, y = self.tool.generate_data(model, rng)
        with patch.object(self.tool, 'HIDDEN', (8,)):
            pipe = self.tool.build_pipeline()
        pipe.set_params(regressor__regressor__max_iter=20)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            pipe.fit(X, y)
            self.tool.evaluate(pipe, X, y)
        predicted = pipe.predict(X)
        self.assertEqual(predicted.shape, y.shape)
        self.assertTrue((predicted[:, :2] > 0).all())    # z_p and a_p
        self.assertIn('MAE', out.getvalue())

    def test_the_cache_is_used_unless_a_new_set_is_asked_for(self):
        data = dict(X=np.zeros((3, 4)), y=np.ones((3, 3)))
        with patch.object(self.tool, '_CACHE') as cache, \
                patch.object(self.tool.np, 'load', return_value=data), \
                contextlib.redirect_stdout(io.StringIO()):
            cache.exists.return_value = True
            X, y = self.tool.load_or_generate()
        self.assertEqual(X.shape, (3, 4))
        self.assertEqual(y.shape, (3, 3))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
