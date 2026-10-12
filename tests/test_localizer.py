import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from pylorenzmie.analysis.Localizer import Localizer


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'image0010.png'


class TestLocalizer(unittest.TestCase):

    def setUp(self):
        import cv2
        self.localizer = Localizer()
        data = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_GRAYSCALE)
        self.data = data.astype(float) / 100.

    def test_properties(self):
        props = self.localizer.properties
        self.assertIn('diameter', props)
        self.assertIn('nfringes', props)
        self.assertEqual(props['diameter'], 31)
        self.assertEqual(props['nfringes'], 4)

    def test_json_roundtrip(self):
        self.localizer.diameter = 41
        s = self.localizer.to_json()
        self.localizer.diameter = 31
        self.localizer.from_json(s)
        self.assertEqual(self.localizer.diameter, 41)

    def test_detect_alias(self):
        '''detect() is a backward-compatibility alias for localize()'''
        self.assertEqual(self.localizer.detect, self.localizer.localize)

    def test_localize_returns_dataframe(self):
        result = self.localizer.localize(self.data)
        self.assertIsInstance(result, pd.DataFrame)

    def test_localize_columns(self):
        result = self.localizer.localize(self.data)
        self.assertIn('x_p', result.columns)
        self.assertIn('y_p', result.columns)
        self.assertIn('bbox', result.columns)

    def test_localize_finds_features(self):
        result = self.localizer.localize(self.data)
        self.assertGreater(len(result), 0)

    def test_localize_list(self):
        result = self.localizer.localize([self.data, self.data])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_crop(self):
        result = self.localizer.localize(self.data)
        if len(result) > 0:
            bbox = result.iloc[0]['bbox']
            cropped = self.localizer.crop(self.data, bbox)
            self.assertIsInstance(cropped, np.ndarray)
            self.assertEqual(cropped.ndim, 2)

    def test_crop_shape(self):
        bbox = ((10, 20), 50, 60)
        cropped = self.localizer.crop(self.data, bbox)
        self.assertEqual(cropped.shape, (60, 50))

    def test_custom_diameter(self):
        result = self.localizer.localize(self.data, diameter=41)
        self.assertIsInstance(result, pd.DataFrame)

    def test_custom_nfringes(self):
        result = self.localizer.localize(self.data, nfringes=15)
        self.assertIsInstance(result, pd.DataFrame)


class TestLocalizerBoxes(unittest.TestCase):
    '''Bounding boxes on synthetic holograms of known geometry.'''

    @staticmethod
    def hologram(shape, x_p, y_p, z_p=200., a_p=0.75):
        from pylorenzmie.theory import Sphere
        from pylorenzmie.theory.LorenzMie import LorenzMie
        ny, nx = shape
        x, y = np.meshgrid(np.arange(nx, dtype=float),
                           np.arange(ny, dtype=float))
        model = LorenzMie(coordinates=np.stack(
            [x.ravel(), y.ravel(), np.zeros(x.size)]))
        model.particle = Sphere(a_p=a_p, n_p=1.45, x_p=x_p, y_p=y_p,
                                z_p=z_p)
        return model.hologram().reshape(shape)

    def test_extent_counts_two_crossings_per_fringe(self):
        # profile 1 + cos: crossings of 1 at radii 5, 15, 25, ...
        r = np.arange(100)
        profile = 1. + np.sin(np.pi * (r - 5.) / 10. + 1e-3)
        crossings = np.where(np.diff(np.sign(profile - 1.)) != 0)[0] + 1
        self.assertGreater(crossings.size, 6)
        for n in (1, 2, 3):
            self.assertEqual(Localizer._extent(profile, n),
                             float(crossings[2 * n - 1]))

    def test_extent_with_too_few_crossings_uses_the_last(self):
        profile = 1. + np.sin(np.pi * (np.arange(30) - 5.) / 10. + 1e-3)
        crossings = np.where(np.diff(np.sign(profile - 1.)) != 0)[0] + 1
        self.assertEqual(Localizer._extent(profile, 50),
                         float(crossings[-1]))

    def test_extent_without_crossings_is_the_profile(self):
        self.assertEqual(Localizer._extent(np.full(40, 2.), 3), 39.)

    def test_box_encloses_the_requested_number_of_fringes(self):
        from pylorenzmie.lib import Azimuthal
        image = self.hologram((500, 500), 250., 250.)
        profile = Azimuthal.avg(image, (250., 250.))
        crossings = np.where(np.diff(np.sign(profile - 1.)) != 0)[0] + 1
        for nfringes in (2, 4, 6):
            found = Localizer().localize(image, nfringes=nfringes)
            self.assertEqual(len(found), 1)
            (x0, y0), w, h = found.bbox.iloc[0]
            half = (w - 1) / 2
            inside = int((crossings <= half).sum())
            # the box reaches the 2 * nfringes-th crossing, to a pixel
            self.assertGreaterEqual(inside, 2 * nfringes - 1)
            self.assertLessEqual(inside, 2 * nfringes + 1)
            self.assertEqual(w, h)
            self.assertAlmostEqual(x0 + half, found.x_p.iloc[0], delta=1.)

    def test_box_is_clipped_to_the_image(self):
        shape = (300, 300)
        image = self.hologram(shape, 15., 150.)
        found = Localizer().localize(image)
        self.assertGreaterEqual(len(found), 1)
        for (x0, y0), w, h in found.bbox:
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x0 + w, shape[1])
            self.assertLessEqual(y0 + h, shape[0])
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

    def test_crop_of_an_edge_feature_is_not_empty(self):
        localizer = Localizer()
        image = self.hologram((300, 300), 15., 150.)
        bbox = localizer.localize(image).bbox.iloc[0]
        cropped = localizer.crop(image, bbox)
        self.assertEqual(cropped.shape, (bbox[2], bbox[1]))
        self.assertGreater(cropped.size, 0)

    def test_bbox_entries_are_python_ints(self):
        image = self.hologram((300, 300), 150., 150.)
        (x0, y0), w, h = Localizer().localize(image).bbox.iloc[0]
        for value in (x0, y0, w, h):
            self.assertIsInstance(value, int)

    def test_featureless_image_gives_an_empty_frame_with_columns(self):
        found = Localizer().localize(np.ones((200, 200)))
        self.assertEqual(len(found), 0)
        self.assertEqual(list(found.columns), ['x_p', 'y_p', 'bbox', 'mass'])

    def test_localize_reports_the_strength(self):
        result = Localizer().localize(self.hologram((200, 200), 100., 100.))
        self.assertIn('mass', result.columns)
        self.assertEqual(result.attrs['nfound'], len(result))

    def test_maxfeatures_keeps_the_strongest_first(self):
        rng = np.random.default_rng(1)
        noisy = 1. + rng.normal(0., 0.03, (300, 300))
        localizer = Localizer()
        everything = localizer.localize(noisy)
        few = localizer.localize(noisy, maxfeatures=2)
        self.assertGreater(len(everything), 2)
        self.assertEqual(len(few), 2)
        self.assertEqual(few.attrs['nfound'], len(everything))
        self.assertEqual(list(few['mass']),
                         sorted(everything['mass'], reverse=True)[:2])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
