'''Tests of the three-panel display of a region, its fit and its residuals.'''
import os
import unittest

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    import qtpy     # chooses the Qt binding (QT_API) before pyqtgraph can
    from pyqtgraph import mkQApp
    from qtpy.QtCore import QRectF
    from pylorenzmie.lmtool.FitWidget import FitWidget
    _qt = True
except Exception:       # no Qt binding, or no usable platform plugin
    _qt = False


def setUpModule():
    if _qt:
        mkQApp()


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestFitWidget(unittest.TestCase):

    def setUp(self):
        self.widget = FitWidget()
        self.addCleanup(self.widget.close)
        self.data = np.random.default_rng(0).normal(1., 0.05, (20, 30))
        self.rect = QRectF(5, 7, 30, 20)

    def test_the_model_is_shown_when_the_widget_is(self):
        self.widget.setData(self.data, self.rect, np.zeros((2, 600)))
        self.widget.showModel(np.ones(self.data.size), noise=0.1)
        self.assertIsNone(self.widget.fit.image)    # not visible yet
        self.widget.show()
        self.assertIsNotNone(self.widget.fit.image)
        np.testing.assert_allclose(self.widget.residuals.image,
                                   (self.data - 1.) / 0.1)

    def test_the_model_is_shown_at_once_when_the_widget_is_visible(self):
        self.widget.show()
        self.widget.setData(self.data, self.rect, np.zeros((2, 600)))
        self.widget.showModel(np.ones(self.data.size))
        np.testing.assert_allclose(self.widget.fit.image, 1.)

    def test_no_model_clears_the_prediction(self):
        self.widget.show()
        self.widget.setData(self.data, self.rect, np.zeros((2, 600)))
        self.widget.showModel(None)
        self.assertIsNone(self.widget.fit.image)

    def test_a_prediction_for_another_region_is_ignored(self):
        self.widget.show()
        self.widget.setData(self.data, self.rect, np.zeros((2, 600)))
        self.widget.showModel(np.ones(10))
        self.assertIsNone(self.widget.fit.image)

    def test_a_setting_goes_to_the_optimizer(self):
        self.widget.setSetting('fraction', 0.5)
        self.assertEqual(self.widget.optimizer.fraction, 0.5)
        self.widget.setSetting('xtol', 1e-3)
        self.assertEqual(self.widget.optimizer.settings['xtol'], 1e-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
