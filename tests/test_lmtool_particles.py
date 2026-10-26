import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from pyqtgraph import mkQApp
    from pyqtgraph.Qt.QtCore import Qt
    from pylorenzmie.lmtool.ParticleTable import ParticleTable, radius_of
    from pylorenzmie.lmtool.results import save_results
    from pylorenzmie.lmtool.tasks import (Cancelled, LocalizeRequest,
                                          run_localize)
    from pylorenzmie.theory import LorenzMie
    _qt = True
except Exception:       # no Qt binding, or no usable platform plugin
    _qt = False


def setUpModule():
    if _qt:
        mkQApp()


def found(n=3):
    return pd.DataFrame(
        [dict(x_p=100. * (k + 1), y_p=50. * (k + 1),
              bbox=((50 * k, 20), 120 + 20 * k, 100)) for k in range(n)])


def result(x_p=10., y_p=20., z_p=300.):
    return pd.Series(dict(x_p=x_p, dx_p=.1, y_p=y_p, dy_p=.1, z_p=z_p,
                          dz_p=1., a_p=1., da_p=.01, n_p=1.45, dn_p=.001,
                          success=True, npix=500, redchi=1.5))


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestParticleTable(unittest.TestCase):

    def setUp(self):
        self.table = ParticleTable()
        self.currents = []
        self.table.currentChanged.connect(self.currents.append)

    def test_the_radius_follows_the_bounding_box_within_limits(self):
        self.assertEqual(radius_of(((0, 0), 200, 150)), 100)
        self.assertEqual(radius_of(((0, 0), 20, 10)), 30)
        self.assertEqual(radius_of(((0, 0), 900, 100)), 250)

    def test_found_particles_fill_the_table(self):
        self.table.setFound(found())
        self.assertEqual(self.table.count, 3)
        self.assertEqual(self.table.rowCount(), 3)
        self.assertEqual(self.table.entry(1)['x_p'], 200.)
        self.assertEqual(self.table.entry(1)['radius'], 70)
        self.assertEqual(self.table.entry(0)['source'], 'found')
        self.assertTrue(all(self.table.entry(r)['include'] for r in range(3)))

    def test_a_new_search_keeps_the_particles_that_were_added(self):
        self.table.add(5., 6., 40)
        self.table.setFound(found(2))
        self.assertEqual([self.table.entry(r)['source'] for r in range(3)],
                         ['found', 'found', 'added'])

    def test_clear_forgets_everything(self):
        self.table.setFound(found())
        self.table.setCurrent(1)
        self.table.clear()
        self.assertEqual(self.table.count, 0)
        self.assertEqual(self.table.current, -1)
        self.assertEqual(self.currents[-1], -1)

    def test_current_is_announced_once(self):
        self.table.setFound(found())
        self.currents.clear()
        self.table.setCurrent(2)
        self.table.setCurrent(2)
        self.table.setCurrent(7)                        # no such row
        self.assertEqual(self.currents, [2, -1])

    def test_next_wraps_and_skips_the_unchecked(self):
        self.table.setFound(found())
        self.table.setData(self.table.index(1, 0), Qt.CheckState.Unchecked,
                           Qt.ItemDataRole.CheckStateRole)
        order = []
        for _ in range(4):
            row = self.table.next()
            self.table.setCurrent(row)
            order.append(row)
        self.assertEqual(order, [0, 2, 0, 2])

    def test_next_with_nothing_checked_is_none(self):
        self.table.setFound(found(2))
        for row in range(2):
            self.table.setData(self.table.index(row, 0),
                               Qt.CheckState.Unchecked,
                               Qt.ItemDataRole.CheckStateRole)
        self.assertIsNone(self.table.next())
        self.assertIsNone(ParticleTable().next())

    def test_remove(self):
        self.table.setFound(found())
        self.table.setCurrent(2)
        self.table.remove(0)
        self.assertEqual(self.table.count, 2)
        self.assertEqual(self.table.current, 1)       # the same particle
        self.table.remove(1)
        self.assertEqual(self.table.current, -1)
        self.table.remove(9)                          # no such row
        self.assertEqual(self.table.count, 1)

    def test_row_at_finds_the_nearest_within_the_tolerance(self):
        self.table.setFound(found())
        self.assertEqual(self.table.rowAt(203., 98.), 1)
        self.assertIsNone(self.table.rowAt(150., 75., tolerance=5.))

    def test_an_estimate_and_a_fit_are_recorded(self):
        self.table.setFound(found())
        self.table.setEstimate(1, dict(z_p=250., a_p=1., n_p=1.4))
        self.assertEqual(self.table.entry(1)['status'], 'estimated')
        self.table.setResult(1, dict(z_p=300., a_p=1.1, n_p=1.45),
                             result(x_p=201., y_p=99.))
        self.assertEqual(self.table.entry(1)['status'], 'fitted')
        self.table.setEstimate(1, dict(z_p=1., a_p=1., n_p=1.4))
        self.assertEqual(self.table.entry(1)['status'], 'fitted')
        # a fitted particle is where the fit put it
        self.assertEqual(self.table.position(1), (201., 99.))
        self.assertEqual(self.table.rowAt(201., 99.), 1)

    def test_the_view_data(self):
        self.table.setFound(found())
        self.table.setResult(0, dict(z_p=300., a_p=1.1, n_p=1.45), result())
        text = lambda r, c: self.table.data(self.table.index(r, c))  # noqa
        self.assertEqual(text(0, 1), '10.0')                # x_p of the fit
        self.assertEqual(text(0, 4), 'fitted')
        self.assertEqual(text(0, 5), '300.0')
        self.assertEqual(text(0, 7), '1.450')
        self.assertEqual(text(0, 8), '1.50')
        self.assertEqual(text(1, 5), '')                    # no fit yet
        self.assertEqual(self.table.data(
            self.table.index(0, 0), Qt.ItemDataRole.CheckStateRole),
            Qt.CheckState.Checked)
        self.assertEqual(self.table.headerData(
            8, Qt.Orientation.Horizontal), 'χ²')

    def test_the_results_are_the_fitted_checked_particles(self):
        self.table.setFound(found())
        self.table.setResult(0, dict(z_p=1.), result(z_p=300.))
        self.table.setResult(2, dict(z_p=1.), result(z_p=400.))
        self.table.setData(self.table.index(2, 0), Qt.CheckState.Unchecked,
                           Qt.ItemDataRole.CheckStateRole)
        frame = self.table.results()
        self.assertEqual(list(frame.index), [0])
        self.assertEqual(frame.index.name, 'particle')
        self.assertEqual(frame.loc[0, 'z_p'], 300.)
        self.assertEqual(frame.loc[0, 'source'], 'found')
        self.assertEqual(list(self.table.results(included=False).index),
                         [0, 2])
        self.assertEqual(len(ParticleTable().results()), 0)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestSaveResults(unittest.TestCase):

    def setUp(self):
        self.results = pd.DataFrame([result(z_p=300.), result(z_p=400.)])
        self.results.index.name = 'particle'
        self.metadata = pd.Series(dict(wavelength=0.447, datafile='x.png'))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def path(self, name):
        return str(Path(self.directory.name) / name)

    def test_json_has_the_metadata_and_a_record_for_each_particle(self):
        filename = save_results(self.results, self.metadata,
                                self.path('r.json'))
        document = json.loads(Path(filename).read_text())
        self.assertEqual(document['metadata']['datafile'], 'x.png')
        self.assertEqual([r['z_p'] for r in document['results']],
                         [300., 400.])

    def test_csv(self):
        filename = save_results(self.results, self.metadata,
                                self.path('r.csv'))
        frame = pd.read_csv(filename, index_col='particle')
        self.assertEqual(list(frame['z_p']), [300., 400.])

    def test_hdf5(self):
        try:
            import tables  # noqa: F401
        except ImportError:
            self.skipTest('PyTables is not installed')
        filename = save_results(self.results, self.metadata,
                                self.path('r.h5'))
        self.assertTrue(filename.endswith('.h5'))
        self.assertEqual(list(pd.read_hdf(filename, key='results')['z_p']),
                         [300., 400.])
        self.assertEqual(pd.read_hdf(filename, key='metadata')['datafile'],
                         'x.png')

    def test_without_pytables_hdf5_becomes_json(self):
        from unittest import mock
        with mock.patch('pylorenzmie.lmtool.results._save_hdf',
                        side_effect=ImportError):
            filename = save_results(self.results, self.metadata,
                                    self.path('r.h5'))
        self.assertTrue(filename.endswith('r.json'))
        self.assertTrue(Path(filename).exists())

    def test_the_directory_is_created(self):
        filename = save_results(self.results, self.metadata,
                                self.path('new/deeper/r.json'))
        self.assertTrue(Path(filename).exists())


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestLocalizeTask(unittest.TestCase):

    def image(self):
        model = LorenzMie(coordinates=LorenzMie.meshgrid((200, 200)))
        model.particle.r_p = [100., 100., 300.]
        model.particle.a_p, model.particle.n_p = 1.0, 1.45
        noise = np.random.default_rng(0).normal(0., 0.02, (200, 200))
        return model.hologram().reshape(200, 200) + noise

    def test_it_finds_the_particle(self):
        found = run_localize(LocalizeRequest(self.image()),
                             threading.Event())
        self.assertGreaterEqual(len(found), 1)
        self.assertAlmostEqual(found.iloc[0]['x_p'], 100., delta=3.)
        self.assertAlmostEqual(found.iloc[0]['y_p'], 100., delta=3.)

    def test_it_stops_when_it_is_cancelled(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            run_localize(LocalizeRequest(self.image()), cancel)

    def test_it_stops_in_the_middle(self):
        noisy = 1. + np.random.default_rng(1).normal(0., 0.03, (300, 300))
        cancel = threading.Event()
        from pylorenzmie.analysis import Localizer
        original = Localizer.localize
        calls = []

        def counting(self, image, *args, check=None, **kwargs):
            def counted():
                calls.append(1)
                if len(calls) == 3:
                    cancel.set()
                check()
            return original(self, image, *args, check=counted, **kwargs)

        from unittest import mock
        with mock.patch.object(Localizer, 'localize', counting):
            with self.assertRaises(Cancelled):
                run_localize(LocalizeRequest(noisy), cancel)
        self.assertEqual(len(calls), 3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
