import unittest
import pandas as pd

from pylorenzmie.analysis.pair_grouping import group_overlapping


def _prediction(x_p, y_p, bbox):
    return dict(x_p=x_p, y_p=y_p, bbox=bbox)


class TestPairGrouping(unittest.TestCase):

    def test_empty(self):
        predictions = pd.DataFrame(columns=['x_p', 'y_p', 'bbox'])
        result = group_overlapping(predictions)
        self.assertEqual(len(result), 0)

    def test_no_overlap_passes_through(self):
        predictions = pd.DataFrame([
            _prediction(10., 10., ((0, 0), 20, 20)),
            _prediction(100., 100., ((90, 90), 20, 20)),
        ])
        result = group_overlapping(predictions)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0].x_p, 10.)
        self.assertEqual(result.iloc[0].bbox, ((0, 0), 20, 20))
        self.assertEqual(result.iloc[1].x_p, 100.)
        self.assertTrue((result.n_particles == 1).all())

    def test_pair_merges(self):
        predictions = pd.DataFrame([
            _prediction(10., 10., ((0, 0), 20, 20)),
            _prediction(25., 10., ((15, 0), 20, 20)),
        ])
        result = group_overlapping(predictions)
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row.x_p, [10., 25.])
        self.assertEqual(row.y_p, [10., 10.])
        self.assertEqual(row.bbox, ((0, 0), 35, 20))
        self.assertEqual(row.n_particles, 2)

    def test_chain_drops_middle_keeps_ends(self):
        # A overlaps B, B overlaps C, A does not overlap C.
        predictions = pd.DataFrame([
            _prediction(10., 10., ((0, 0), 20, 20)),
            _prediction(28., 10., ((18, 0), 20, 20)),
            _prediction(46., 10., ((36, 0), 20, 20)),
        ])
        result = group_overlapping(predictions)
        self.assertEqual(len(result), 2)
        self.assertEqual(sorted(result.x_p), [10., 46.])
        self.assertTrue((result.n_particles == 1).all())
        for bbox in result.bbox:
            self.assertEqual(bbox, ((0, 0), 20, 20) if bbox[0] == (0, 0)
                             else ((36, 0), 20, 20))

    def test_returns_dataframe_with_expected_columns(self):
        predictions = pd.DataFrame([
            _prediction(10., 10., ((0, 0), 20, 20)),
        ])
        result = group_overlapping(predictions)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns),
                             ['x_p', 'y_p', 'bbox', 'n_particles'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
