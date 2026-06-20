import unittest
import pandas as pd

from pylorenzmie.analysis import Trajectory


class TestTrajectory(unittest.TestCase):

    def setUp(self):
        self.trajectory = Trajectory()

    def test_data_empty(self):
        self.assertIsInstance(self.trajectory.data, pd.DataFrame)
        self.assertEqual(len(self.trajectory.data), 0)

    def test_append_dataframe(self):
        df = pd.DataFrame({'a_p': [1.1], 'z_p': [100.]})
        self.trajectory.append(df)
        self.assertEqual(len(self.trajectory.data), 1)

    def test_append_series(self):
        s = pd.Series({'a_p': 1.1, 'z_p': 100.})
        self.trajectory.append(s)
        self.assertEqual(len(self.trajectory.data), 1)

    def test_append_multiple(self):
        df = pd.DataFrame({'a_p': [1.1, 1.2], 'z_p': [100., 110.]})
        self.trajectory.append(df)
        self.trajectory.append(df)
        self.assertEqual(len(self.trajectory.data), 4)

    def test_clear(self):
        df = pd.DataFrame({'a_p': [1.1]})
        self.trajectory.append(df)
        self.trajectory.clear()
        self.assertEqual(len(self.trajectory.data), 0)

    def test_to_csv(self):
        import tempfile, os
        df = pd.DataFrame({'a_p': [1.1], 'z_p': [100.]})
        self.trajectory.append(df)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            self.trajectory.to_csv(path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_properties(self):
        self.assertEqual(self.trajectory.properties, dict())

    def test_json_roundtrip(self):
        s = self.trajectory.to_json()
        t = Trajectory()
        t.from_json(s)
        self.assertEqual(t.properties, dict())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
