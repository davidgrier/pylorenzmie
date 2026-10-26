'''Characterization results collected over a series of frames.

A :class:`Trajectory` stores the table that
:class:`~pylorenzmie.analysis.Frame.Frame` returns for each frame of a video
and combines them in one :class:`pandas.DataFrame` (``data``) that can be
written to a file.  It records the measurements and nothing else: tagging each
row with its frame number, and linking the measurements of one particle in
successive frames into a track, for example with :func:`trackpy.link`, is left
to the caller, as the video tutorial shows.
'''

from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Properties, Results
import pandas as pd


class Trajectory(LMObject):
    '''Accumulate per-frame characterization results over time.

    Each call to :meth:`append` stores one frame's worth of
    :class:`~pylorenzmie.analysis.Frame` output.  :attr:`data`
    returns the combined time series as a single
    :class:`pandas.DataFrame`.

    Inherits from :class:`pylorenzmie.lib.LMObject`.
    '''

    def __init__(self) -> None:
        super().__init__()
        self._frames: list[pd.DataFrame] = []

    @LMObject.properties.getter
    def properties(self) -> Properties:
        return dict()

    @property
    def data(self) -> pd.DataFrame:
        '''All results concatenated into a single DataFrame.'''
        if not self._frames:
            return pd.DataFrame()
        return pd.concat(self._frames, ignore_index=True)

    def append(self, results: Results) -> None:
        '''Append one frame of results.

        Parameters
        ----------
        results : pandas.DataFrame or pandas.Series
            Output from :meth:`~pylorenzmie.analysis.Frame.Frame.optimize` or
            :meth:`~pylorenzmie.analysis.Frame.Frame.analyze`.
        '''
        if isinstance(results, pd.Series):
            results = results.to_frame().T
        self._frames.append(results)

    def clear(self) -> None:
        '''Remove all stored results.'''
        self._frames = []

    def to_csv(self, path: str, **kwargs) -> None:
        '''Write accumulated results to a CSV file.

        Parameters
        ----------
        path : str
            Destination file path.
        **kwargs
            Additional keyword arguments passed to
            :meth:`pandas.DataFrame.to_csv`.
        '''
        self.data.to_csv(path, **kwargs)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        import cv2
        from pathlib import Path
        from pylorenzmie.analysis import Frame
        from pylorenzmie.utilities import example_hologram

        frame = Frame()
        frame.instrument.wavelength = 0.447
        frame.instrument.magnification = 0.048
        frame.instrument.n_m = 1.34

        trajectory = cls()
        for _ in range(3):
            results = frame.analyze(example_hologram('image0010.png').data)
            trajectory.append(results)

        print(trajectory.data)


if __name__ == '__main__':  # pragma: no cover
    Trajectory.example()
