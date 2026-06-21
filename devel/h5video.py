'''HDF5 video reader for files created by QVideo.'''

import logging

import h5py
import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

Image = NDArray[np.uint8]


class h5video:
    '''Reader for HDF5 videos created by QVideo.

    Use as a context manager.  Frames are stored under the ``images/``
    group, keyed by timestamp strings.  Keys are sorted lexicographically
    on open.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Examples
    --------
    >>> with h5video('recording.h5') as vid:
    ...     for frame in vid:
    ...         process(frame)
    '''

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._file = None
        self._keys: list[str] = []
        self.index = 0

    def __enter__(self) -> 'h5video':
        self._file = h5py.File(self.filename, 'r')
        self._keys = sorted(self._file['images/'].keys())
        self.index = 0
        return self

    def __exit__(self, *args) -> None:
        self._file.close()
        self._file = None

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> 'h5video':
        self.index = 0
        return self

    def __next__(self) -> Image:
        if self.index >= len(self):
            raise StopIteration
        image = self._read(self.index)
        self.index += 1
        return image

    def _read(self, index: int) -> Image:
        return np.array(self._file['images/' + self._keys[index]])

    @property
    def nframes(self) -> int:
        '''Total number of frames in the video.'''
        return len(self)

    @property
    def shape(self) -> tuple:
        '''Shape of a single frame, or ``()`` if the file is empty.'''
        if not self._keys:
            return ()
        return self._read(0).shape

    def get_image(self) -> Image:
        '''Return the frame at the current index.

        Returns
        -------
        image : ndarray

        Raises
        ------
        IndexError
            If the current index is out of range.
        '''
        if self.index < 0 or self.index >= len(self):
            raise IndexError(
                f'Index {self.index} out of range ({len(self)})')
        return self._read(self.index)

    def get_time(self) -> str:
        '''Return the timestamp key for the current frame.'''
        return self._keys[self.index]

    def rewind(self) -> Image:
        '''Reset to the first frame and return it.'''
        self.index = 0
        return self.get_image()

    def next(self) -> Image | None:
        '''Advance to the next frame and return it, or None at end.'''
        if self.index >= len(self) - 1:
            return None
        self.index += 1
        return self.get_image()

    def goto(self, index: int) -> None:
        '''Set the current frame index.

        Parameters
        ----------
        index : int
        '''
        self.index = index
