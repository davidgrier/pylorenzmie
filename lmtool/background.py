'''The background that lmtool divides its holograms by.

The window keeps what the normalization needs: the :class:`Normalizer`, the
dark count and the reason why the last background that was asked for was not
used.  None of it needs Qt, so it is here and not in the window.
'''

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pylorenzmie.lib.lmtypes import Image
from pylorenzmie.utilities import Normalizer

logger = logging.getLogger(__name__)


class Background:
    '''The normalization of the holograms that a window shows.

    Parameters
    ----------
    dark : float, optional
        Dark count (black level) of the camera, in gray levels, subtracted
        from the hologram and the background before the division (see
        :class:`~pylorenzmie.utilities.normalizer.Normalizer`).  Default: 0.

    Attributes
    ----------
    normalizer : Normalizer
        What the holograms are divided by.  The median of each hologram
        until a background is read.
    dark : float
        The dark count that the normalizer subtracts.
    refused : str
        Why the last background that was asked for was not used, or an empty
        string.

    Raises
    ------
    ValueError
        If the dark count is not finite and non-negative.
    '''

    def __init__(self, dark: float = 0.) -> None:
        self.dark = float(dark)
        self.normalizer = Normalizer(dark=self.dark)
        self.refused = ''

    def use(self, background: str | float) -> str | None:
        '''Divide by a background from now on.

        Parameters
        ----------
        background : str or float
            A number, or the name of an image file, as for
            :meth:`~pylorenzmie.utilities.normalizer.Normalizer.read`.

        Returns
        -------
        str or None
            Why the background could not be used, which keeps the previous
            one; None if it could.
        '''
        try:
            self.normalizer = Normalizer.read(background, dark=self.dark)
        except (OSError, ValueError) as error:
            logger.error(f'Could not set the background: {error}')
            self.refused = str(error)
            return self.refused
        self.refused = ''
        return None

    def set_dark(self, dark: float) -> None:
        '''Subtract a dark count before normalizing.

        Parameters
        ----------
        dark : float
            Dark count in gray levels, finite and not negative.

        Raises
        ------
        ValueError
            If the dark count is not finite and non-negative.  The previous
            one stays.
        '''
        self.normalizer.dark = dark        # validates it
        self.dark = self.normalizer.dark

    def normalize(self, raw: Image) -> tuple[NDArray[np.floating], str]:
        '''Normalize a hologram.

        A background that does not fit the hologram (an image of another
        shape, for example) is not used for it: the hologram is divided by
        its median instead.

        Parameters
        ----------
        raw : numpy.ndarray
            The hologram as the camera made it.

        Returns
        -------
        data : numpy.ndarray
            The normalized hologram.
        description : str
            What it was divided by, and why not by what was asked for
            (see :meth:`describe`).
        '''
        try:
            return self.normalizer(raw), self.describe()
        except ValueError as error:
            logger.error(f'Could not normalize by the background: {error}')
            median = Normalizer(dark=self.dark)
            return (median(raw),
                    self.describe(median, f'{error}; using the median'))

    def describe(self,
                 normalizer: Normalizer | None = None,
                 fallback: str = '') -> str:
        '''What the holograms are divided by, and what is wrong.

        Parameters
        ----------
        normalizer : Normalizer, optional
            The normalizer that was used.  Default: the current one.
        fallback : str, optional
            Why a hologram is divided by the median instead.  Default: the
            reason why the last background was refused, if there is one.

        Returns
        -------
        str
            A line for the status bar.
        '''
        normalizer = normalizer or self.normalizer
        if normalizer.method != 'reference':
            used = normalizer.method
        elif normalizer.source:
            used = Path(normalizer.source).name
        elif np.ndim(normalizer.reference) == 0:
            used = f'{normalizer.reference:g}'
        else:
            used = 'image'
        problem = fallback or self.refused
        dark = f', dark {normalizer.dark:g}' if normalizer.dark else ''
        return (f'background: {used}{dark}'
                + (f' ({problem})' if problem else ''))

    def start_directory(self, datafile: str | None) -> str:
        '''Where a dialog for a background starts: at the last background
        or, failing that, in the folder of the hologram, which is where a
        background is likely to be.

        Parameters
        ----------
        datafile : str or None
            The file of the hologram that is shown.

        Returns
        -------
        str
            A file or a folder, or an empty string.
        '''
        if self.normalizer.source:
            return self.normalizer.source
        return str(Path(datafile).parent) if datafile else ''
