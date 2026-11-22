'''Abstract base class for hologram parameter estimators.'''

from abc import abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pylorenzmie.lib import LMObject
from pylorenzmie.analysis.Hologram import Hologram


class BaseEstimator(LMObject):
    '''Abstract base for particle parameter estimators.

    Subclasses must implement :meth:`estimate` and the
    :attr:`~pylorenzmie.lib.LMObject.LMObject.properties` getter inherited
    from :class:`~pylorenzmie.lib.LMObject`.

    See Also
    --------
    ~pylorenzmie.analysis.Estimator.Estimator : conventional azimuthal-profile
        estimator.
    ~pylorenzmie.analysis.DEEstimator.DEEstimator : global
        differential-evolution estimator.

    Notes
    -----
    :attr:`exclude` is set by :class:`~pylorenzmie.analysis.Feature.Feature`
    to the pixels that its mask leaves out.  An estimator that has no use
    for it ignores it.
    '''

    exclude: NDArray[np.bool_] | None = None
    '''Pixels that the estimate must not use, or ``None``.'''

    @abstractmethod
    def estimate(self, hologram: Hologram) -> pd.Series:
        '''Estimate particle parameters from a normalized hologram.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram crop to analyze.

        Returns
        -------
        result : pandas.Series
            Estimated particle properties.
        '''
