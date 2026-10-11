'''Abstract base class for hologram parameter estimators.'''

from abc import abstractmethod
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Result
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
    '''

    @abstractmethod
    def estimate(self, hologram: Hologram) -> Result:
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
