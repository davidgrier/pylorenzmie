import pylorenzmie.utilities.configuration as config

if config.has_catch():
    from .catchLocalizer import catchLocalizer as Localizer
    from .catchEstimator import catchEstimator as Estimator
else:
    from .Localizer import Localizer
    from .Estimator import Estimator


__all__ = [Localizer, Estimator]
