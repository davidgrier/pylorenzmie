'''The model prediction that lmtool shows while the user changes parameters.'''

import numpy as np
from pyqtgraph.Qt.QtCore import QObject, QTimer, pyqtSignal

from pylorenzmie.lmtool.LMSession import LMSession
from pylorenzmie.lib.lmtypes import Coordinates


class Preview(QObject):
    '''Computes what the model predicts for the parameters of a session.

    The prediction is the hologram of the region of interest, for the Fit
    tab, and the radial profile, for the Profile tab.  A preview has models
    of its own and reads the parameters from the session, so it never touches
    the model that a fit is changing in another thread.

    A change of a parameter, of the region or of the radius asks for a new
    prediction with :meth:`request`; the prediction is computed once, when
    control returns to the event loop, however many requests came in before
    that.  A slider that changes six parameters at once, or a fit that
    changes five, computes one prediction.

    Parameters
    ----------
    session : LMSession
        Source of the parameters.  The preview requests a prediction
        whenever the session announces a change.
    '''

    #: Emitted with the model hologram of the region (a flat array, or None
    #: if there is no region yet), the model's radial profile, and the noise.
    updated = pyqtSignal(object, object, float)

    def __init__(self, session: LMSession) -> None:
        super().__init__()
        self.session = session
        self._model = session.cls()
        self._profile = session.cls()
        self._profile.x_p = 0.
        self._profile.y_p = 0.
        self._coordinates = None
        self._radius = 100
        #: Number of predictions computed (for the tests, and curiosity).
        self.count = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.refresh)
        session.parametersChanged.connect(self.request)

    def setCoordinates(self, coordinates: Coordinates) -> None:
        '''Set the pixels of the region of interest, shape (2, npts).'''
        self._coordinates = np.reshape(coordinates, (2, -1))
        self.request()

    def setRadius(self, radius: int) -> None:
        '''Set the number of points of the radial profile, in pixels.'''
        self._radius = int(radius)
        self.request()

    def request(self, *args) -> None:
        '''Ask for a prediction, to be computed once the event loop runs.'''
        if not self._timer.isActive():
            self._timer.start()

    def refresh(self) -> None:
        '''Compute the prediction now and emit :attr:`updated`.'''
        self._timer.stop()
        properties = self.session.properties
        hologram = None
        if self._coordinates is not None:
            self._model.properties = properties
            self._model.coordinates = self._coordinates
            hologram = self._model.hologram()
        # the profile is of a particle at the origin
        self._profile.properties = {name: value
                                    for name, value in properties.items()
                                    if name not in ('x_p', 'y_p')}
        self._profile.coordinates = np.arange(self._radius)
        self.count += 1
        self.updated.emit(hologram, self._profile.hologram(),
                          float(properties['noise']))
