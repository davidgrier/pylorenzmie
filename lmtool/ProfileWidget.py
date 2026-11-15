from typing import Any
import numpy as np
from numpy.typing import NDArray
from qtpy.QtCore import Qt
import pyqtgraph as pg

from pylorenzmie.lmtool.required import required
from pylorenzmie.theory import LorenzMie


class ProfileWidget(pg.PlotWidget):
    '''Radial-profile plot showing experimental data and model prediction.

    The widget computes the prediction with its own model, which can be
    replaced.  Without a model (``widget.model = None``) it shows the
    prediction that it is given with :meth:`showTheory`.
    '''

    def __init__(self,
                 *args: Any,
                 model: LorenzMie | None = None,
                 radius: int = 100,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._configurePlot()
        self.model = LorenzMie() if model is None else model
        self._data: NDArray[np.floating] | None = None
        self._stdev: NDArray[np.floating] | None = None
        self.radius = radius

    def _configurePlot(self) -> None:
        self.setBackground('w')
        self.showGrid(True, True, 0.2)
        opts = {'font-size': '14pt', 'color': 'gray'}
        self.setLabel('bottom', 'r [pixels]', **opts)
        self.setLabel('left', 'b(r)', **opts)
        pen = pg.mkPen('k', width=3, style=Qt.PenStyle.DashLine)
        self.addLine(y=1, pen=pen)
        pen = pg.mkPen('k', width=3)
        self.getAxis('bottom').setPen(pen)
        self.getAxis('left').setPen(pen)
        self.theory = pg.PlotCurveItem(pen=pg.mkPen('r', width=3))
        self.experiment = pg.PlotCurveItem(pen=pg.mkPen('k', width=3))
        pen = pg.mkPen('k', width=1, style=Qt.PenStyle.DashLine)
        self.upperLimit = pg.PlotCurveItem(pen=pen)
        self.lowerLimit = pg.PlotCurveItem(pen=pen)
        brush = pg.mkBrush(255, 165, 0, 128)
        self.region = pg.FillBetweenItem(self.upperLimit,
                                         self.lowerLimit,
                                         brush)
        self.addItem(self.theory)
        self.addItem(self.experiment)
        self.addItem(self.upperLimit)
        self.addItem(self.lowerLimit)
        self.addItem(self.region)

    @property
    def properties(self) -> dict[str, LorenzMie.Property]:
        '''Properties of the model that computes the theory curve.

        Setting them leaves out the position, because the profile is
        always centered.'''
        return required(self.model).properties

    @properties.setter
    def properties(self, properties: dict[str, LorenzMie.Property]) -> None:
        # Profile is always centred at the origin, so x_p/y_p are irrelevant.
        properties = {k: v for k, v in properties.items()
                      if k not in ('x_p', 'y_p')}
        if self._model is None or len(properties) == 0:
            return
        self._model.properties = properties
        self.plotTheory()

    @property
    def model(self) -> LorenzMie | None:
        '''The model that computes the theory curve, or None if the curve is
        given with :meth:`showTheory`.  A model is moved to the origin.'''
        return self._model

    @model.setter
    def model(self, model: LorenzMie | None) -> None:
        # Profile is always computed at the origin; zero the position in-place.
        if model is not None:
            particle = model.single()
            particle.x_p = 0.0
            particle.y_p = 0.0
        self._model = model

    @property
    def data(self) -> tuple[NDArray[np.floating] | None,
                            NDArray[np.floating] | None]:
        '''The azimuthal average of the hologram and its standard deviation.

        Both are None until they are set.'''
        return (self._data, self._stdev)

    @data.setter
    def data(self,
             data: tuple[NDArray[np.floating], NDArray[np.floating]]) -> None:
        self._data, self._stdev = data
        self.plotData()

    @property
    def radius(self) -> int:
        '''Number of pixels of the profile that are shown.'''
        return self._radius

    @radius.setter
    def radius(self, radius: int) -> None:
        self._radius = radius
        if self._model is not None:
            self._model.coordinates = np.arange(radius, dtype=float)
        self.plotData()
        self.plotTheory()

    def plotTheory(self) -> None:
        '''Show the hologram of the model as the theory curve.'''
        if self._model is not None:
            self.theory.setData(self._model.hologram())

    def showTheory(self, profile: np.ndarray) -> None:
        '''Show a model prediction that was computed elsewhere.'''
        self.theory.setData(profile)

    def plotData(self) -> None:
        '''Show the azimuthal profile of the data, with its standard
        deviation.'''
        if self._data is None or self._stdev is None:
            return
        radius = min(self.radius, len(self._data))
        data = self._data[0:radius]
        stdev = self._stdev[0:radius]
        self.experiment.setData(data)
        self.upperLimit.setData(data + stdev)
        self.lowerLimit.setData(data - stdev)

    @classmethod
    def example(cls) -> None:
        '''Run a demonstration of the widget, for use as a script.'''
        from pylorenzmie.theory import AberratedLorenzMie as model

        app = pg.mkQApp()
        widget = cls()
        widget.model = model()
        widget.radius = 150
        widget.show()
        app.exec()


if __name__ == '__main__':  # pragma: no cover
    ProfileWidget.example()
