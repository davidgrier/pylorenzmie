import pyqtgraph as pg
from PyQt5.QtCore import (Qt, pyqtProperty, pyqtSlot)
from pylorenzmie.theory import LorenzMie
import numpy as np
from typing import (Optional, Tuple, Dict)


class ProfileWidget(pg.PlotWidget):

    def __init__(self,
                 *args,
                 model: Optional[LorenzMie] = None,
                 radius: int = 100,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configurePlot()
        self.model = model or LorenzMie()
        self._data = None
        self.radius = radius

    def _configurePlot(self) -> None:
        self.setBackground('w')
        self.showGrid(True, True, 0.2)
        opts = {'font-size': '14pt', 'color': 'gray'}
        self.setLabel('bottom', 'r [pixels]', **opts)
        self.setLabel('left', 'b(r)', **opts)
        pen = pg.mkPen('k', width=3, style=Qt.DashLine)
        self.addLine(y=1, pen=pen)
        pen = pg.mkPen('k', width=3)
        self.getAxis('bottom').setPen(pen)
        self.getAxis('left').setPen(pen)
        self.theory = pg.PlotCurveItem(pen=pg.mkPen('r', width=3))
        self.experiment = pg.PlotCurveItem(pen=pg.mkPen('k', width=3))
        pen = pg.mkPen('k', width=1, style=Qt.DashLine)
        self.upper = pg.PlotCurveItem(pen=pen)
        self.lower = pg.PlotCurveItem(pen=pen)
        brush = pg.mkBrush(255, 165, 0, 128)
        self.region = pg.FillBetweenItem(self.upper, self.lower, brush)
        self.addItem(self.theory)
        self.addItem(self.experiment)
        self.addItem(self.upper)
        self.addItem(self.lower)
        self.addItem(self.region)

    @pyqtProperty(dict)
    def properties(self) -> Dict[str, float]:
        return self.model.properties

    @properties.setter
    def properties(self, properties) -> None:
        if 'x_p' in properties:
            properties.pop('x_p')
        if 'y_p' in properties:
            properties.pop('y_p')
        if len(properties) == 0:
            return
        self.model.properties = properties
        self.plotTheory()

    @pyqtProperty(LorenzMie)
    def model(self) -> LorenzMie:
        return self._model

    @model.setter
    def model(self, model: LorenzMie) -> None:
        model.x_p = 0.0
        model.y_p = 0.0
        self._model = model

    @pyqtProperty(tuple)
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._data, self._stdev)

    @data.setter
    def data(self, data: Tuple[np.ndarray, np.ndarray]) -> None:
        self._data, self._stdev = data
        self.plotData()

    @pyqtProperty(int)
    def radius(self) -> int:
        return self._radius

    @radius.setter
    def radius(self, radius: int) -> None:
        self._radius = radius
        self.model.coordinates = np.arange(radius)
        self.plotData()
        self.plotTheory()

    def plotTheory(self) -> None:
        self.theory.setData(self.model.hologram())

    def plotData(self):
        if self._data is None:
            return
        radius = min(self.radius, len(self._data))
        data = self._data[0:radius]
        stdev = self._stdev[0:radius]
        self.experiment.setData(data)
        self.upper.setData(data + stdev)
        self.lower.setData(data - stdev)

    @pyqtSlot(str, float)
    def setProperty(self, name, value):
        self.model.properties = {name: value}


def example():
    from pylorenzmie.theory import AberratedLorenzMie as model

    app = pg.mkQApp()
    widget = ProfileWidget()
    widget.model = model()
    widget.radius = 150
    widget.show()
    app.exec_()


if __name__ == '__main__':
    example()
