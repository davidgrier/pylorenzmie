import pyqtgraph as pg
from pylorenzmie.analysis import Optimizer
from pylorenzmie.theory import LorenzMie
from PyQt5.QtCore import (pyqtProperty, QRectF)
import numpy as np
from typing import Dict


class FitWidget(pg.GraphicsLayoutWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configurePlot()
        self.optimizer = Optimizer()

    def _configurePlot(self):
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.setBackground('w')
        pen = pg.mkPen('k', width=2)
        plots = [self.addPlot(row=0, column=0),
                 self.addPlot(row=0, column=1),
                 self.addPlot(row=0, column=2)]
        for plot in plots:
            plot.getAxis('bottom').setPen(pen)
            plot.getAxis('left').setPen(pen)
            plot.setAspectLocked()
        options = dict(border=pen, axisOrder='row-major')
        self.region = pg.ImageItem(**options)
        self.fit = pg.ImageItem(**options)
        self.residuals = pg.ImageItem(**options)
        plots[0].addItem(self.region)
        plots[1].addItem(self.fit)
        plots[2].addItem(self.residuals)
        plots[1].setXLink(plots[0])
        plots[2].setXLink(plots[0])
        plots[1].setYLink(plots[0])
        plots[2].setYLink(plots[0])
        cm = pg.colormap.getFromMatplotlib('bwr')
        self.residuals.setColorMap(cm)

    def optimize(self, data, coords):
        self.optimizer.data = data.flatten()
        self.optimizer.model.coordinates = coords.reshape((2, -1))
        result = self.optimizer.optimize()
        hologram = self.optimizer.model.hologram().reshape(data.shape)
        self.fit.setImage(hologram)
        self.residuals.setImage(data - hologram)
        self.fit.setRect(self.rect)
        self.residuals.setRect(self.rect)
        return result

    def setData(self, data: np.ndarray, rect: QRectF) -> None:
        self.region.setImage(data)
        self.region.setRect(rect)
        self.rect = rect

    @pyqtProperty(LorenzMie)
    def model(self) -> LorenzMie:
        return self.optimizer.model

    @model.setter
    def model(self, model: LorenzMie) -> None:
        self.optimizer.model = model

    @pyqtProperty(dict)
    def properties(self) -> Dict[str, float]:
        return self.optimizer.model.properties

    @properties.setter
    def properties(self, properties: Dict[str, float]) -> None:
        self.optimizer.model.properties = properties
