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
        self.fraction = 0.5

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

    def mask(self, data):
        data = data.flatten()
        mask = np.random.choice([True, False], data.size,
                                p=[self.fraction, 1-self.fraction])
        mask[data == np.max(data)] = False
        return mask

    def optimize(self, data, coords):
        mask = self.mask(data)
        coords = coords.reshape((2, -1))
        self.optimizer.data = data.flatten()[mask]
        self.optimizer.model.coordinates = coords[:, mask]
        result = self.optimizer.optimize()
        self.optimizer.model.coordinates = coords
        hologram = self.optimizer.model.hologram().reshape(data.shape)
        hologram = np.clip(hologram, np.min(data), np.max(data))
        self.fit.setImage(hologram)
        noise = self.optimizer.model.instrument.noise
        self.residuals.setImage((data - hologram)/noise)
        self.residuals.setLevels((-10, 10))
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
