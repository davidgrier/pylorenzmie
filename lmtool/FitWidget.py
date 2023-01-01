import pyqtgraph as pg
from pylorenzmie.analysis import Optimizer
from pylorenzmie.theory import LorenzMie
from PyQt5.QtCore import (pyqtProperty, QRectF, pyqtSlot)
from PyQt5.QtWidgets import QFileDialog
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import (Optional, Dict)


class FitWidget(pg.GraphicsLayoutWidget):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configurePlot()
        self.optimizer = Optimizer()
        self.fraction = 0.25
        self.result = None

    def _configurePlot(self) -> None:
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
        cm = pg.colormap.get('CET-D1')
        self.residuals.setColorMap(cm)
        self.residuals.setLevels((-10, 10))

    def mask(self, data: np.ndarray) -> np.ndarray:
        data = data.flatten()
        mask = np.random.choice([True, False], data.size,
                                p=[self.fraction, 1-self.fraction])
        mask[data == np.max(data)] = False
        return mask

    def optimize(self,
                 data: np.ndarray,
                 coords: np.ndarray) -> pd.Series:
        mask = self.mask(data)
        coords = coords.reshape((2, -1))
        self.optimizer.data = data.flatten()[mask]
        self.optimizer.model.coordinates = coords[:, mask]
        self.result = self.optimizer.optimize()
        self.optimizer.model.coordinates = coords
        hologram = self.optimizer.model.hologram().reshape(data.shape)
        hologram = np.clip(hologram, np.min(data), np.max(data))
        self.fit.setImage(hologram)
        noise = self.optimizer.model.instrument.noise
        self.residuals.setImage((data - hologram)/noise)
        self.fit.setRect(self.rect)
        self.residuals.setRect(self.rect)
        return self.result

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

    @pyqtSlot(str, object)
    def setSetting(self, name, value):
        if name == 'fraction':
            self.fraction = value
        else:
            self.optimizer.settings[name] = value

    def filename(self) -> str:
        directory = Path('~/data').expanduser()
        directory.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%m_%d_%Y-%H_%M_%S')
        return str(directory / f'lmtool_{timestamp}.h5')

    @pyqtSlot()
    def saveResult(self, filename: Optional[str] = None) -> None:
        filename = filename or self.filename()
        self.result.to_hdf(filename, 'result', mode='w')
        self.optimizer.metadata.to_hdf(filename, 'metadata')

    @pyqtSlot()
    def saveResultAs(self) -> None:
        get = QFileDialog.getSaveFileName
        filename, _ = get(self, 'Save Results',
                          self.filename(), 'HDF5 (*.h5)')
        if filename is not None:
            self.saveResult(filename)
