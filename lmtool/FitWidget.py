from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import (pyqtProperty, QRectF, pyqtSlot)
from pyqtgraph.Qt.QtWidgets import QFileDialog

from pylorenzmie.analysis import Optimizer
from pylorenzmie.theory import LorenzMie


class FitWidget(pg.GraphicsLayoutWidget):
    '''Three-panel widget showing the ROI, model fit, and normalized residuals.'''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configurePlot()
        self.optimizer = Optimizer()
        self.fraction = 0.25
        self.datafile = None
        self.result = None
        self._data = None
        self.rect = None

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
            plot.enableAutoRange(axis='xy', enable=True)
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
        self._regionPlot = plots[0]
        cm = pg.colormap.get('CET-D1')
        self.residuals.setColorMap(cm)
        self.residuals.setLevels((-10, 10))
        cb = pg.ColorBarItem(values=(-10, 10),
                             limits=(-10, 10),
                             interactive=False,
                             colorMap=cm,
                             pen=pen)
        self.addItem(cb)

    def mask(self, data: NDArray[float]) -> NDArray[bool]:
        data = data.flatten()
        mask = np.random.choice([True, False], data.size,
                                p=[self.fraction, 1-self.fraction])
        mask[data == np.max(data)] = False
        return mask

    def optimize(self,
                 data: NDArray[float],
                 coords: NDArray[float]) -> pd.Series:
        mask = self.mask(data)
        coords = coords.reshape((2, -1))
        self.optimizer.data = data.flatten()[mask]
        self.optimizer.model.coordinates = coords[:, mask]
        self.result = self.optimizer.optimize()
        self.optimizer.model.coordinates = coords
        self._data = data
        self._updateFitDisplay()
        return self.result

    def setData(self, data: NDArray[float], rect: QRectF,
                coordinates: NDArray[float]) -> None:
        '''Display data, compute and show the current model prediction.

        Parameters
        ----------
        data : ndarray
            Cropped hologram region.
        rect : QRectF
            Screen rectangle for positioning the images.
        coordinates : ndarray, shape (2, npts)
            Pixel coordinates for the cropped region.
        '''
        self._data = data
        self.rect = rect
        self.optimizer.model.coordinates = coordinates.reshape(2, -1)
        self.region.setImage(data)
        self.region.setRect(rect)
        self._updateFitDisplay()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._updateFitDisplay()

    def _updateFitDisplay(self) -> None:
        if self._data is None or self.rect is None:
            return
        if not self.isVisible():
            return
        hologram = self.optimizer.model.hologram().reshape(self._data.shape)
        hologram = np.clip(hologram, np.min(self._data), np.max(self._data))
        self.fit.setImage(hologram)
        noise = self.optimizer.model.instrument.noise
        self.residuals.setImage((self._data - hologram) / noise)
        self.fit.setRect(self.rect)
        self.residuals.setRect(self.rect)
        self._regionPlot.autoRange()

    def refreshPreview(self, properties: dict | None = None) -> None:
        '''Recompute and redisplay the model prediction without re-optimizing.

        Parameters
        ----------
        properties : dict, optional
            Model properties to apply before recomputing the hologram.
        '''
        if self._data is None:
            return
        if properties:
            self.optimizer.model.properties = properties
        self._updateFitDisplay()

    @pyqtProperty(LorenzMie)
    def model(self) -> LorenzMie:
        return self.optimizer.model

    @model.setter
    def model(self, model: LorenzMie) -> None:
        self.optimizer.model = model

    @pyqtProperty(dict)
    def properties(self) -> LorenzMie.Properties:
        return self.optimizer.model.properties

    @properties.setter
    def properties(self, properties: LorenzMie.Properties) -> None:
        self.optimizer.model.properties = properties

    @pyqtSlot(str, object)
    def setSetting(self, name: str, value: LorenzMie.Property) -> None:
        if name == 'fraction':
            self.fraction = value
        else:
            self.optimizer.settings[name] = value

    def filename(self) -> str:
        directory = Path('~/data/lmtool').expanduser()
        directory.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%m_%d_%Y-%H_%M_%S')
        return str(directory / f'result_{timestamp}.h5')

    @pyqtSlot()
    def saveResult(self, filename: str | None = None) -> None:
        if self.result is None:
            return
        filename = filename or self.filename()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=pd.io.pytables.PerformanceWarning)
            self.result.to_hdf(filename, 'result', mode='w')
            metadata = self.optimizer.metadata
            metadata['datafile'] = self.datafile
            metadata.to_hdf(filename, 'metadata')

    def saveJson(self, filename: str) -> None:
        if self.result is None:
            return
        s = pd.concat([self.result, self.optimizer.metadata])
        s['datafile'] = self.datafile
        s.to_json(filename, indent=4)

    @pyqtSlot()
    def saveResultAs(self) -> None:
        get = QFileDialog.getSaveFileName
        filename, _ = get(self, 'Save Results',
                          self.filename(),
                          'HDF5 (*.h5);;JSON (*.json)')
        if not filename:
            return
        if '.h5' in filename:
            self.saveResult(filename)
        elif '.json' in filename:
            self.saveJson(filename)
