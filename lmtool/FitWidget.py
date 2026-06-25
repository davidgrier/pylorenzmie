from collections.abc import Callable
from datetime import datetime
import logging
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import (pyqtProperty, pyqtSignal, pyqtSlot,
                                  QObject, QRectF, QThread)
from pyqtgraph.Qt.QtGui import QShowEvent
from pyqtgraph.Qt.QtWidgets import QFileDialog
from pylorenzmie.analysis import Optimizer
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.lib.lmtypes import Coordinates, Image
from pylorenzmie.theory import LorenzMie

logger = logging.getLogger(__name__)


class _Worker(QObject):
    '''Background thread worker for a single-argument callable.'''

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn: Callable, arg: object) -> None:
        super().__init__()
        self._fn = fn
        self._arg = arg

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.finished.emit(self._fn(self._arg))
        except Exception as e:
            self.error.emit(str(e))


class FitWidget(pg.GraphicsLayoutWidget):
    '''Three-panel widget showing the ROI, model fit, and normalized residuals.'''

    #: Emitted when an optimization thread is launched.
    optimizationStarted = pyqtSignal()
    #: Emitted with the ``pd.Series`` result when optimization succeeds.
    optimizationFinished = pyqtSignal(object)
    #: Emitted with an error message string when optimization fails.
    optimizationError = pyqtSignal(str)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configurePlot()
        self.optimizer = Optimizer(model=LorenzMie())
        self.optimizer.fraction = 0.25
        self.datafile = None
        self.result = None
        self._data = None
        self.rect = None
        self._thread = None
        self._worker = None
        self._opt_hologram = None

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

    def optimize(self,
                 data: Image,
                 coordinates: Coordinates) -> pd.Series:
        '''Fit the model to data and update the display.

        Blocks the calling thread until the fit completes. For non-blocking
        use in a GUI context call :meth:`optimizeAsync` instead.

        Parameters
        ----------
        data : ndarray
            Normalized hologram crop.
        coordinates : ndarray, shape (2, npts)
            Pixel coordinates for the crop.

        Returns
        -------
        result : pandas.Series
            Fitted parameters and uncertainties.
        '''
        hologram = Hologram._from_slice(data, coordinates)
        self.optimizer.mask.exclude = (data == np.max(data))
        self.result = self.optimizer.optimize(hologram)
        self.optimizer.model.coordinates = hologram.flat_coordinates
        self._data = data
        self._updateFitDisplay()
        return self.result

    def optimizeAsync(self,
                      data: Image,
                      coordinates: Coordinates) -> None:
        '''Start optimization in a background thread.

        Returns immediately. Emits :attr:`optimizationStarted` on entry,
        then :attr:`optimizationFinished` (or :attr:`optimizationError`)
        when the thread completes. Use :meth:`optimize` for synchronous
        (blocking) operation.

        Parameters
        ----------
        data : ndarray
            Normalized hologram crop.
        coordinates : ndarray, shape (2, npts)
            Pixel coordinates for the crop.
        '''
        if self._thread is not None and self._thread.isRunning():
            return
        hologram = Hologram._from_slice(data, coordinates)
        self.optimizer.mask.exclude = (data == np.max(data))
        self._opt_hologram = hologram
        self._data = data
        worker = _Worker(self.optimizer.optimize, hologram)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._onWorkerFinished)
        worker.error.connect(self._onWorkerError)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(self._onThreadFinished)
        self._worker = worker
        self._thread = thread
        thread.start()
        self.optimizationStarted.emit()

    @pyqtSlot(object)
    def _onWorkerFinished(self, result: pd.Series) -> None:
        self.result = result
        self.optimizer.model.coordinates = self._opt_hologram.flat_coordinates
        self._updateFitDisplay()
        self.optimizationFinished.emit(result)

    @pyqtSlot(str)
    def _onWorkerError(self, message: str) -> None:
        self.optimizationError.emit(message)

    @pyqtSlot()
    def _onThreadFinished(self) -> None:
        self._worker = None
        self._thread = None

    def shutdown(self) -> None:
        '''Stop any running optimization thread and wait for it to finish.'''
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    def setData(self, data: Image, rect: QRectF,
                coordinates: Coordinates) -> None:
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

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._updateFitDisplay()

    def _updateFitDisplay(self) -> None:
        '''Recompute the model hologram and refresh all three display panels.

        No-op when data has not been loaded or the widget is not visible.
        '''
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
        # fraction is a direct Optimizer attribute, not part of .settings
        if name == 'fraction':
            self.optimizer.fraction = value
        else:
            self.optimizer.settings[name] = value

    def _output_path(self) -> str:
        directory = Path('~/data/lmtool').expanduser()
        directory.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%m_%d_%Y-%H_%M_%S')
        return str(directory / f'result_{timestamp}.h5')

    @pyqtSlot()
    def saveResult(self, filename: str | None = None) -> None:
        if self.result is None:
            return
        filename = filename or self._output_path()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', category=pd.errors.PerformanceWarning)
                self.result.to_hdf(filename, 'result', mode='w')
                metadata = self.optimizer.metadata
                metadata['datafile'] = self.datafile
                metadata.to_hdf(filename, 'metadata')
        except ImportError:
            logger.warning('tables not installed; saving as JSON instead')
            self.saveJson(str(Path(filename).with_suffix('.json')))

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
                          self._output_path(),
                          'HDF5 (*.h5);;JSON (*.json)')
        if not filename:
            return
        suffix = Path(filename).suffix
        if suffix == '.h5':
            self.saveResult(filename)
        elif suffix == '.json':
            self.saveJson(filename)
