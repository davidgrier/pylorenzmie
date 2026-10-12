from datetime import datetime
import logging
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import pyqtSignal, pyqtSlot, QRectF
from pyqtgraph.Qt.QtGui import QShowEvent
from pyqtgraph.Qt.QtWidgets import QFileDialog
from pylorenzmie.analysis import Optimizer
from pylorenzmie.lib.lmtypes import Coordinates, Image
from pylorenzmie.theory import LorenzMie

logger = logging.getLogger(__name__)


class FitWidget(pg.GraphicsLayoutWidget):
    '''Three-panel widget showing the ROI, model fit, and normalized residuals.'''

    #: Emitted with the file name when a result has been saved.
    resultSaved = pyqtSignal(str)
    #: Emitted with an error message when a result could not be saved.
    saveFailed = pyqtSignal(str)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configurePlot()
        self.optimizer = Optimizer(model=LorenzMie())
        self.optimizer.fraction = 0.25
        self.datafile = None
        self.result = None
        self._data = None
        self.rect = None
        self.metadata = None
        self._model_hologram = None

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

    def setResult(self, result: pd.Series,
                  metadata: pd.Series | None = None) -> None:
        '''Keep the result of a fit, so that it can be saved.

        Parameters
        ----------
        result : pandas.Series
            Fitted values, uncertainties and statistics.
        metadata : pandas.Series, optional
            The fixed parameters and the settings of the fit.
        '''
        self.result = result
        self.metadata = metadata

    def setData(self, data: Image, rect: QRectF,
                coordinates: Coordinates) -> None:
        '''Display data and wait for the model prediction.

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
        self._model_hologram = None
        self.region.setImage(data)
        self.region.setRect(rect)
        self._updateFitDisplay()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._updateFitDisplay()

    def showModel(self, hologram: np.ndarray | None,
                  noise: float = 0.05) -> None:
        '''Show the model prediction for the region that is displayed.

        Parameters
        ----------
        hologram : ndarray or None
            Model hologram of the region, one value per pixel, in the order
            of the data; None if there is none to show.
        noise : float, optional
            Noise that the residuals are divided by.  Default: 0.05.
        '''
        if hologram is None or self._data is None:
            self._model_hologram = None
            return
        if np.size(hologram) != self._data.size:
            return                    # the prediction is of another region
        self._model_hologram = (np.reshape(hologram, self._data.shape), noise)
        self._updateFitDisplay()

    def _updateFitDisplay(self) -> None:
        '''Refresh all three display panels from the model prediction.

        No-op when there is nothing to show or the widget is not visible.
        '''
        if (self._data is None or self.rect is None
                or self._model_hologram is None or not self.isVisible()):
            return
        hologram, noise = self._model_hologram
        # the color scale of the data applies to the model, which is not
        # clipped: the residuals show the whole difference
        levels = (np.min(self._data), np.max(self._data))
        self.region.setLevels(levels)
        self.fit.setImage(hologram, levels=levels)
        self.residuals.setImage((self._data - hologram) / noise)
        self.fit.setRect(self.rect)
        self.residuals.setRect(self.rect)
        self._regionPlot.autoRange()

    @pyqtSlot(str, object)
    def setSetting(self, name: str, value: LorenzMie.Property) -> None:
        # fraction is a direct Optimizer attribute, not part of .settings
        if name == 'fraction':
            self.optimizer.fraction = value
        else:
            self.optimizer.settings[name] = value

    def _output_path(self) -> str:
        directory = Path('~/data/lmtool').expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%m_%d_%Y-%H_%M_%S')
        return str(directory / f'result_{timestamp}.h5')

    @pyqtSlot()
    def saveResult(self, filename: str | None = None) -> None:
        '''Save the result of the last fit.

        Writes HDF5 if PyTables is installed and JSON otherwise (the file
        name then has the suffix ``.json``).  Emits :attr:`resultSaved`
        with the name of the file that was written, or :attr:`saveFailed`
        with the reason.  Without a ``filename`` the file goes to
        ``~/data/lmtool`` with a time stamp in its name.
        '''
        if self.result is None:
            self.saveFailed.emit('There is no result to save: optimize first')
            return
        try:
            filename = filename or self._output_path()
            try:
                self._saveHdf(filename)
            except ImportError:
                logger.warning('tables not installed; saving as JSON instead')
                filename = str(Path(filename).with_suffix('.json'))
                self._saveJson(filename)
        except Exception as error:
            logger.error(f'Could not save the result: {error!r}')
            self.saveFailed.emit(f'Could not save the result: {error}')
            return
        self.resultSaved.emit(filename)

    def _metadata(self) -> pd.Series:
        '''The fixed parameters and settings of the fit, and the file.'''
        metadata = (self.metadata if self.metadata is not None
                    else self.optimizer.metadata).copy()
        metadata['datafile'] = self.datafile
        return metadata

    def _saveHdf(self, filename: str) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=pd.errors.PerformanceWarning)
            self.result.to_hdf(filename, key='result', mode='w')
            metadata = self._metadata()
            metadata.to_hdf(filename, key='metadata')

    def _saveJson(self, filename: str) -> None:
        s = pd.concat([self.result, self._metadata()])
        s.to_json(filename, indent=4)

    def saveJson(self, filename: str) -> None:
        '''Save the result of the last fit as JSON.'''
        if self.result is None:
            self.saveFailed.emit('There is no result to save: optimize first')
            return
        try:
            self._saveJson(filename)
        except Exception as error:
            logger.error(f'Could not save the result: {error!r}')
            self.saveFailed.emit(f'Could not save the result: {error}')
            return
        self.resultSaved.emit(filename)

    @pyqtSlot()
    def saveResultAs(self) -> None:
        get = QFileDialog.getSaveFileName
        filename, _ = get(self, 'Save Results',
                          self._output_path(),
                          'HDF5 (*.h5);;JSON (*.json)')
        if not filename:
            return
        if Path(filename).suffix == '.json':
            self.saveJson(filename)
        else:
            self.saveResult(filename)
