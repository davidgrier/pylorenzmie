import logging
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import pyqtSlot, QRectF
from pyqtgraph.Qt.QtGui import QShowEvent
from pylorenzmie.analysis import Optimizer
from pylorenzmie.lib.lmtypes import Coordinates, Image
from pylorenzmie.theory import LorenzMie

logger = logging.getLogger(__name__)


class FitWidget(pg.GraphicsLayoutWidget):
    '''Three-panel widget showing the ROI, model fit, and normalized residuals.'''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configurePlot()
        self.optimizer = Optimizer(model=LorenzMie())
        self.optimizer.fraction = 0.25
        self._data = None
        self.rect = None
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
