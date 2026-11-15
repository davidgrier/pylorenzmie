import logging
from typing import Any, Optional, cast

import numpy as np
from qtpy.QtCore import Slot, QRectF
from qtpy.QtGui import QShowEvent
import pyqtgraph as pg

from pylorenzmie.analysis import Optimizer
from pylorenzmie.lib.lmtypes import Coordinates, Image
from pylorenzmie.lmtool.required import required
from pylorenzmie.theory import LorenzMie

logger = logging.getLogger(__name__)


class FitWidget(pg.GraphicsLayoutWidget):
    '''Three-panel widget of the ROI, the model fit and the residuals.'''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configurePlot()
        self.optimizer = Optimizer(model=LorenzMie())
        self.optimizer.fraction = 0.25
        self._data = None
        self._rect = None
        self._model_hologram = None

    def _configurePlot(self) -> None:
        self.ci.layout.setContentsMargins(  # pyright: ignore[reportAttributeAccessIssue]
            0, 0, 0, 0)
        self.setBackground('w')
        pen = pg.mkPen('k', width=2)
        # a PlotItem forwards the methods of its ViewBox at run time, which
        # the stubs of pyqtgraph do not show
        plots: list[Any] = [self.ci.addPlot(row=0, col=0),
                            self.ci.addPlot(row=0, col=1),
                            self.ci.addPlot(row=0, col=2)]
        for plot in plots:
            plot.getAxis('bottom').setPen(pen)
            plot.getAxis('left').setPen(pen)
            plot.setAspectLocked()
            plot.enableAutoRange(axis='xy', enable=True)
        self.region = pg.ImageItem(border=pen, axisOrder='row-major')
        self.fit = pg.ImageItem(border=pen, axisOrder='row-major')
        self.residuals = pg.ImageItem(border=pen, axisOrder='row-major')
        plots[0].addItem(self.region)
        plots[1].addItem(self.fit)
        plots[2].addItem(self.residuals)
        plots[1].setXLink(plots[0])
        plots[2].setXLink(plots[0])
        plots[1].setYLink(plots[0])
        plots[2].setYLink(plots[0])
        self._regionPlot: Any = plots[0]
        cm = required(pg.colormap.get('CET-D1'))
        self.residuals.setColorMap(cm)
        self.residuals.setLevels((-10, 10))
        cb = pg.ColorBarItem(values=(-10, 10),
                             limits=(-10, 10),
                             interactive=False,
                             colorMap=cm,
                             pen=pen)  # pyright: ignore[reportArgumentType]
        self.addItem(cb)

    def setData(self, data: Image,
                rect: QRectF,
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
        self._rect = rect
        self._model_hologram = None
        self.region.setImage(data)
        self.region.setRect(rect)
        self._updateFitDisplay()

    # Optional, and not a union with None: with Qt mocked, as in the docs
    # job, a class of Qt has no ``|``
    def showEvent(self, event: Optional[QShowEvent]) -> None:
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
        if (self._data is None or self._rect is None
                or self._model_hologram is None or not self.isVisible()):
            return
        hologram, noise = self._model_hologram
        # the color scale of the data applies to the model, which is not
        # clipped: the residuals show the whole difference
        levels = (np.min(self._data), np.max(self._data))
        self.region.setLevels(levels)
        self.fit.setImage(hologram, levels=levels)
        self.residuals.setImage((self._data - hologram) / noise)
        self.fit.setRect(self._rect)
        self.residuals.setRect(self._rect)
        self._regionPlot.autoRange()

    @Slot(str, object)
    def setSetting(self, name: str, value: LorenzMie.Property) -> None:
        # fraction is a direct Optimizer attribute, not part of .settings
        '''Set a setting of the optimizer by name.

        ``fraction`` is an attribute of the optimizer itself, and the
        other names are keys of its ``settings``.'''
        if name == 'fraction':
            self.optimizer.fraction = cast(float, value)
        else:
            self.optimizer.settings[name] = value
