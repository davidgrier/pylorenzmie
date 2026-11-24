from typing import Any

import numpy as np
from numpy.typing import NDArray
from qtpy.QtCore import Signal, Slot, QRectF
import pyqtgraph as pg


class ImageWidget(pg.GraphicsLayoutWidget):
    '''Image display with a circular ROI for selecting a particle region.'''

    #: Emitted with ``(x, y)`` centre coordinates when the ROI moves.
    roiChanged = Signal(float, float)
    #: Emitted with the new radius in pixels when the ROI is resized.
    radiusChanged = Signal(int)

    def __init__(self,
                 *args: Any,
                 data: NDArray[np.floating] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._radius = 100
        self._shape: tuple[int, ...] | None = None  # of the image shown
        self._configurePlot()
        self._connectSignals()
        self.data = np.ones((480, 640)) if data is None else data

    def _configurePlot(self) -> None:
        self.setBackground('w')
        pen = pg.mkPen('k', width=2)
        self.image = pg.ImageItem(border=pen)
        self.image.axisOrder = 'row-major'
        # a PlotItem forwards the methods of its ViewBox at run time, which
        # the stubs of pyqtgraph do not show
        plot: Any = self.ci.addPlot(row=0, col=0)
        plot.addItem(self.image)
        plot.getAxis('bottom').setPen(pen)
        plot.getAxis('left').setPen(pen)
        plot.setAspectLocked()
        pen = pg.mkPen('w', width=3)
        hpen = pg.mkPen('y', width=3)
        self.roi = pg.CircleROI([0, 0],
                                pen=pen,
                                handlePen=pen,
                                hoverPen=hpen,
                                handleHoverPen=hpen,
                                radius=self._radius,
                                parent=self.image)

    def _connectSignals(self) -> None:
        self.roi.sigRegionChangeFinished.connect(self._handleChange)

    @Slot(object)
    def _handleChange(self, roi: pg.ROI) -> None:
        radius = int(self.roi.size()[0]) // 2
        x0, y0 = roi.pos()
        if radius != self._radius:
            self._radius = radius
            self.radiusChanged.emit(radius)
        self.roiChanged.emit(x0 + radius, y0 + radius)

    @property
    def data(self) -> NDArray[np.floating]:
        '''The image that is shown.  Setting it keeps the ROI where it is if
        the new image has the shape of the old one.'''
        return self._data

    @data.setter
    def data(self, data: NDArray[np.floating]) -> None:
        changed = data.shape != self._shape
        self._shape = data.shape
        self._data = data
        self.image.setImage(data)
        h, w = data.shape
        self.image.setRect(QRectF(0, 0, w, h))
        if changed:
            self.roi.setPos([0, 0])
            self.roi.setSize([200, 200])

    @property
    def x_p(self) -> float:
        '''Horizontal position of the center of the ROI, in pixels.'''
        return self.roi.pos()[0] + self._radius

    @x_p.setter
    def x_p(self, x_p: float) -> None:
        pos = self.roi.pos()
        pos[0] = x_p - self._radius
        self.roi.setPos(pos)

    @property
    def y_p(self) -> float:
        '''Vertical position of the center of the ROI, in pixels.'''
        return self.roi.pos()[1] + self._radius

    @y_p.setter
    def y_p(self, y_p: float) -> None:
        pos = self.roi.pos()
        pos[1] = y_p - self._radius
        self.roi.setPos(pos)

    @property
    def radius(self) -> int:
        '''Radius of the ROI, in pixels.'''
        return self._radius

    def setRegion(self, x_p: float, y_p: float, radius: int) -> None:
        '''Place the ROI with its center at ``(x_p, y_p)``.

        Emits :attr:`roiChanged` (and :attr:`radiusChanged` if the
        radius changed) once.
        '''
        self.roi.setSize([2*radius, 2*radius], finish=False)
        self.roi.setPos([x_p - radius, y_p - radius])

    def rect(self) -> QRectF:  # pyright: ignore[reportIncompatibleMethodOverride]
        '''The rectangle of the ROI, in pixels of the image.'''
        pos = self.roi.pos()
        size = self.roi.size()
        return QRectF(*pos, *size)

    @classmethod
    def example(cls) -> None:

        '''Run a demonstration of the widget, for use as a script.'''
        def radius(r: int) -> None:
            print(f'radius = {r}', end='\r')

        def position(x: float, y: float) -> None:
            print(f'position = ({x:.1f}, {y:.1f})', end='\r')

        app = pg.mkQApp()
        widget = cls()
        widget.show()
        widget.radiusChanged.connect(radius)
        widget.roiChanged.connect(position)
        app.exec()


if __name__ == '__main__':  # pragma: no cover
    ImageWidget.example()
