import pyqtgraph as pg
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, pyqtProperty)
import numpy as np
from typing import Optional


class ImageWidget(pg.GraphicsLayoutWidget):

    roiChanged = pyqtSignal(float, float)
    radiusChanged = pyqtSignal(int)

    def __init__(self,
                 *args,
                 data: Optional[np.ndarray] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._radius = 100
        self._configurePlot()
        self._connectSignals()
        self.data = data or np.ones((480, 640))

    def _configurePlot(self) -> None:
        self.setBackground('w')
        self.image = pg.ImageItem(border=pg.mkPen('k', width=2))
        self.image.axisOrder = 'row-major'
        options = dict(enableMenu=False,
                       enableMouse=False,
                       invertY=False,
                       lockAspect=True)
        self.addViewBox(**options).addItem(self.image)
        self.roi = pg.CircleROI([100, 100],
                                radius=self._radius,
                                parent=self.image)
        self.roi.setPen(pg.mkPen('w', width=3))
        self.roi.hoverPen = pg.mkPen('y', width=3)

    def _connectSignals(self) -> None:
        self.roi.sigRegionChangeFinished.connect(self.handleChange)

    @pyqtSlot(object)
    def handleChange(self, roi) -> None:
        radius = int(roi.size()[0]) // 2
        if radius == self._radius:
            x0, y0 = roi.pos()
            self.roiChanged.emit(x0+radius, y0+radius)
        else:
            self._radius = radius
            self.radiusChanged.emit(radius)

    @pyqtProperty(np.ndarray)
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.image.setImage(data)

    @pyqtProperty(float)
    def x_p(self) -> float:
        return self.roi.pos()[0] + self._radius

    @x_p.setter
    def x_p(self, x_p: float) -> None:
        pos = self.roi.pos()
        pos[0] = x_p - self._radius
        self.roi.setPos(pos)

    @pyqtProperty(float)
    def y_p(self) -> float:
        return self.roi.pos()[1] + self._radius

    @y_p.setter
    def y_p(self, y_p: float) -> None:
        pos = self.roi.pos()
        pos[1] = y_p - self._radius
        self.roi.setPos(pos)


def example():

    def radius(r):
        print(f'radius = {r}', end='\r')

    def position(x, y):
        print(f'position = ({x:.1f}, {y:.1f})', end='\r')

    app = pg.mkQApp()
    widget = ImageWidget()
    widget.show()
    widget.radiusChanged.connect(radius)
    widget.roiChanged.connect(position)
    app.exec_()


if __name__ == '__main__':
    example()
