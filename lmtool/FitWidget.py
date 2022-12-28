import pyqtgraph as pg
from pylorenzmie.analysis import Optimizer


class FitWidget(pg.GraphicsLayoutWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setupUi()
        self.optimizer = Optimizer()

    def _setupUi(self):
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.setBackground('w')
        options = dict(border=pg.mkPen('k', width=2),
                       axisOrder='row-major')
        self.region = pg.ImageItem(**options)
        self.fit = pg.ImageItem(**options)
        self.residuals = pg.ImageItem(**options)
        options = dict(enableMenu=False,
                       enableMouse=False,
                       invertY=False,
                       lockAspect=True)
        self.addViewBox(**options).addItem(self.region)
        self.addViewBox(**options).addItem(self.fit)
        self.addViewBox(**options).addItem(self.residuals)
        cm = pg.colormap.getFromMatplotlib('bwr')
        self.residuals.setColorMap(cm)

    def optimize(self, data, coords):
        self.optimizer.data = data.flatten()
        self.optimizer.model.coordinates = coords.reshape((2, -1))
        result = self.optimizer.optimize()
        hologram = self.optimizer.model.hologram().reshape(data.shape)
        self.fit.setImage(hologram)
        self.residuals.setImage(data - hologram)
        return result
