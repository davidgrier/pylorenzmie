# /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pylorenzmie.utilities import (azistd, coordinates)
from pylorenzmie.lmtool.LMWidget import LMWidget
import json
from PyQt5.QtCore import (pyqtProperty, pyqtSlot)
from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from PyQt5 import uic
from typing import (Type, Optional, Union, Tuple)
import logging





logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

'''
To do
* correct normalization (currently it simply divides by the mean)
* interactive residuals (currently only updates after fits)
* support for cuda-accelerated kernels.
* dark count
* reorganize so that implementation is separate from executable
* toml config file
'''


class LMTool(QMainWindow):

    uiFile = 'LMTool.ui'

    def __init__(self,
                 controls: Type[LMWidget],
                 filename: Optional[str] = None,
                 background: Union[str, float, None] = None):
        super(LMTool, self).__init__()
        uic.loadUi(self.uiFile, self)
        self._setupTheory(controls())
        self.readHologram(filename)
        self._connectSignals()

    def _setupTheory(self, controls: LMWidget) -> None:
        layout = self.controls.parent().layout()
        layout.replaceWidget(self.controls, controls)
        self.controls.close()
        self.controls = controls
        self.profileWidget.model = self.controls.cls()
        self.profileWidget.properties = self.controls.properties
        self.profileWidget.radius = self.imageWidget.radius
        self.fitWidget.model = self.controls.cls()
        self.fitWidget.properties = self.controls.properties
        self.optimizerWidget.settings = self.fitWidget.optimizer.settings

    def _connectSignals(self) -> None:
        self.imageWidget.roiChanged.connect(self._handleROIChanged)
        self.imageWidget.radiusChanged.connect(self._handleRadiusChanged)
        self.controls.propertyChanged.connect(self._handlePropertyChanged)
        self.actionOpen.triggered.connect(self.readHologram)
        self.actionSaveParameters.triggered.connect(self.saveParameters)
        self.saveResult.triggered.connect(self.fitWidget.saveResult)
        self.saveResultAs.triggered.connect(self.fitWidget.saveResultAs)
        self.actionRobust.toggled.connect(self.setRobust)
        self.actionOptimize.triggered.connect(self.optimize)
        connect = self.optimizerWidget.settingChanged.connect
        connect(self.fitWidget.setSetting)

    @pyqtProperty(np.ndarray)
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self._data = data / np.mean(data)
        self.coordinates = coordinates(data.shape, flatten=False)
        self.controls.x_p.setRange((0, data.shape[1]-1))
        self.controls.y_p.setRange((0, data.shape[0]-1))
        self.imageWidget.data = self._data
        self._updateProfile()

    @pyqtSlot()
    def readHologram(self, filename: Optional[str] = None) -> None:
        if filename is None:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self, 'Open Hologram', '', 'Images (*.png)')
        self.fitWidget.datafile = filename
        if filename is None:
            return
        self.data = cv2.imread(filename, 0).astype(float)

    @pyqtSlot()
    def saveParameters(self, filename: Optional[str] = None) -> None:
        if filename is None:
            get = QFileDialog.getSaveFileName
            filename, _ = get(self, 'Save Parameters', '', 'JSON (*.json)')
        if filename is None:
            return
        properties = self.controls.properties
        with open(filename, 'w') as f:
            json.dump(properties, f, indent=4, sort_keys=True)

    def _updateProfile(self) -> None:
        x_p = self.controls.x_p.value()
        y_p = self.controls.y_p.value()
        self.profileWidget.data = azistd(self._data, (x_p, y_p))

    @pyqtSlot(float, float)
    def _handleROIChanged(self, x_p: float, y_p: float) -> None:
        self.controls.blockSignals(True)
        self.controls.properties = dict(x_p=x_p, y_p=y_p)
        self.controls.blockSignals(False)
        self._updateProfile()
        self.fitWidget.setData(*self.crop(True))

    @pyqtSlot(int)
    def _handleRadiusChanged(self, radius: int) -> None:
        self.profileWidget.radius = radius
        self.fitWidget.setData(*self.crop(True))

    @pyqtSlot(str, float)
    def _handlePropertyChanged(self, name: str, value: float) -> None:
        if name == 'x_p':
            self.imageWidget.x_p = value
        elif name == 'y_p':
            self.imageWidget.y_p = value
        self.profileWidget.properties = {name: value}

    def crop(self, rect: bool = False) -> Tuple:
        get = self.imageWidget.roi.getArraySlice
        (sy, sx), _ = get(self._data, self.imageWidget.image)
        if rect:
            return self.data[sy, sx], self.imageWidget.rect()
        else:
            return self.data[sy, sx], self.coordinates[:, sy, sx]

    @pyqtSlot(bool)
    def setRobust(self, state: bool) -> None:
        optimizer = self.fitWidget.optimizer
        optimizer.robust = state
        self.optimizerWidget.settings = optimizer.settings

    @pyqtSlot()
    def optimize(self) -> None:
        logger.info('Starting optimization...')
        optimizer = self.fitWidget.optimizer
        optimizer.model.properties = self.controls.properties
        optimizer.fixed = self.controls.fixed
        optimizer.robust = self.actionRobust.isChecked()
        result = self.fitWidget.optimize(*self.crop())
        self.controls.properties = optimizer.model.properties
        logger.info(f'Finished!\n{result}')
        self.statusBar().showMessage('Optimization complete', 2000)


def lmtool():
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
    from PyQt5.QtWidgets import QApplication
    from pathlib import Path
    import sys
    import argparse

    basedir = Path(__file__).parent.parent.resolve()
    filename = basedir / 'docs' / 'tutorials' / 'crop.png'
    filename = str(filename)

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default=filename,
                        nargs='?', action='store')
    parser.add_argument('-b', '--background', dest='background',
                        default=None, action='store',
                        help='background value or file name')
    args, unparsed = parser.parse_known_args()
    qt_args = sys.argv[:1] + unparsed

    background = args.background
    if background is not None and background.isdigit():
        background = int(background)

    app = QApplication(qt_args)
    lmtool = LMTool(ALMWidget, args.filename, background)
    lmtool.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    lmtool()
