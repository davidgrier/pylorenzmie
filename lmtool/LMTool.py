# /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pylorenzmie.utilities import (azistd, coordinates)
import json

from PyQt5.QtCore import (pyqtProperty, pyqtSlot)
from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from PyQt5 import uic
from typing import (Optional, Union)
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

'''
To do
* support for sampling which pixels to fit
* saving parameters
* correct normalization (currently it simply divides by the mean)
* interactive residuals (currently only updates after fits)
* support for cuda-accelerated kernels.
* set ROI for new picture
'''


class LMTool(QMainWindow):

    uifile = 'LMTool.ui'

    def __init__(self,
                 filename: Optional[str] = None,
                 background: Union[str, float, None] = None):
        super(LMTool, self).__init__()
        uic.loadUi(self.uifile, self)
        self._setupTheory()
        self.readHologram(filename)
        self._connectSignals()

    def _setupTheory(self):
        model = type(self.profileWidget.model)()
        self.fitWidget.optimizer.model = model

    @pyqtProperty(object)
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
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

    def _connectSignals(self) -> None:
        self.imageWidget.roiChanged.connect(self._handleROIChanged)
        self.imageWidget.radiusChanged.connect(self._handleRadiusChanged)
        self.controls.propertyChanged.connect(self._handlePropertyChanged)
        self.actionOpen.triggered.connect(self.readHologram)
        self.actionSave_Parameters.triggered.connect(self.saveParameters)
        self.actionOptimize.triggered.connect(self.optimize)

    def _updateProfile(self):
        x_p = self.controls.x_p.value()
        y_p = self.controls.y_p.value()
        self.profileWidget.data = azistd(self._data, (x_p, y_p))

    @pyqtSlot(float, float)
    def _handleROIChanged(self, x_p, y_p):
        self.controls.blockSignals(True)
        self.controls.properties = dict(x_p=x_p, y_p=y_p)
        self.controls.blockSignals(False)
        self._updateProfile()
        data, coords = self.crop()
        self.fitWidget.region.setImage(data)

    @pyqtSlot(int)
    def _handleRadiusChanged(self, radius):
        self.profileWidget.radius = radius
        data, coords = self.crop()
        self.fitWidget.region.setImage(data)

    @pyqtSlot(str, float)
    def _handlePropertyChanged(self, name, value):
        if name == 'x_p':
            self.imageWidget.x_p = value
        elif name == 'y_p':
            self.imageWidget.y_p = value
        self.profileWidget.properties = {name: value}

    def crop(self):
        x0, y0 = list(map(int, self.imageWidget.roi.pos()))
        w, h = list(map(int, self.imageWidget.roi.size()))
        x1, y1 = x0+w, y0+h
        return self.data[y0:y1, x0:x1], self.coordinates[:, y0:y1, x0:x1]

    @pyqtSlot()
    def optimize(self):
        self.statusBar().showMessage('Optimizing parameters...')
        logger.info('Starting optimization...')
        optimizer = self.fitWidget.optimizer
        optimizer.model.properties = self.controls.properties
        optimizer.fixed = self.controls.fixed
        optimizer.robust = self.actionRobust.isChecked()
        result = self.fitWidget.optimize(*self.crop())
        self.controls.properties = optimizer.model.properties
        logger.info(f'Finished!\n{result}')
        self.statusBar().showMessage('Optimization complete')

    ###
    #
    # Routines for loading data
    #
    def setupData(self, data, background):
        if type(data) is str:
            self.openHologram(data)
        else:
            self.data = data.astype(float)
        if not self.autonormalize:
            if type(background) is str:
                self.openBackground(background)
            elif type(background) is int:
                self.frame.background = background

    @pyqtSlot()
    def openBackground(self, filename=None):
        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(
                self, 'Open Background', '', 'Images (*.png)')
        background = cv2.imread(filename, 0).astype(float)
        if background is None:
            return
        self.frame.background = background


def main():
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
    lmtool = LMTool(args.filename, background)
    lmtool.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
