#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyqtgraph.Qt import uic
from pyqtgraph.Qt.QtCore import pyqtProperty, pyqtSlot, QRectF
from pyqtgraph.Qt.QtGui import QAction, QCloseEvent
from pyqtgraph.Qt.QtWidgets import (QAbstractItemView, QFileDialog,
                                     QHeaderView, QLabel, QMainWindow,
                                     QProgressBar, QTableView)

from pylorenzmie.lib import (Azimuthal, LMObject)
from pylorenzmie.lib.lmtypes import Coordinates, Image
from pylorenzmie.lmtool.LMWidget import LMWidget
from pylorenzmie.lmtool.ParticleTable import ParticleTable
from pylorenzmie.lmtool.Preview import Preview
from pylorenzmie.lmtool.results import save_results
from pylorenzmie.lmtool.tasks import (FitResult, LocalizeRequest, Snapshot,
                                       TaskRunner, run_estimate, run_fit,
                                       run_localize)
from pylorenzmie.theory import BACKENDS, model_class
from pylorenzmie.utilities import Normalizer, estimate_noise

_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)


# TODO: interactive residuals (currently only updates after fits)
# TODO: support for cuda-accelerated kernels
# TODO: toml config file
# TODO: support for gamma correction


class LMTool(QMainWindow):
    '''Main window for the LMTool hologram-fitting application.

    Parameters
    ----------
    controls : type[LMWidget]
        LMWidget subclass (not an instance) providing parameter controls.
    filename : str, optional
        Path to a hologram image to load on startup.
    background : str or float, optional
        Background of the holograms: a number, or the name of an image file
        (see :meth:`~pylorenzmie.utilities.normalizer.Normalizer.read`).
        Default: normalize each hologram by its median.
    dark : float, optional
        Dark count (black level) of the camera, in gray levels: subtracted
        from the hologram and the background before the division (see
        :class:`~pylorenzmie.utilities.normalizer.Normalizer`).
        Default: 0.
    '''

    uiFile = 'LMTool.ui'

    #: Number of candidate particles that Find Particles keeps, the
    #: strongest first.  A noisy image can have hundreds of candidates.
    MAX_CANDIDATES = 10

    #: How far the ROI can be from a particle, in pixels, and still be on it.
    PARTICLE_TOLERANCE = 10.

    def __init__(self,
                 controls: type[LMWidget],
                 filename: str | None = None,
                 background: str | float | None = None,
                 dark: float = 0.) -> None:
        super().__init__()
        uic.loadUi(_DIR / self.uiFile, self)
        self.dark = float(dark)
        self.normalizer = Normalizer(dark=self.dark)
        self._refused = ''        # why the last background was not used
        self._raw = None
        self._data = None
        self.coordinates = None
        self._pre_estimate = None
        self._moving = False
        self._nfound = 0
        self._datafile = None
        self._metadata = None
        self._row = -1
        self._search_again = False
        self.particles = ParticleTable()
        self.tasks = TaskRunner()
        self._setupTheory(controls())
        self._setupStatusBar()
        self._setupParticles()
        self._connectSignals()
        if background is not None:
            self._useBackground(background)
        self.readHologram(filename)

    def _setupTheory(self, controls: LMWidget) -> None:
        layout = self.controls.parent().layout()
        layout.replaceWidget(self.controls, controls)
        self.controls.close()
        self.controls = controls
        self.session = controls.session
        self.preview = Preview(self.session)
        self.preview.updated.connect(self._showPreview)
        self.profileWidget.model = None       # the preview has the model
        self.profileWidget.radius = self.imageWidget.radius
        self.preview.setRadius(self.imageWidget.radius)
        self.optimizerWidget.settings = self.fitWidget.optimizer.settings

    def _setupParticles(self) -> None:
        '''The Particles tab, and the actions that add and remove particles.'''
        view = QTableView()
        view.setModel(self.particles)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        view.verticalHeader().hide()
        header = view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        view.setColumnWidth(0, 30)
        self.particleView = view
        self.tabs.addTab(view, 'particles')
        self.actionAddParticle = QAction('Add Particle', self)
        self.actionAddParticle.setShortcut('Ctrl+Shift+A')
        self.actionRemoveParticle = QAction('Remove Particle', self)
        self.actionRemoveParticle.setShortcut('Ctrl+Backspace')
        self.menuParticles.addSeparator()
        self.menuParticles.addAction(self.actionAddParticle)
        self.menuParticles.addAction(self.actionRemoveParticle)

    def _setupStatusBar(self) -> None:
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumWidth(150)
        self._progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress)
        # what the holograms are divided by, and why not what was asked
        # for: a message would be replaced by the next one
        self._background = QLabel()
        self.statusBar().addPermanentWidget(self._background)
        self._showBackground(self.normalizer)

    def _showBackground(self, normalizer: Normalizer,
                        fallback: str = '') -> None:
        '''Show the background that ``normalizer`` divides by, and what is
        wrong: why a hologram is divided by the median instead
        (``fallback``), or why the last background was refused.'''
        if normalizer.method != 'reference':
            used = normalizer.method
        elif normalizer.source:
            used = Path(normalizer.source).name
        elif np.ndim(normalizer.reference) == 0:
            used = f'{normalizer.reference:g}'
        else:
            used = 'image'
        problem = fallback or self._refused
        dark = f', dark {normalizer.dark:g}' if normalizer.dark else ''
        self._background.setText(f'background: {used}{dark}'
                                 + (f' ({problem})' if problem else ''))

    def _connectSignals(self) -> None:
        self.imageWidget.roiChanged.connect(self._handleROIChanged)
        self.imageWidget.radiusChanged.connect(self._handleRadiusChanged)
        self.session.parametersChanged.connect(self._onParametersChanged)
        self.actionOpen.triggered.connect(self.readHologram)
        self.actionOpenBackground.triggered.connect(self.readBackground)
        self.actionEstimateNoise.triggered.connect(self.estimateNoise)
        self.actionFindParticles.triggered.connect(self.findParticles)
        self.actionNextParticle.triggered.connect(self.nextParticle)
        self.actionAddParticle.triggered.connect(self.addParticle)
        self.actionRemoveParticle.triggered.connect(self.removeParticle)
        self.particles.currentChanged.connect(self._onCurrentParticle)
        self.particleView.selectionModel().currentRowChanged.connect(
            lambda current, _: self.particles.setCurrent(current.row()))
        self.actionEstimate.triggered.connect(self.estimate)
        self.actionUndoEstimate.triggered.connect(self.undoEstimate)
        self.actionSaveParameters.triggered.connect(self.saveParameters)
        self.saveResult.triggered.connect(self.saveResults)
        self.saveResultAs.triggered.connect(self.saveResultsAs)
        self.actionRobust.toggled.connect(self.setRobust)
        self.actionOptimize.triggered.connect(self.optimize)
        self.optimizerWidget.settingChanged.connect(
            self.fitWidget.setSetting)
        self.actionCancel.triggered.connect(self.tasks.cancel)
        self.tasks.started.connect(self._onTaskStarted)
        self.tasks.finished.connect(self._onTaskFinished)
        self.tasks.failed.connect(self._onTaskFailed)
        self.tasks.cancelled.connect(self._onTaskCancelled)

    def _setUIBusy(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        self.controls.setEnabled(not busy)
        self.imageWidget.setEnabled(not busy)
        self.actionOptimize.setEnabled(not busy)
        self.actionEstimate.setEnabled(not busy)
        self.actionFindParticles.setEnabled(not busy)
        self.actionEstimateNoise.setEnabled(not busy)
        self.actionNextParticle.setEnabled(not busy)
        self.actionCancel.setEnabled(busy)

    @pyqtProperty(np.ndarray)
    def data(self) -> Image:
        return self._data

    @data.setter
    def data(self, data: Image) -> None:
        self._raw = data
        try:
            self._data = self.normalizer(data)
            self._showBackground(self.normalizer)
        except ValueError as error:
            logger.error(f'Could not normalize by the background: {error}')
            fallback = Normalizer(dark=self.dark)
            self._data = fallback(data)
            self._showBackground(fallback, f'{error}; using the median')
        self.coordinates = LMObject.meshgrid(data.shape, flatten=False)
        self.controls.x_p.setRange((0, data.shape[1]-1))
        self.controls.y_p.setRange((0, data.shape[0]-1))
        self.imageWidget.data = self._data
        self._setNoise(self._data)
        self._updateProfile()
        self._showCrop()

    @pyqtSlot()
    def readHologram(self, filename: str | None = None) -> None:
        if not filename:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self, 'Open Hologram', self._datafile or '',
                              'Images (*.png *.tif *.tiff)')
        if not filename:
            return
        data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if data is None:
            logger.error(f'Could not read image: {filename!r}')
            self.statusBar().showMessage(f'Could not read: {filename}', 5000)
            return
        self._datafile = filename
        self.particles.clear()
        self.data = data.astype(float)
        self.findParticles()

    def _setNoise(self, image: Image) -> None:
        '''Set the noise of the model to the estimate for ``image``.'''
        low, high = (self.controls.noise.range()
                     if hasattr(self.controls, 'noise') else (0., np.inf))
        noise = float(np.clip(estimate_noise(image), low, high))
        self.session.update(dict(noise=noise))

    @pyqtSlot()
    def estimateNoise(self) -> None:
        '''Estimate the noise from the region of interest.

        The estimate is the width of the pixel-to-pixel variation, which
        does not include the fringes or a smooth background.  Every
        residual in the Fit tab is divided by it, so the reduced chi-squared
        of a fit is meaningful only when it is right.
        '''
        if self._data is None:
            return
        data, _, _ = self.crop()
        self._setNoise(data)
        self.statusBar().showMessage(
            f'Noise: {self.session.get("noise"):.3f}', 5000)

    @pyqtSlot()
    def findParticles(self) -> None:
        '''Look for particles in the image, in the background.

        The strongest candidates go in the Particles tab, and the ROI moves to
        the first one, with the size of the region that encloses its fringes.
        :meth:`nextParticle` selects the others.  Particles that were added
        by hand stay.
        '''
        if self._data is None:
            return
        if self.tasks.busy:
            if self.tasks.name == 'localize':    # an older image: start over
                self._search_again = True
                self.tasks.cancel()
            return
        self.tasks.start('localize', run_localize,
                         LocalizeRequest(self._data, self.MAX_CANDIDATES))

    def _onLocalized(self, found: pd.DataFrame) -> None:
        self._nfound = found.attrs.get('nfound', len(found))
        self.particles.setFound(found)
        if self.particles.count == 0:
            self.statusBar().showMessage(
                'No particles found: move the ROI by hand and add it '
                '(Ctrl+Shift+A)', 8000)
            return
        self.nextParticle()

    @pyqtSlot()
    def nextParticle(self) -> None:
        '''Select the next particle in the Particles tab that is checked.'''
        if self.particles.count == 0:
            self.findParticles()
            return
        row = self.particles.next()
        if row is None:
            self.statusBar().showMessage('No particle is checked', 5000)
            return
        self.particles.setCurrent(row)

    @pyqtSlot()
    def addParticle(self) -> None:
        '''Add the region of interest to the particles.'''
        if self._data is None:
            return
        row = self.particles.add(self.session.get('x_p'),
                                 self.session.get('y_p'),
                                 self.imageWidget.radius)
        self.particles.setCurrent(row)

    @pyqtSlot()
    def removeParticle(self) -> None:
        '''Remove the particle that is selected.'''
        self.particles.remove(self.particles.current)

    @pyqtSlot(int)
    def _onCurrentParticle(self, row: int) -> None:
        '''Move the ROI to the particle that was selected.'''
        if row < 0:
            return
        selection = self.particleView.selectionModel()
        if selection.currentIndex().row() != row:
            self.particleView.selectRow(row)
        entry = self.particles.entry(row)
        x_p, y_p = self.particles.position(row)
        self.imageWidget.setRegion(x_p, y_p, entry['radius'])
        if entry['properties'] is not None:
            self.session.update(entry['properties'])
        n = self.particles.count
        message = (f'Particle {row + 1} of {n} at ({x_p:.0f}, {y_p:.0f})')
        if entry['source'] == 'found' and self._nfound > n:
            message += (f', among the strongest of {self._nfound} candidates: '
                        'the image may be noisy')
        elif n == 1 and entry['source'] == 'found':
            message = f'Only one particle found, at ({x_p:.0f}, {y_p:.0f})'
        self.statusBar().showMessage(message, 8000)

    def _particleRow(self) -> int:
        '''Row of the particle that the ROI is on, adding one if there is
        none.'''
        x_p, y_p = self.session.get('x_p'), self.session.get('y_p')
        row = self.particles.current
        if row >= 0:
            x, y = self.particles.position(row)
            if np.hypot(x - x_p, y - y_p) <= self.PARTICLE_TOLERANCE:
                return row
        row = self.particles.rowAt(x_p, y_p, self.PARTICLE_TOLERANCE)
        if row is None:
            row = self.particles.add(x_p, y_p, self.imageWidget.radius)
        return row

    def _background_start(self) -> str:
        '''Where the dialog for a background starts: at the last background
        or, failing that, in the folder of the hologram, which is where a
        background is likely to be.'''
        if self.normalizer.source:
            return self.normalizer.source
        return str(Path(self._datafile).parent) if self._datafile else ''

    @pyqtSlot()
    def readBackground(self, filename: str | None = None) -> None:
        if not filename:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self, 'Open Background',
                              self._background_start(),
                              'Images (*.png *.tif *.tiff)')
        if filename:
            self.setBackground(filename)

    def _useBackground(self, background: str | float) -> str | None:
        '''Normalize by a background; return why it could not, if it could
        not, and keep the previous background.'''
        try:
            self.normalizer = Normalizer.read(background, dark=self.dark)
        except (OSError, ValueError) as error:
            logger.error(f'Could not set the background: {error}')
            self._refused = str(error)
            self._showBackground(self.normalizer)
            return self._refused
        self._refused = ''
        self._showBackground(self.normalizer)
        if self._raw is not None:
            self.data = self._raw
        return None

    def setDark(self, dark: float) -> None:
        '''Subtract a dark count (black level) before normalizing.

        Parameters
        ----------
        dark : float
            Dark count in gray levels, finite and not negative.

        Raises
        ------
        ValueError
            If the dark count is not finite and non-negative.
        '''
        self.normalizer.dark = dark        # validates it
        self.dark = self.normalizer.dark
        self._showBackground(self.normalizer)
        if self._raw is not None:
            self.data = self._raw

    def setBackground(self, background: str | float) -> bool:
        '''Normalize holograms by a background.

        Parameters
        ----------
        background : str or float
            A number, or the name of an image file, as for
            :meth:`~pylorenzmie.utilities.normalizer.Normalizer.read`.

        Returns
        -------
        bool
            True if the background was set, False if it could not be read,
            which leaves the previous background and reports the reason.
        '''
        error = self._useBackground(background)
        if error:
            self.statusBar().showMessage(error, 8000)
        return error is None

    @pyqtSlot()
    def saveParameters(self, filename: str | None = None) -> None:
        if not filename:
            get = QFileDialog.getSaveFileName
            filename, _ = get(self, 'Save Parameters', '', 'JSON (*.json)')
        if not filename:
            return
        properties = self.session.properties
        with open(filename, 'w') as f:
            json.dump(properties, f, indent=4, sort_keys=True)

    def _updateProfile(self) -> None:
        x_p = self.session.get('x_p')
        y_p = self.session.get('y_p')
        self.profileWidget.data = Azimuthal.std(self._data, (x_p, y_p))

    @pyqtSlot(float, float)
    def _handleROIChanged(self, x_p: float, y_p: float) -> None:
        if self._data is None:
            return
        self._moving = True               # the ROI moved, so do not move it
        try:
            self.session.update(dict(x_p=x_p, y_p=y_p))
        finally:
            self._moving = False
        self._updateProfile()
        self._showCrop()

    @pyqtSlot(int)
    def _handleRadiusChanged(self, radius: int) -> None:
        if self._data is None:
            return
        self.profileWidget.radius = radius
        self.preview.setRadius(radius)
        self._showCrop()

    @pyqtSlot(list)
    def _onParametersChanged(self, names: list[str]) -> None:
        '''Show parameters that changed in the ROI, the profile and the fit.'''
        if self._moving:
            return
        for name in ('x_p', 'y_p'):
            if name in names:
                value = self.session.get(name)
                if abs(getattr(self.imageWidget, name) - value) > 1e-6:
                    setattr(self.imageWidget, name, value)
        # the preview shows the other parameters when the session announces
        # them

    def _showCrop(self) -> None:
        '''Show the region of interest and ask for the prediction for it.'''
        data, rect, coordinates = self.crop()
        self.fitWidget.setData(data, rect, coordinates)
        self.preview.setCoordinates(coordinates)

    @pyqtSlot(object, object, float)
    def _showPreview(self, hologram, profile, noise: float) -> None:
        self.fitWidget.showModel(hologram, noise)
        self.profileWidget.showTheory(profile)

    def crop(self) -> tuple[Image, QRectF, Coordinates]:
        get = self.imageWidget.roi.getArraySlice
        (sy, sx), _ = get(self._data, self.imageWidget.image)
        # the rectangle of the pixels that were cut out: the ROI itself
        # extends past the image when the particle is near an edge
        rect = QRectF(sx.start, sy.start,
                      sx.stop - sx.start, sy.stop - sy.start)
        return (self._data[sy, sx], rect, self.coordinates[:, sy, sx])

    def _busy(self) -> bool:
        '''True, with a message, if a task is running.'''
        if self.tasks.busy:
            self.statusBar().showMessage(
                f'Busy with the task "{self.tasks.name}": wait, or press Esc',
                3000)
        return self.tasks.busy

    def _snapshot(self) -> Snapshot:
        '''A copy of what a task needs: the session and the region.'''
        data, _, coordinates = self.crop()
        optimizer = self.fitWidget.optimizer
        return Snapshot.take(self.session.cls, self.session.properties,
                             data, coordinates, self.session.fixed,
                             optimizer.settings, optimizer.fraction)

    @pyqtSlot()
    def estimate(self) -> None:
        '''Estimate the height, radius and index of the particle.

        The search runs in a thread, on a copy of the session.
        '''
        if self._data is None or self._busy():
            return
        props = self.session.properties
        # Save only the scalar physical parameters for undo; x_p/y_p are
        # excluded because the user may have moved the ROI since estimating.
        self._pre_estimate = {k: props[k] for k in ('z_p', 'a_p', 'n_p')
                              if k in props}
        self.actionUndoEstimate.setEnabled(True)
        self._row = self._particleRow()
        self.tasks.start('estimate', run_estimate, self._snapshot())

    def _onEstimated(self, result: pd.Series) -> None:
        fixed = self.session.fixed
        updates = {k: float(result[k]) for k in ('z_p', 'a_p', 'n_p')
                   if k in result and k not in fixed
                   and result[k] is not None
                   and np.isfinite(result[k])}
        if updates:
            self.session.update(updates)
        if 0 <= self._row < self.particles.count:
            self.particles.setEstimate(self._row, self.session.properties)
        logger.info(f'Estimate: {result}')
        self.statusBar().showMessage('Estimation complete', 2000)

    @pyqtSlot()
    def undoEstimate(self) -> None:
        if self._pre_estimate is None:
            return
        self.session.update(self._pre_estimate)
        self._pre_estimate = None
        self.actionUndoEstimate.setEnabled(False)

    @pyqtSlot(bool)
    def setRobust(self, state: bool) -> None:
        optimizer = self.fitWidget.optimizer
        optimizer.robust = state
        self.optimizerWidget.settings = optimizer.settings

    def closeEvent(self, event: QCloseEvent) -> None:
        self.tasks.shutdown()
        super().closeEvent(event)

    @pyqtSlot()
    def optimize(self) -> None:
        '''Fit the model to the region of interest.

        The fit runs in a thread, on a copy of the session; the result
        arrives when it is done, and Cancel (Esc) stops it.
        '''
        if self._data is None or self._busy():
            return
        self._row = self._particleRow()
        self.tasks.start('optimize', run_fit, self._snapshot())

    def _onOptimized(self, fit: FitResult) -> None:
        self.session.update(fit.properties)
        self._metadata = fit.metadata
        if 0 <= self._row < self.particles.count:
            self.particles.setResult(self._row, self.session.properties,
                                     fit.result)
            self.particles.setCurrent(self._row)
        logger.info(f'Optimization complete\n{fit.result}')
        message = 'Optimization complete'
        if 'redchi' in fit.result:
            message += f': reduced \u03c7\u00b2 = {fit.result["redchi"]:.2f}'
        self.statusBar().showMessage(message, 8000)

    @pyqtSlot(str)
    def _onTaskStarted(self, name: str) -> None:
        if name == 'localize':
            # the window stays usable: a search changes nothing
            self.actionFindParticles.setEnabled(False)
            self.actionCancel.setEnabled(True)
            self.statusBar().showMessage('Looking for particles ... '
                                         '(Esc cancels)')
            return
        self._setUIBusy(True)
        verb = 'Estimating' if name == 'estimate' else 'Optimizing'
        self.statusBar().showMessage(f'{verb} ... (Esc cancels)')

    def _taskEnded(self, name: str) -> None:
        if name == 'localize':
            self.actionFindParticles.setEnabled(True)
            self.actionCancel.setEnabled(False)
        else:
            self._setUIBusy(False)

    @pyqtSlot(str, object)
    def _onTaskFinished(self, name: str, result: object) -> None:
        self._taskEnded(name)
        if name == 'localize':
            self._onLocalized(result)
        elif name == 'estimate':
            self._onEstimated(result)
        else:
            self._onOptimized(result)

    @pyqtSlot(str, str)
    def _onTaskFailed(self, name: str, message: str) -> None:
        self._taskEnded(name)
        what = {'localize': 'Search for particles', 'estimate': 'Estimation',
                'optimize': 'Optimization'}[name]
        logger.error(f'{what} failed: {message}')
        self.statusBar().showMessage(f'{what} failed: {message}', 8000)

    @pyqtSlot(str)
    def _onTaskCancelled(self, name: str) -> None:
        self._taskEnded(name)
        if name == 'estimate':            # nothing changed: nothing to undo
            self._pre_estimate = None
            self.actionUndoEstimate.setEnabled(False)
        self.statusBar().showMessage('Cancelled', 3000)
        if name == 'localize' and self._search_again:
            self._search_again = False
            self.findParticles()          # for the image that replaced it

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------

    def _output_path(self) -> str:
        timestamp = datetime.now().strftime('%m_%d_%Y-%H_%M_%S')
        return str(Path('~/data/lmtool').expanduser() /
                   f'results_{timestamp}.h5')

    @pyqtSlot()
    def saveResults(self, filename: str | None = None) -> None:
        '''Save the fits of the particles that are checked, one row each.

        The format follows the suffix of the file name: HDF5 (or JSON, if
        PyTables is not installed), JSON or CSV.  Without a file name the
        file goes to ``~/data/lmtool``, with a time stamp in its name.
        '''
        results = self.particles.results()
        if len(results) == 0:
            self.statusBar().showMessage(
                'There is nothing to save: optimize a particle first', 8000)
            return
        metadata = (self._metadata if self._metadata is not None
                    else pd.Series(dtype=object)).copy()
        metadata['datafile'] = self._datafile
        try:
            saved = save_results(results, metadata,
                                 filename or self._output_path())
        except Exception as error:
            logger.exception('Could not save the results')
            self.statusBar().showMessage(
                f'Could not save the results: {error}', 8000)
            return
        self.statusBar().showMessage(
            f'Saved {len(results)} particle(s) to {saved}', 8000)

    @pyqtSlot()
    def saveResultsAs(self) -> None:
        get = QFileDialog.getSaveFileName
        filename, _ = get(self, 'Save Results', self._output_path(),
                          'HDF5 (*.h5);;JSON (*.json);;CSV (*.csv)')
        if filename:
            self.saveResults(filename)


def argument_parser() -> argparse.ArgumentParser:
    '''The command line of lmtool.

    Returns
    -------
    argparse.ArgumentParser
        Parser of the file to open and of the options ``--background``,
        ``--distorted`` and ``--backend``.
    '''
    default_file = str(_DIR.parent / 'docs' / 'tutorials' / 'crop.png')

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default=default_file,
                        nargs='?', action='store')
    parser.add_argument('-b', '--background', dest='background',
                        default=None, action='store',
                        help='background: image path or intensity value')
    parser.add_argument('-D', '--dark', type=float, default=0.,
                        help='dark count (black level) of the camera, in '
                        'gray levels, subtracted before normalizing')
    parser.add_argument('-d', '--distorted', action='store_true',
                        help='model the aberrations as geometric '
                        'distortions rather than phase factors '
                        '(numpy and numba backends)')
    parser.add_argument('-B', '--backend', choices=BACKENDS, default=None,
                        help='calculator of the scattering model '
                        '(default: PYLORENZMIE_BACKEND, else numba '
                        'if installed, else numpy)')
    return parser


def lmtool() -> None:
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
    from pyqtgraph.Qt.QtWidgets import QApplication
    import sys

    logging.basicConfig(level=logging.WARNING,
                        format='%(name)s: %(levelname)s: %(message)s')

    parser = argument_parser()
    args, unparsed = parser.parse_known_args()
    qt_args = sys.argv[:1] + unparsed
    try:
        model = model_class(args.backend, args.distorted)
    except (ValueError, ImportError) as error:
        parser.error(str(error))

    app = QApplication(qt_args)
    tool = LMTool(ALMWidget.with_model(model), args.filename,
                  args.background, args.dark)
    tool.show()
    sys.exit(app.exec())


if __name__ == '__main__':  # pragma: no cover
    lmtool()
