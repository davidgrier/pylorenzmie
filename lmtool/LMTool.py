from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from qtpy import uic
from qtpy.QtCore import Slot, QRectF
from qtpy.QtGui import QAction, QCloseEvent
from qtpy.QtWidgets import (QAbstractItemView, QFileDialog,
                            QHeaderView, QLabel, QMainWindow, QMenu,
                            QProgressBar, QTableView, QTabWidget,
                            QVBoxLayout)

from pylorenzmie.analysis import catch_available
from pylorenzmie.lib import (Azimuthal, LMObject)
from pylorenzmie.lib.lmtypes import Coordinates, Image
from pylorenzmie.lmtool.background import Background
from pylorenzmie.lmtool.FitWidget import FitWidget
from pylorenzmie.lmtool.ImageWidget import ImageWidget
from pylorenzmie.lmtool.LMWidget import LMWidget
from pylorenzmie.lmtool.OptimizerWidget import OptimizerWidget
from pylorenzmie.lmtool.ParticleTable import ParticleTable
from pylorenzmie.lmtool.Preview import Preview
from pylorenzmie.lmtool.ProfileWidget import ProfileWidget
from pylorenzmie.lmtool.required import required
from pylorenzmie.lmtool.results import save_results
from pylorenzmie.lmtool.tasks import (FitResult, LocalizeRequest, Snapshot,
                                       Task, TaskRunner, run_estimate,
                                       run_fit, run_localize)
from pylorenzmie.utilities import Normalizer, estimate_noise, imread

_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)

#: Where results go when the user has not chosen a file.
RESULTS_DIRECTORY = Path('~/data/lmtool')


@dataclass(frozen=True)
class LoadedImage:
    '''The hologram that the window shows, in the forms that it needs.

    Parameters
    ----------
    raw : numpy.ndarray
        The hologram as it was read.
    data : numpy.ndarray
        The hologram divided by its background.
    coordinates : numpy.ndarray
        Pixel coordinates of the hologram, shape (2, ny, nx).
    saturated : numpy.ndarray of bool or None
        Saturated pixels, if the saturation level is known.
    '''

    raw: Image
    data: NDArray[np.floating]
    coordinates: Coordinates
    saturated: NDArray[np.bool_] | None


@dataclass(frozen=True)
class _TaskInfo:
    '''What the window says about a task, and whether it can be used while
    the task runs.'''

    running: str          # 'Estimating' in 'Estimating ... (Esc cancels)'
    failed: str           # 'Estimation' in 'Estimation failed: ...'
    modal: bool = True    # a search changes nothing: the window stays usable


_TASKS = {Task.LOCALIZE: _TaskInfo('Looking for particles',
                                   'Search for particles',
                                   modal=False),
          Task.ESTIMATE: _TaskInfo('Estimating', 'Estimation'),
          Task.OPTIMIZE: _TaskInfo('Optimizing', 'Optimization')}


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
    saturation : float, optional
        Gray level at and above which a pixel is saturated, so that fits
        leave it out.  Default: ``None``, the largest value that the image
        can hold (255 for 8 bits).
    '''

    uiFile = 'LMTool.ui'

    # what the layout in LMTool.ui makes
    controlsLayout: QVBoxLayout
    tabs: QTabWidget
    imageWidget: ImageWidget
    profileWidget: ProfileWidget
    fitWidget: FitWidget
    optimizerWidget: OptimizerWidget
    menuParticles: QMenu
    menuOptimization: QMenu
    actionOpen: QAction
    actionOpenBackground: QAction
    actionSaveParameters: QAction
    saveResult: QAction
    saveResultAs: QAction
    actionEstimate: QAction
    actionUndoEstimate: QAction
    actionEstimateNoise: QAction
    actionRobust: QAction
    actionFindParticles: QAction
    actionNextParticle: QAction
    actionOptimize: QAction
    actionCancel: QAction

    #: The panel with the parameters of the model, made by the class that the
    #: window was given and put in ``controlsLayout`` when the window is made.
    controls: LMWidget

    #: Number of candidate particles that Find Particles keeps, the
    #: strongest first.  A noisy image can have hundreds of candidates.
    MAX_CANDIDATES = 10

    #: How far the ROI can be from a particle, in pixels, and still be on it.
    PARTICLE_TOLERANCE = 10.

    def __init__(self,
                 controls: type[LMWidget],
                 filename: str | None = None,
                 background: str | float | None = None,
                 dark: float = 0.,
                 saturation: float | None = None) -> None:
        super().__init__()
        # The state comes first: loading the layout reads the attributes of
        # the window, and a property that needs the state would fail.
        #: What the holograms are divided by.
        self.background = Background(dark, saturation)
        self._image: LoadedImage | None = None
        self._datafile: str | None = None
        self._pre_estimate: dict[str, float] | None = None   # for undo
        self._metadata: pd.Series | None = None    # of the last fit
        self._row = -1            # particle that a task works on
        self._nfound = 0          # candidates that the last search found
        self._search_again = False    # a new image replaced the searched one
        self._moving = False      # the ROI follows the session
        self._syncing = False     # the session moves the ROI
        self.particles = ParticleTable()
        self.tasks = TaskRunner()
        self._handlers: dict[Task, Callable[[Any], None]] = {
            Task.LOCALIZE: self._onLocalized,
            Task.ESTIMATE: self._onEstimated,
            Task.OPTIMIZE: self._onOptimized}
        uic.loadUi(_DIR / self.uiFile, self)
        self._setupControls(controls())
        self._setupStatusBar()
        self._setupParticles()
        self._connectSignals()
        # fits discount what the model does not explain, unless the user
        # chooses otherwise in the settings
        self.setRobust()
        if background is not None:
            self.background.use(background)
            self._showBackground()
        self.readHologram(filename)

    def _setupControls(self, controls: LMWidget) -> None:
        '''Show the panel of parameters, and connect what depends on its
        session: the preview of the model and the settings of the fit.'''
        self.controlsLayout.addWidget(controls)
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
        view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        required(view.verticalHeader()).hide()
        header = required(view.horizontalHeader())
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

        # CATCH is optional: it needs PyTorch and its own package.
        self.actionCATCH = QAction('Use CATCH', self)
        self.actionCATCH.setCheckable(True)
        available = catch_available()
        self.actionCATCH.setEnabled(available)
        self.actionCATCH.setChecked(available)
        self.actionCATCH.setToolTip(
            'Find and estimate particles with the neural networks of CATCH'
            if available else
            'CATCH is not installed: pip install '
            'git+https://github.com/laltman2/CATCH.git')
        self.actionCATCH.setStatusTip(self.actionCATCH.toolTip())
        self.menuParticles.addSeparator()
        self.menuParticles.addAction(self.actionCATCH)

    def _say(self, message: str, timeout: int = 0) -> None:
        '''Show a message in the status bar for *timeout* ms (0: until the
        next message).'''
        required(self.statusBar()).showMessage(message, timeout)

    def _setupStatusBar(self) -> None:
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumWidth(150)
        self._progress.setVisible(False)
        required(self.statusBar()).addPermanentWidget(self._progress)
        # what the holograms are divided by, and why not what was asked
        # for: a message would be replaced by the next one
        self._background = QLabel()
        required(self.statusBar()).addPermanentWidget(self._background)
        self._showBackground()

    def _showBackground(self) -> None:
        '''Show what the holograms are divided by, and what is wrong.'''
        self._background.setText(self.background.describe())

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
        required(self.particleView.selectionModel()).currentRowChanged.connect(
            lambda current, _: self.particles.setCurrent(current.row()))
        self.actionEstimate.triggered.connect(self.estimate)
        self.actionUndoEstimate.triggered.connect(self.undoEstimate)
        self.actionSaveParameters.triggered.connect(self.saveParameters)
        self.saveResult.triggered.connect(self.saveResults)
        self.saveResultAs.triggered.connect(self.saveResultsAs)
        self.actionRobust.triggered.connect(self.setRobust)
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

    @property
    def normalizer(self) -> Normalizer:
        '''What the holograms are divided by.'''
        return self.background.normalizer

    @property
    def dark(self) -> float:
        '''The dark count that is subtracted before the division.'''
        return self.background.dark

    @property
    def raw(self) -> Image | None:
        '''The hologram as it was read, or None before one is.'''
        return None if self._image is None else self._image.raw

    @property
    def data(self) -> NDArray[np.floating] | None:
        '''The normalized hologram, or None before one is read.

        Setting it takes the raw hologram, normalizes it by the background
        and shows it.'''
        return None if self._image is None else self._image.data

    @data.setter
    def data(self, data: Image) -> None:
        normalized, description = self.background.normalize(data)
        self._background.setText(description)
        coordinates = LMObject.meshgrid(data.shape, flatten=False)
        saturated = self.background.saturated(data)
        self._image = LoadedImage(data, normalized, coordinates, saturated)
        self.controls.x_p.setRange((0, data.shape[1]-1))
        self.controls.y_p.setRange((0, data.shape[0]-1))
        self.imageWidget.data = normalized
        self._setNoise(normalized)
        self._updateProfile()
        self._showCrop()

    @property
    def coordinates(self) -> Coordinates | None:
        '''Pixel coordinates of the hologram, or None before one is read.'''
        return None if self._image is None else self._image.coordinates

    @Slot()
    def readHologram(self, filename: str | None = None) -> None:
        '''Read a hologram from a file, which is asked for if not given.'''
        if not filename:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self,
                              'Open Hologram',
                              self._datafile or '',
                              'Images (*.png *.tif *.tiff)')
        if not filename:
            return
        data = imread(filename)
        if data is None:
            logger.error(f'Could not read image: {filename!r}')
            self._say(f'Could not read: {filename}', 5000)
            return
        self._datafile = filename
        self.particles.clear()
        # in the type of the file, which tells where it saturates
        self.data = data
        self.findParticles()

    def _setNoise(self, image: NDArray[np.floating]) -> bool:
        '''Set the noise of the model to the estimate for ``image``.

        Returns False, and leaves the noise as it is, if the image has too
        few finite pixels for an estimate.
        '''
        try:
            estimate = estimate_noise(image)
        except ValueError as error:
            logger.warning(f'The noise could not be estimated: {error}')
            return False
        control = self.controls.control('noise')
        low, high = control.range() if control else (0., np.inf)
        noise = float(np.clip(estimate, low, high))
        self.session.update(dict(noise=noise))
        return True

    @Slot()
    def estimateNoise(self) -> None:
        '''Estimate the noise from the region of interest.

        The estimate is the width of the pixel-to-pixel variation, which
        does not include the fringes or a smooth background.  Every
        residual in the Fit tab is divided by it, so the reduced chi-squared
        of a fit is meaningful only when it is right.
        '''
        if self.data is None:
            return
        data, _, _ = self.crop()
        if self._setNoise(data):
            self._say(f'Noise: {self.session.get("noise"):.3f}', 5000)
        else:
            self._say('Too few pixels of the region can be used to '
                      'estimate the noise', 8000)

    @Slot()
    def findParticles(self) -> None:
        '''Look for particles in the image, in the background.

        The strongest candidates go in the Particles tab, and the ROI moves to
        the first one, with the size of the region that encloses its fringes.
        :meth:`nextParticle` selects the others.  Particles that were added
        by hand stay.
        '''
        if self.data is None:
            return
        if self.tasks.busy:
            # a search of an older image is replaced by one of this image
            if self.tasks.name == Task.LOCALIZE:
                self._search_again = True
                self.tasks.cancel()
            return
        self.tasks.start(
            Task.LOCALIZE,
            run_localize,
            LocalizeRequest(
                self.data,
                self.MAX_CANDIDATES,
                self.useCATCH,
                self._instrument(),
                cast(LoadedImage, self._image).saturated,
            ),
        )

    def _onLocalized(self, found: pd.DataFrame) -> None:
        self._nfound = found.attrs.get('nfound', len(found))
        self.particles.setFound(found)
        if self.particles.count == 0:
            self._say(
                'No particles found: move the ROI by hand and add it '
                '(Ctrl+Shift+A)', 8000)
        else:
            self.nextParticle()
        self._warn(found.attrs.get('warning'))

    def _warn(self, warning: str | None) -> None:
        '''Show a warning of a task, after any other message about it.'''
        if warning:
            logger.warning(warning)
            self._say(warning, 15000)

    @Slot()
    def nextParticle(self) -> None:
        '''Select the next particle in the Particles tab that is checked.'''
        if self.particles.count == 0:
            self.findParticles()
            return
        row = self.particles.next()
        if row is None:
            self._say('No particle is checked', 5000)
            return
        self.particles.setCurrent(row)

    @Slot()
    def addParticle(self) -> None:
        '''Add the region of interest to the particles.'''
        if self.data is None:
            return
        row = self.particles.add(self.session.get('x_p'),
                                 self.session.get('y_p'),
                                 self.imageWidget.radius)
        self.particles.setCurrent(row)

    @Slot()
    def removeParticle(self) -> None:
        '''Remove the particle that is selected.'''
        self.particles.remove(self.particles.current)

    @Slot(int)
    def _onCurrentParticle(self, row: int) -> None:
        '''Move the ROI to the particle that was selected.'''
        if row < 0:
            return
        selection = required(self.particleView.selectionModel())
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
        self._say(message, 8000)

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

    @Slot()
    def readBackground(self, filename: str | None = None) -> None:
        '''Read a background image from a file, which is asked for if
        not given.'''
        if not filename:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self,
                              'Open Background',
                              self.background.start_directory(self._datafile),
                              'Images (*.png *.tif *.tiff)')
        if filename:
            self.setBackground(filename)

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
        self.background.set_dark(dark)
        self._showBackground()
        if self._image is not None:
            self.data = self._image.raw

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
        error = self.background.use(background)
        self._showBackground()
        if error:
            self._say(error, 8000)
        elif self._image is not None:
            self.data = self._image.raw
        return error is None

    @Slot()
    def saveParameters(self, filename: str | None = None) -> None:
        '''Save the parameters of the model as JSON, in a file that is
        asked for if not given.'''
        if not filename:
            get = QFileDialog.getSaveFileName
            filename, _ = get(self, 'Save Parameters', '', 'JSON (*.json)')
        if not filename:
            return
        try:
            # the text first: a value that is not JSON must not leave a
            # truncated file
            text = json.dumps(self.session.properties,
                              indent=4,
                              sort_keys=True)
            Path(filename).write_text(text)
        except (OSError, TypeError, ValueError) as error:
            logger.exception('Could not save the parameters')
            self._say(f'Could not save the parameters: {error}', 8000)
            return
        self._say(f'Saved the parameters to {filename}', 5000)

    def _updateProfile(self) -> None:
        x_p = self.session.get('x_p')
        y_p = self.session.get('y_p')
        self.profileWidget.data = Azimuthal.std(self.data, (x_p, y_p))

    @Slot(float, float)
    def _handleROIChanged(self, x_p: float, y_p: float) -> None:
        if self.data is None:
            return
        if not self._syncing:             # else the session moved the ROI,
            self._moving = True           # and has the position already
            try:
                self.session.update(dict(x_p=x_p, y_p=y_p))
            finally:
                self._moving = False
        self._updateProfile()
        self._showCrop()

    @Slot(int)
    def _handleRadiusChanged(self, radius: int) -> None:
        if self.data is None:
            return
        self.profileWidget.radius = radius
        self.preview.setRadius(radius)
        self._showCrop()

    @Slot(list)
    def _onParametersChanged(self, names: list[str]) -> None:
        '''Show parameters that changed in the ROI, the profile and the fit.'''
        if self._moving:
            return
        # Moving the ROI to x_p announces its position, with the old y_p: the
        # session must not take that back while it moves the ROI to y_p.
        self._syncing = True
        try:
            if 'x_p' in names:
                x_p = self.session.get('x_p')
                if abs(self.imageWidget.x_p - x_p) > 1e-6:
                    self.imageWidget.x_p = x_p
            if 'y_p' in names:
                y_p = self.session.get('y_p')
                if abs(self.imageWidget.y_p - y_p) > 1e-6:
                    self.imageWidget.y_p = y_p
        finally:
            self._syncing = False
        # the preview shows the other parameters when the session announces
        # them

    def _showCrop(self) -> None:
        '''Show the region of interest and ask for the prediction for it.'''
        data, rect, coordinates = self.crop()
        self.fitWidget.setData(data, rect, coordinates)
        self.preview.setCoordinates(coordinates)

    @Slot(object, object, float)
    def _showPreview(self,
                     hologram: NDArray[np.floating],
                     profile: NDArray[np.floating],
                     noise: float) -> None:
        self.fitWidget.showModel(hologram, noise)
        self.profileWidget.showTheory(profile)

    def _roiSlices(self) -> tuple[slice, slice]:
        '''Rows and columns of the pixels inside the ROI.'''
        image = self._image
        if image is None:
            raise RuntimeError('there is no hologram')
        get = self.imageWidget.roi.getArraySlice
        found = get(image.data, self.imageWidget.image)
        if found is None:
            raise RuntimeError('the ROI is outside the image')
        slices, _ = cast(tuple[tuple[slice, slice], Any], found)
        return slices

    def crop(self) -> tuple[NDArray[np.floating], QRectF, Coordinates]:
        '''Return the pixels inside the ROI, the rectangle that they cover
        and their coordinates.'''
        sy, sx = self._roiSlices()
        image = cast(LoadedImage, self._image)
        # the rectangle of the pixels that were cut out: the ROI itself
        # extends past the image when the particle is near an edge
        rect = QRectF(sx.start,
                      sy.start,
                      sx.stop - sx.start,
                      sy.stop - sy.start)
        return (image.data[sy, sx], rect, image.coordinates[:, sy, sx])

    def _busy(self) -> bool:
        '''True, with a message, if a task is running.'''
        if self.tasks.busy:
            self._say(
                f'Busy with the task "{self.tasks.name}": wait, or press Esc',
                3000)
        return self.tasks.busy

    def _snapshot(self) -> Snapshot:
        '''A copy of what a task needs: the session and the region.'''
        data, _, coordinates = self.crop()
        saturated = cast(LoadedImage, self._image).saturated
        if saturated is not None:
            saturated = saturated[self._roiSlices()]
        optimizer = self.fitWidget.optimizer
        return Snapshot.take(self.session.cls,
                             self.session.properties,
                             data,
                             coordinates,
                             self.session.fixed,
                             optimizer.settings,
                             optimizer.fraction,
                             self.useCATCH,
                             saturated)

    @property
    def useCATCH(self) -> bool:
        '''True if CATCH, and not the circle transform and the search of
        differential evolution, finds and estimates the particles.'''
        return self.actionCATCH.isEnabled() and self.actionCATCH.isChecked()

    def _instrument(self) -> dict[str, float]:
        '''Wavelength, magnification and index of the medium of the model.'''
        return {name: self.session.get(name)
                for name in ('wavelength', 'magnification', 'n_m')}

    @Slot()
    def estimate(self) -> None:
        '''Estimate the height, radius and index of the particle.

        The search runs in a thread, on a copy of the session.
        '''
        if self.data is None or self._busy():
            return
        props = self.session.properties
        # Save only the scalar physical parameters for undo; x_p/y_p are
        # excluded because the user may have moved the ROI since estimating.
        self._pre_estimate = {k: props[k] for k in ('z_p', 'a_p', 'n_p')
                              if k in props}
        self.actionUndoEstimate.setEnabled(True)
        self._row = self._particleRow()
        self.tasks.start(Task.ESTIMATE, run_estimate, self._snapshot())

    def _onEstimated(self, result: pd.Series) -> None:
        fixed = self.session.fixed
        values = result.to_dict()
        updates = {k: float(values[k]) for k in ('z_p', 'a_p', 'n_p')
                   if k in values and k not in fixed
                   and values[k] is not None
                   and np.isfinite(result[k])}
        if updates:
            self.session.update(updates)
        if 0 <= self._row < self.particles.count:
            self.particles.setEstimate(self._row, self.session.properties)
        logger.info(f'Estimate: {result}')
        self._say('Estimation complete', 2000)
        self._warn(result.attrs.get('warning'))

    @Slot()
    def undoEstimate(self) -> None:
        '''Restore the parameters that the last estimate replaced.'''
        if self._pre_estimate is None:
            return
        self.session.update(self._pre_estimate)
        self._pre_estimate = None
        self.actionUndoEstimate.setEnabled(False)

    @Slot()
    def setRobust(self) -> None:
        '''Set up a robust fit, with a loss whose scale is matched to the
        residuals, and show the settings, which may then be changed.'''
        optimizer = self.fitWidget.optimizer
        optimizer.robust = True
        optimizer.settings['f_scale'] = 'auto'
        self.optimizerWidget.settings = optimizer.settings

    # Optional, and not a union with None, as in FitWidget.showEvent
    def closeEvent(self, event: Optional[QCloseEvent]) -> None:
        self.tasks.shutdown()
        super().closeEvent(event)

    @Slot()
    def optimize(self) -> None:
        '''Fit the model to the region of interest.

        The fit runs in a thread, on a copy of the session; the result
        arrives when it is done, and Cancel (Esc) stops it.
        '''
        if self.data is None or self._busy():
            return
        self._row = self._particleRow()
        self.tasks.start(Task.OPTIMIZE, run_fit, self._snapshot())

    def _onOptimized(self, fit: FitResult) -> None:
        self.session.update(fit.properties)
        self._metadata = fit.metadata
        if 0 <= self._row < self.particles.count:
            self.particles.setResult(self._row,
                                     self.session.properties,
                                     fit.result)
            self.particles.setCurrent(self._row)
        logger.info(f'Optimization complete\n{fit.result}')
        message = 'Optimization complete'
        if 'redchi' in fit.result:
            message += f': reduced \u03c7\u00b2 = {fit.result["redchi"]:.2f}'
        self._say(message, 8000)

    @Slot(str)
    def _onTaskStarted(self, name: str) -> None:
        info = _TASKS[Task(name)]
        if info.modal:
            self._setUIBusy(True)
        else:
            self.actionFindParticles.setEnabled(False)
            self.actionCancel.setEnabled(True)
        self._say(f'{info.running} ... (Esc cancels)')

    def _taskEnded(self, name: str) -> None:
        if _TASKS[Task(name)].modal:
            self._setUIBusy(False)
        else:
            self.actionFindParticles.setEnabled(True)
            self.actionCancel.setEnabled(False)

    @Slot(str, object)
    def _onTaskFinished(self, name: str, result: object) -> None:
        self._taskEnded(name)
        self._handlers[Task(name)](result)

    @Slot(str, str)
    def _onTaskFailed(self, name: str, message: str) -> None:
        self._taskEnded(name)
        what = _TASKS[Task(name)].failed
        logger.error(f'{what} failed: {message}')
        self._say(f'{what} failed: {message}', 8000)

    @Slot(str)
    def _onTaskCancelled(self, name: str) -> None:
        self._taskEnded(name)
        if name == Task.ESTIMATE:         # nothing changed: nothing to undo
            self._pre_estimate = None
            self.actionUndoEstimate.setEnabled(False)
        self._say('Cancelled', 3000)
        if name == Task.LOCALIZE and self._search_again:
            self._search_again = False
            self.findParticles()          # for the image that replaced it

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------

    def _output_path(self) -> str:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return str(RESULTS_DIRECTORY.expanduser() /
                   f'results_{timestamp}.h5')

    @Slot()
    def saveResults(self, filename: str | None = None) -> None:
        '''Save the fits of the particles that are checked, one row each.

        The format follows the suffix of the file name: HDF5 (or JSON, if
        PyTables is not installed), JSON or CSV.  Without a file name the
        file goes to ``~/data/lmtool``, with a time stamp in its name.
        '''
        results = self.particles.results()
        if len(results) == 0:
            self._say(
                'There is nothing to save: optimize a particle first', 8000)
            return
        metadata = (self._metadata if self._metadata is not None
                    else pd.Series(dtype=object)).copy()
        metadata['datafile'] = self._datafile
        try:
            saved = save_results(results,
                                 metadata,
                                 filename or self._output_path())
        except Exception as error:
            logger.exception('Could not save the results')
            self._say(
                f'Could not save the results: {error}', 8000)
            return
        self._say(
            f'Saved {len(results)} particle(s) to {saved}', 8000)

    @Slot()
    def saveResultsAs(self) -> None:
        '''Ask for a file and save the results of the fits in it.'''
        get = QFileDialog.getSaveFileName
        filename, _ = get(self,
                          'Save Results',
                          self._output_path(),
                          'HDF5 (*.h5);;JSON (*.json);;CSV (*.csv)')
        if filename:
            self.saveResults(filename)
