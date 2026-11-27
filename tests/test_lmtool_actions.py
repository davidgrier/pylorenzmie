'''Tests of the actions of lmtool that nothing else exercised, and of the
hook that keeps an unhandled exception from closing the program.

They need a Qt binding and run without a display.
'''
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    import qtpy     # chooses the Qt binding (QT_API) before pyqtgraph can
    from pyqtgraph import mkQApp
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (QFileDialog, QInputDialog, QMenu,
                                QMessageBox)
    from pylorenzmie.lmtool import launch
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
    from pylorenzmie.lmtool.LMTool import LMTool
    from tests.test_lmtool import IMAGE, ToolTestCase
    _qt = True
except Exception:       # no Qt binding, or no usable platform plugin
    _qt = False
    ToolTestCase = unittest.TestCase


def setUpModule():
    if _qt:
        mkQApp()


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestSaveParameters(ToolTestCase):

    def message(self, tool):
        return tool.statusBar().currentMessage()

    def test_the_parameters_are_saved_as_json(self):
        tool = self.tool()
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'parameters.json')
            tool.saveParameters(filename)
            saved = json.loads(Path(filename).read_text())
        self.assertEqual(saved['z_p'], tool.session.get('z_p'))
        self.assertEqual(set(saved), set(tool.session.properties))
        self.assertIn('Saved the parameters', self.message(tool))

    def test_the_file_is_asked_for_when_none_is_given(self):
        tool = self.tool()
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'chosen.json')
            with mock.patch('pylorenzmie.lmtool.LMTool.QFileDialog.'
                            'getSaveFileName',
                            return_value=(filename, '')) as dialog:
                tool.saveParameters()
            self.assertTrue(Path(filename).exists())
        dialog.assert_called_once()

    def test_cancelling_the_dialog_writes_nothing(self):
        tool = self.tool()
        with mock.patch('pylorenzmie.lmtool.LMTool.QFileDialog.'
                        'getSaveFileName',
                        return_value=('', '')):
            with mock.patch.object(Path, 'write_text') as write:
                tool.saveParameters()
        write.assert_not_called()

    def test_a_file_that_cannot_be_written_is_reported(self):
        '''An OSError in a slot used to close the program.'''
        tool = self.tool()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertLogs(level='ERROR'):
                tool.saveParameters(directory)       # a directory
                tool.saveParameters(str(Path(directory) / 'no' / 'x.json'))
        self.assertIn('Could not save the parameters', self.message(tool))

    def test_a_value_that_is_not_json_leaves_the_old_file_alone(self):
        tool = self.tool()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'parameters.json'
            path.write_text('precious')
            properties = mock.PropertyMock(return_value=dict(x=object()))
            session = type(tool.session)
            with mock.patch.object(session, 'properties', properties):
                with self.assertLogs(level='ERROR'):
                    tool.saveParameters(str(path))
            self.assertEqual(path.read_text(), 'precious')
        self.assertIn('Could not save the parameters', self.message(tool))


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestUndoEstimate(ToolTestCase):

    @staticmethod
    def estimator(**values):
        def estimate(snapshot, cancel):
            return pd.Series(values)
        return mock.patch('pylorenzmie.lmtool.LMTool.run_estimate', estimate)

    def test_undo_restores_the_parameters_that_the_estimate_replaced(self):
        tool = self.tool()
        before = {name: tool.session.get(name)
                  for name in ('z_p', 'a_p', 'n_p')}
        self.assertFalse(tool.actionUndoEstimate.isEnabled())
        with self.estimator(z_p=312., a_p=1.23, n_p=1.47):
            tool.estimate()
            self.wait(tool)
        self.assertEqual(tool.session.get('z_p'), 312.)
        self.assertEqual(tool.session.get('a_p'), 1.23)
        self.assertTrue(tool.actionUndoEstimate.isEnabled())
        tool.undoEstimate()
        for name, value in before.items():
            self.assertEqual(tool.session.get(name), value, name)
        self.assertFalse(tool.actionUndoEstimate.isEnabled())

    def test_undo_does_not_move_the_particle_the_user_moved(self):
        tool = self.tool()
        with self.estimator(z_p=312., a_p=1.23, n_p=1.47):
            tool.estimate()
            self.wait(tool)
        tool.session.update(dict(x_p=111., y_p=222.))
        tool.undoEstimate()
        self.assertEqual(tool.session.get('x_p'), 111.)
        self.assertEqual(tool.session.get('y_p'), 222.)

    def test_undo_twice_changes_nothing_the_second_time(self):
        tool = self.tool()
        with self.estimator(z_p=312., a_p=1.23, n_p=1.47):
            tool.estimate()
            self.wait(tool)
        tool.undoEstimate()
        restored = tool.session.properties
        tool.session.update(dict(z_p=77.))
        tool.undoEstimate()
        self.assertEqual(tool.session.get('z_p'), 77.)
        self.assertNotEqual(tool.session.properties, restored)

    def test_undo_without_an_estimate_does_nothing(self):
        tool = self.tool()
        before = tool.session.properties
        tool.undoEstimate()
        self.assertEqual(tool.session.properties, before)

    def test_the_action_undoes_the_estimate(self):
        tool = self.tool()
        before = tool.session.get('z_p')
        with self.estimator(z_p=312., a_p=1.23, n_p=1.47):
            tool.estimate()
            self.wait(tool)
        tool.actionUndoEstimate.trigger()
        self.assertEqual(tool.session.get('z_p'), before)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestPosition(ToolTestCase):
    '''The ROI and the session agree on the position of the particle.'''

    def test_both_coordinates_of_an_update_reach_the_roi(self):
        tool = self.tool()
        tool.session.update(dict(x_p=111., y_p=222.))
        self.assertEqual(tool.session.get('x_p'), 111.)
        self.assertEqual(tool.session.get('y_p'), 222.)
        self.assertAlmostEqual(tool.imageWidget.x_p, 111.)
        self.assertAlmostEqual(tool.imageWidget.y_p, 222.)

    def test_a_fit_keeps_the_position_that_it_found(self):
        '''The y_p of a fit was lost whenever x_p changed too.'''
        tool = self.tool()
        tool.session.update(dict(x_p=190., y_p=200.))
        tool.optimize()
        self.wait(tool)
        fitted = tool.particles.results().iloc[0]
        self.assertAlmostEqual(tool.session.get('x_p'), fitted['x_p'])
        self.assertAlmostEqual(tool.session.get('y_p'), fitted['y_p'])
        self.assertNotEqual(fitted['y_p'], 200.)

    def test_moving_the_roi_still_moves_the_session(self):
        tool = self.tool()
        tool.imageWidget.x_p, tool.imageWidget.y_p = 150., 160.
        self.assertAlmostEqual(tool.session.get('x_p'), 150., delta=0.1)
        self.assertAlmostEqual(tool.session.get('y_p'), 160., delta=0.1)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestStartingOver(ToolTestCase):
    '''A change of the optics or the background starts over.'''

    def fitted(self):
        '''A window with a fit that is not saved.'''
        tool = self.tool()
        row = tool.particles.add(195., 193., 100)
        result = pd.Series(dict(a_p=0.75, redchi=1., success=True))
        tool.particles.setResult(row, tool.session.properties, result)
        self.assertEqual(tool.particles.unsaved(), 1)
        return tool

    def edit(self, tool, name, value):
        '''Change a parameter with its control, as the user does.'''
        getattr(tool.controls, name).setValue(value)

    def answer(self, discard):
        '''Answer the question about unsaved fits, if it is asked.'''
        button = QMessageBox.StandardButton
        chosen = button.Discard if discard else button.Cancel
        patcher = mock.patch.object(QMessageBox,
                                    'question',
                                    return_value=chosen)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def searches(self, tool):
        patcher = mock.patch.object(tool, 'findParticles')
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_new_optics_clear_the_particles_and_search_again(self):
        for name, value in (('wavelength', 0.6), ('magnification', 0.1),
                            ('n_m', 1.4)):
            with self.subTest(name):
                tool = self.tool()
                tool.particles.add(100., 100., 50)
                search = self.searches(tool)
                self.edit(tool, name, value)
                search.assert_not_called()           # it waits
                tool._applyOptics()
                search.assert_called_once_with()
                self.assertEqual(tool.particles.count, 0)

    def test_the_search_starts_when_the_value_is_left_alone(self):
        tool = self.tool()
        search = self.searches(tool)
        for value in (0.08, 0.09, 0.1):
            self.edit(tool, 'magnification', value)
        end = time.time() + 5.
        while not search.called and time.time() < end:
            mkQApp().processEvents()
            time.sleep(0.01)
        search.assert_called_once_with()

    def test_it_waits_while_the_slider_is_held(self):
        tool = self.tool()
        search = self.searches(tool)
        slider = tool.controls.magnification.slider
        slider.setSliderDown(True)
        self.edit(tool, 'magnification', 0.1)
        tool._applyOptics()
        search.assert_not_called()
        slider.setSliderDown(False)
        tool._applyOptics()
        search.assert_called_once_with()

    def test_other_changes_do_not_start_over(self):
        tool = self.tool()
        search = self.searches(tool)
        self.edit(tool, 'a_p', 1.2)                     # not the optics
        tool.session.update(dict(magnification=0.1))    # not the user
        tool._applyOptics()
        self.edit(tool, 'magnification', 0.2)           # and back again
        self.edit(tool, 'magnification', 0.1)
        tool._applyOptics()
        self.edit(tool, 'n_m', tool.session.get('n_m') + 3e-5)   # unseen
        tool._applyOptics()
        search.assert_not_called()

    def test_unsaved_fits_are_discarded_if_the_user_agrees(self):
        tool = self.fitted()
        question = self.answer(discard=True)
        search = self.searches(tool)
        self.edit(tool, 'magnification', 0.1)
        tool._applyOptics()
        question.assert_called_once()
        search.assert_called_once_with()
        self.assertEqual(tool.particles.count, 0)

    def test_otherwise_the_fits_and_the_old_value_stay(self):
        tool = self.fitted()
        before = tool.session.get('magnification')
        count = tool.particles.count
        question = self.answer(discard=False)
        search = self.searches(tool)
        self.edit(tool, 'magnification', 0.1)
        tool._applyOptics()
        self.assertAlmostEqual(tool.session.get('magnification'), before)
        self.assertAlmostEqual(tool.controls.magnification.value(),
                               before,
                               places=3)
        self.assertEqual(tool.particles.count, count)
        tool._applyOptics()                   # the return is not a change
        question.assert_called_once()
        search.assert_not_called()

    def test_saved_fits_are_not_asked_about(self):
        tool = self.fitted()
        with tempfile.TemporaryDirectory() as directory:
            tool.saveResults(str(Path(directory) / 'fits.json'))
        self.assertEqual(tool.particles.unsaved(), 0)
        question = self.answer(discard=False)
        search = self.searches(tool)
        self.edit(tool, 'magnification', 0.1)
        tool._applyOptics()
        question.assert_not_called()
        search.assert_called_once_with()

    def test_fits_that_are_not_checked_are_not_asked_about(self):
        tool = self.fitted()
        row = tool.particles.count - 1
        index = tool.particles.index(row, 0)
        tool.particles.setData(index,
                               Qt.CheckState.Unchecked,
                               Qt.ItemDataRole.CheckStateRole)
        self.assertEqual(tool.particles.unsaved(), 0)

    def test_a_fit_asked_for_at_once_uses_the_new_optics_first(self):
        tool = self.tool()
        search = self.searches(tool)
        self.edit(tool, 'magnification', 0.1)
        with mock.patch.object(tool.tasks, 'start') as start:
            tool.optimize()
        search.assert_called_once_with()
        start.assert_called_once()

    def test_a_new_background_starts_over_if_the_user_agrees(self):
        for discard in (False, True):
            with self.subTest(discard=discard):
                tool = self.fitted()
                count = tool.particles.count
                self.answer(discard)
                search = self.searches(tool)
                with mock.patch.object(QFileDialog,
                                       'getOpenFileName',
                                       return_value=(str(IMAGE), '')):
                    tool.readBackground()
                self.assertEqual(search.called, discard)
                self.assertEqual(tool.particles.count,
                                 0 if discard else count)

    def test_a_running_fit_is_cancelled_and_forgotten(self):
        tool = self.tool()
        started = self.slowly('fit')
        tool.optimize()
        self.assertTrue(started.wait(10))
        with mock.patch.object(tool, '_onOptimized') as record:
            tool.setBackground(100)
            self.wait(tool)
        record.assert_not_called()
        self.assertFalse(tool.tasks.busy)
        self.assertEqual(tool.particles.unsaved(), 0)
        self.assertFalse(tool._search_again)

    def test_a_new_image_asks_about_unsaved_fits(self):
        for discard in (False, True):
            with self.subTest(discard=discard):
                tool = self.fitted()
                count = tool.particles.count
                question = self.answer(discard)
                search = self.searches(tool)
                tool.readHologram(str(IMAGE))
                question.assert_called_once()
                self.assertEqual(search.called, discard)
                self.assertEqual(tool.particles.count,
                                 0 if discard else count)

    def test_a_file_that_cannot_be_read_discards_nothing(self):
        tool = self.fitted()
        question = self.answer(discard=True)
        with self.assertLogs(level='ERROR'):
            tool.readHologram('no-such-image.png')
        question.assert_not_called()
        self.assertEqual(tool.particles.unsaved(), 1)

    def test_a_new_image_cancels_a_running_fit(self):
        tool = self.tool()
        started = self.slowly('fit')
        tool.optimize()
        self.assertTrue(started.wait(10))
        with mock.patch.object(tool, '_onOptimized') as record:
            tool.readHologram(str(IMAGE))
            self.wait(tool)
        record.assert_not_called()
        self.assertFalse(tool.tasks.busy)
        self.assertEqual(tool.particles.unsaved(), 0)

    def test_a_new_image_uses_optics_that_were_just_changed(self):
        tool = self.tool()
        self.edit(tool, 'magnification', 0.1)
        search = self.searches(tool)
        tool.readHologram(str(IMAGE))
        search.assert_called_once_with()
        self.assertFalse(tool._settle.isActive())
        self.assertEqual(tool._edited, {})

    def test_the_optics_may_be_typed_while_they_are_fixed(self):
        tool = self.tool()
        for name in tool.OPTICS:
            with self.subTest(name):
                control = getattr(tool.controls, name)
                self.assertTrue(control.fixed())
                self.assertTrue(control.spinbox.isEnabled())
                self.assertTrue(control.slider.isEnabled())
        self.edit(tool, 'magnification', 0.1)
        self.assertIn('magnification', tool.session.fixed)

    def dialog(self, dark, accepted=True):
        '''Answer the dialog that asks for the dark count.'''
        patcher = mock.patch.object(QInputDialog,
                                    'getDouble',
                                    return_value=(dark, accepted))
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_the_file_menu_sets_the_dark_count(self):
        tool = self.tool(background=100)
        menu = tool.findChild(QMenu, 'menuFile')
        self.assertIn(tool.actionSetDark, menu.actions())
        ask = self.dialog(22.)
        search = self.searches(tool)
        tool.actionSetDark.trigger()
        self.assertEqual(ask.call_args.args[3], 0.)     # the value so far
        self.assertEqual(tool.dark, 22.)
        np.testing.assert_allclose(tool.data, (tool.raw - 22.) / 78.)
        search.assert_called_once_with()

    def test_a_dark_count_that_does_not_change_discards_nothing(self):
        tool = self.fitted()
        tool.setDark(22.345)           # more places than the dialog shows
        tool.particles.add(195., 193., 100)
        count = tool.particles.count
        question = self.answer(discard=True)
        search = self.searches(tool)
        for dark, accepted in ((50., False), (22.34, True), (22.345, True)):
            with self.subTest(dark=dark, accepted=accepted):
                self.dialog(round(dark, 2), accepted)
                tool.readDark()
                self.assertEqual(tool.dark, 22.345)
        question.assert_not_called()
        search.assert_not_called()
        self.assertEqual(tool.particles.count, count)

    def test_a_dark_count_asks_about_unsaved_fits(self):
        for discard in (False, True):
            with self.subTest(discard=discard):
                tool = self.fitted()
                question = self.answer(discard)
                search = self.searches(tool)
                self.dialog(10.)
                tool.readDark()
                question.assert_called_once()
                self.assertEqual(tool.dark, 10. if discard else 0.)
                self.assertEqual(search.called, discard)
                self.assertEqual(tool.particles.unsaved(),
                                 0 if discard else 1)

    def test_a_new_dark_count_starts_over(self):
        tool = self.tool(background=100)
        tool.particles.add(100., 100., 50)
        search = self.searches(tool)
        tool.setDark(10.)
        search.assert_called_once_with()
        self.assertEqual(tool.particles.count, 0)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestRobust(ToolTestCase):

    def assertRobust(self, tool):
        optimizer = tool.fitWidget.optimizer
        for settings in (optimizer.settings, tool.optimizerWidget.settings):
            self.assertEqual((settings['method'], settings['loss'],
                              settings['f_scale']),
                             ('trf', 'cauchy', 'auto'))

    def test_the_menu_entry_is_not_a_switch(self):
        self.assertFalse(self.tool().actionRobust.isCheckable())

    def test_fits_are_robust_at_the_start(self):
        self.assertRobust(self.tool())

    def test_the_settings_may_be_changed(self):
        tool = self.tool()
        optimizer = tool.fitWidget.optimizer
        panel = tool.optimizerWidget.params
        panel.child('loss').setValue('huber')
        panel.child('f_scale').setValue(2.)
        self.assertEqual((optimizer.settings['loss'],
                          optimizer.settings['f_scale']), ('huber', 2.))
        panel.child('method').setValue('lm')
        self.assertFalse(optimizer.robust)
        self.assertFalse(panel.child('f_scale').opts['enabled'])

    def test_the_menu_sets_up_a_robust_fit_again(self):
        tool = self.tool()
        panel = tool.optimizerWidget.params
        panel.child('f_scale').setValue(1.)
        panel.child('method').setValue('lm')
        self.assertIn(tool.actionRobust, tool.menuOptimization.actions())
        tool.actionRobust.trigger()
        self.assertRobust(tool)
        self.assertTrue(panel.child('f_scale').opts['enabled'])

    def test_a_fit_uses_the_settings_of_the_panel(self):
        tool = self.tool()
        tool.optimizerWidget.params.child('f_scale').setValue(4.)
        seen = {}

        def fit(snapshot, cancel):
            seen.update(snapshot.settings)
            raise RuntimeError('stop')

        with mock.patch('pylorenzmie.lmtool.LMTool.run_fit', fit):
            with self.assertLogs(level='ERROR'):
                tool.optimize()
                self.wait(tool)
        self.assertEqual((seen['method'], seen['loss'], seen['f_scale']),
                         ('trf', 'cauchy', 4.))


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestSaveResultsAs(ToolTestCase):

    def test_the_file_that_is_chosen_receives_the_results(self):
        tool = self.tool()
        tool.optimize()
        self.wait(tool)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'chosen.csv')
            with mock.patch('pylorenzmie.lmtool.LMTool.QFileDialog.'
                            'getSaveFileName',
                            return_value=(filename, '')) as dialog:
                tool.saveResultsAs()
            saved = pd.read_csv(filename)
        self.assertEqual(len(saved), 1)
        self.assertIn('z_p', saved.columns)
        self.assertIn('Saved 1 particle', tool.statusBar().currentMessage())
        # the dialog starts at a default name in ~/data/lmtool
        self.assertIn('results_', dialog.call_args.args[2])

    def test_cancelling_the_dialog_saves_nothing(self):
        tool = self.tool()
        tool.optimize()
        self.wait(tool)
        cancelled = mock.Mock(return_value=('', ''))
        with mock.patch('pylorenzmie.lmtool.LMTool.QFileDialog.'
                        'getSaveFileName', cancelled):
            with mock.patch('pylorenzmie.lmtool.LMTool.save_results') as save:
                tool.saveResultsAs()
        save.assert_not_called()

    def test_the_default_name_has_a_time_stamp_in_the_data_folder(self):
        tool = self.tool()
        path = Path(tool._output_path())
        self.assertEqual(path.parent, Path('~/data/lmtool').expanduser())
        self.assertRegex(path.name,
                         r'^results_\d{4}(-\d\d){2}_\d\d(-\d\d){2}\.h5$')


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestUnhandledExceptions(unittest.TestCase):
    '''PyQt aborts the process when a slot raises, unless a hook is set.'''

    def test_the_hook_logs_the_exception_and_tells_the_user(self):
        error = ValueError('no good')
        with mock.patch.object(launch, 'QMessageBox') as box:
            with self.assertLogs(launch.logger, 'ERROR') as logged:
                launch.report_exception(ValueError, error, None)
        box.return_value.exec.assert_called_once()
        text = box.call_args.args[2]
        self.assertEqual(text, 'ValueError: no good')
        self.assertIn('Unhandled exception', logged.output[0])

    def test_a_keyboard_interrupt_goes_to_the_default_hook(self):
        with mock.patch.object(sys, '__excepthook__') as default:
            with mock.patch.object(launch, 'QMessageBox') as box:
                launch.report_exception(KeyboardInterrupt,
                                        KeyboardInterrupt(),
                                        None)
        default.assert_called_once()
        box.assert_not_called()

    def test_a_message_box_that_fails_does_not_raise(self):
        failing = mock.Mock(side_effect=RuntimeError('no display'))
        with mock.patch.object(launch, 'QMessageBox', failing):
            with self.assertLogs(launch.logger, 'ERROR') as logged:
                launch.report_exception(ValueError, ValueError('x'), None)
        self.assertEqual(len(logged.output), 2)

    def test_the_program_installs_the_hook(self):
        saved = sys.excepthook
        self.addCleanup(setattr, sys, 'excepthook', saved)
        no_file = mock.Mock(return_value=('', ''))
        run = mock.Mock(return_value=0)
        with mock.patch.object(sys, 'argv', ['lmtool']):
            with mock.patch.object(launch.QApplication, 'exec', run):
                with mock.patch.object(launch.sys, 'exit') as leave:
                    with mock.patch('pylorenzmie.lmtool.LMTool.'
                                    'QFileDialog.getOpenFileName',
                                    no_file):
                        with mock.patch('pylorenzmie.lmtool.LMTool.'
                                        'LMTool.show'):
                            launch.lmtool()
        leave.assert_called_once_with(0)
        self.assertIs(sys.excepthook, launch.report_exception)

    def test_a_slot_that_raises_does_not_abort_the_process(self):
        '''Without the hook, this script ends with a core dump.'''
        script = textwrap.dedent('''
            import os, sys
            from unittest import mock
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            import qtpy
            from qtpy.QtCore import QObject, Signal, Slot
            from qtpy.QtWidgets import QApplication
            from pylorenzmie.lmtool import launch

            app = QApplication([])

            class Source(QObject):
                go = Signal()

                @Slot()
                def boom(self):
                    raise OSError('disk full')

            source = Source()
            source.go.connect(source.boom)
            sys.excepthook = launch.report_exception
            with mock.patch.object(launch, 'QMessageBox'):
                source.go.emit()
            print('survived')
        ''')
        result = subprocess.run([sys.executable, '-I', '-c', script],
                                capture_output=True,
                                text=True,
                                check=False,
                                timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('survived', result.stdout)
        self.assertTrue(re.search('disk full', result.stderr), result.stderr)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
