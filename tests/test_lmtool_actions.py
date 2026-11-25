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
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    import qtpy     # chooses the Qt binding (QT_API) before pyqtgraph can
    from pyqtgraph import mkQApp
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
class TestRobust(ToolTestCase):

    def test_the_action_selects_robust_optimization(self):
        tool = self.tool()
        optimizer = tool.fitWidget.optimizer
        self.assertFalse(optimizer.robust)
        tool.actionRobust.setChecked(True)
        self.assertTrue(optimizer.robust)
        self.assertEqual(tool.optimizerWidget.settings['method'], 'trf')
        self.assertEqual(tool.optimizerWidget.settings['loss'], 'cauchy')
        tool.actionRobust.setChecked(False)
        self.assertFalse(optimizer.robust)
        self.assertEqual(tool.optimizerWidget.settings['method'], 'lm')
        self.assertEqual(tool.optimizerWidget.settings['loss'], 'linear')

    def test_a_fit_uses_the_robust_settings(self):
        tool = self.tool()
        tool.setRobust(True)
        seen = {}

        def fit(snapshot, cancel):
            seen.update(snapshot.settings)
            raise RuntimeError('stop')

        with mock.patch('pylorenzmie.lmtool.LMTool.run_fit', fit):
            with self.assertLogs(level='ERROR'):
                tool.optimize()
                self.wait(tool)
        self.assertEqual((seen['method'], seen['loss']), ('trf', 'cauchy'))


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
