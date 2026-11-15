# lmtool

`lmtool`, the interactive program for fitting holograms, built with
pyqtgraph and `qtpy`, which runs on PyQt6, PyQt5 and PySide6.  It is started by the `lmtool` command or
`python -m pylorenzmie.lmtool`.  This directory is a package without an
`__init__.py`: the program is launched by name and not imported.

- `LMTool.py` is the main window and `LMSession.py` its state.
- `launch.py`, `arguments.py` and `__main__.py` start the program.
- The widgets are `ImageWidget.py`, `FitWidget.py`, `ProfileWidget.py`,
  `ParameterWidget.py`, `OptimizerWidget.py`, `ALMWidget.py`,
  `ParticleTable.py`, `DoubleSlider.py`, `DoubleSpinBox.py` and
  `Preview.py`, with the layouts in the `.ui` files.
- `required.py` has the one function that says that Qt made an object that
  its stubs allow to be `None`.
- `parameters.py` says which parameters are shown and how, `results.py`
  saves the fits, and `tasks.py` runs the work in a worker thread.
  `splash.png` is rendered by `tools/make_splash.py`.

Update this file when the role or contents of the directory change.
