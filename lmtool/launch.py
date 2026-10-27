'''Start lmtool, showing a splash screen while the pipeline loads.

Importing the analysis pipeline takes seconds, and the first run after an
installation also compiles the Numba kernels.  Nothing slow may therefore be
imported at the top of this module: the window of the splash screen has to
exist before :mod:`pylorenzmie.lmtool.LMTool` is imported.
'''

import logging
import sys

from pyqtgraph.Qt.QtCore import Qt
from pyqtgraph.Qt.QtGui import QColor, QFont, QPixmap
from pyqtgraph.Qt.QtWidgets import QApplication, QSplashScreen

from pylorenzmie import __version__
from pylorenzmie.lmtool.arguments import argument_parser


def _splash() -> QSplashScreen:
    pixmap = QPixmap(420, 140)
    pixmap.fill(QColor('#57068c'))
    splash = QSplashScreen(pixmap)
    splash.setFont(QFont(splash.font().family(), 14))
    return splash


def _show(splash: QSplashScreen, message: str) -> None:
    '''Display *message* and repaint before the next slow step.'''
    splash.showMessage(f'lmtool {__version__}\n{message}',
                       Qt.AlignmentFlag.AlignCenter, QColor('white'))
    QApplication.processEvents()


def lmtool() -> None:
    logging.basicConfig(level=logging.WARNING,
                        format='%(name)s: %(levelname)s: %(message)s')

    parser = argument_parser()
    args, unparsed = parser.parse_known_args()
    app = QApplication.instance() or QApplication(sys.argv[:1] + unparsed)

    splash = _splash()
    splash.show()
    # macOS does not display the splash screen unless it is also raised
    # and activated.
    splash.raise_()
    splash.activateWindow()

    # Everything below is slow to import: keep it after the splash screen.
    _show(splash, 'Loading the analysis pipeline...')
    from pylorenzmie.theory import model_class
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
    from pylorenzmie.lmtool.LMTool import LMTool

    _show(splash, 'Preparing the scattering model...')
    try:
        model = model_class(args.backend, args.distorted)
    except (ValueError, ImportError) as error:
        splash.close()
        parser.error(str(error))

    _show(splash, 'Starting...')
    tool = LMTool(ALMWidget.with_model(model), args.filename,
                  args.background, args.dark)
    tool.show()
    splash.finish(tool)
    sys.exit(app.exec())


if __name__ == '__main__':  # pragma: no cover
    lmtool()
