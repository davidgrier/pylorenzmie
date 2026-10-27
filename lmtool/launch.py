'''Start lmtool, showing a splash screen while the pipeline loads.

Importing the analysis pipeline takes seconds, and the first run after an
installation also compiles the Numba kernels.  Nothing slow may therefore be
imported at the top of this module: the window of the splash screen has to
exist before :mod:`pylorenzmie.lmtool.LMTool` is imported.
'''

import logging
from pathlib import Path
import sys

from pyqtgraph.Qt.QtCore import Qt
from pyqtgraph.Qt.QtGui import QColor, QFont, QPainter, QPixmap
from pyqtgraph.Qt.QtWidgets import QApplication, QSplashScreen

from pylorenzmie import __version__
from pylorenzmie.lmtool.arguments import argument_parser


def _splash() -> QSplashScreen:
    '''A hologram of a particle, with the name and version on top.

    The image is rendered by ``devel/make_splash.py``, at twice the size
    on the screen, so that it is sharp on high-resolution displays.
    '''
    pixmap = QPixmap(str(Path(__file__).parent / 'splash.png'))
    pixmap.setDevicePixelRatio(2.)
    width, height = pixmap.width() // 2, pixmap.height() // 2

    painter = QPainter(pixmap)
    painter.setPen(QColor('white'))
    font = QFont()
    font.setPointSize(44)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(0, 0, width - 24, height - 62,
                     Qt.AlignmentFlag.AlignRight
                     | Qt.AlignmentFlag.AlignBottom, 'lmtool')
    font.setPointSize(13)
    font.setBold(False)
    painter.setFont(font)
    painter.drawText(0, 0, width - 24, height - 40,
                     Qt.AlignmentFlag.AlignRight
                     | Qt.AlignmentFlag.AlignBottom,
                     f'pylorenzmie {__version__}')
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setFont(QFont(splash.font().family(), 11))
    return splash


def _show(splash: QSplashScreen, message: str) -> None:
    '''Display *message* and repaint before the next slow step.'''
    splash.showMessage(message,
                       Qt.AlignmentFlag.AlignBottom
                       | Qt.AlignmentFlag.AlignRight,
                       QColor('#d9b8f0'))
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
