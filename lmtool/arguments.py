'''Command line of lmtool.

Imports nothing heavy, so that the application can read its arguments, and
show a splash screen, before the slow imports of the analysis pipeline.
'''

import argparse
from pathlib import Path

from pylorenzmie.backends import BACKENDS

_ROOT = Path(__file__).parent.parent


def argument_parser() -> argparse.ArgumentParser:
    '''The command line of lmtool.

    Returns
    -------
    argparse.ArgumentParser
        Parser of the file to open and of the options ``--background``,
        ``--distorted`` and ``--backend``.
    '''
    default_file = str(_ROOT / 'docs' / 'tutorials' / 'crop.png')

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
