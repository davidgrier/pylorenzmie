from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('pylorenzmie')
except PackageNotFoundError:
    __version__ = 'unknown'

__all__ = ['__version__']
