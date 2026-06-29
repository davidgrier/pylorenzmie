'''Sphinx configuration for pylorenzmie.'''

import sys
from pathlib import Path

# The repo root is one level up from docs/.  Adding it to sys.path lets
# autodoc import pylorenzmie without requiring an editable install.
sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----------------------------------------------------

project = 'pylorenzmie'
author = 'David G. Grier'
copyright = '2026, David G. Grier'
from importlib.metadata import version as _get_version, PackageNotFoundError
try:
    release = _get_version('pylorenzmie')
except PackageNotFoundError:
    release = '0.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

# Napoleon settings for NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

# autodoc settings
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'undoc-members': False,
    'show-inheritance': True,
}

# Optional heavy dependencies not available on doc-build hosts
autodoc_mock_imports = [
    'cupy',
    'jax',
    'jax.numpy',
    'numba',
    'torch',
    'triton',
    'triton.language',
    'triton.language.extra.cuda.libdevice',
    'pyqtgraph',
    'trackpy',
]

# intersphinx: link to external package docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

exclude_patterns = ['_build']

# Suppress warnings that cannot be fixed in source
suppress_warnings = [
    'sphinx_autodoc_typehints.local_function',   # AberratedLorenzMie factory
    'sphinx_autodoc_typehints.forward_reference', # numpy internal type
]

nitpick_ignore = [
    ('py:class', '_SeriesLikeCoef_co'),  # numpy internal type, not our code
]


def setup(app):
    '''Suppress duplicate-object warnings from Sphinx 8.x.'''
    from sphinx.domains.python import PythonDomain
    _orig = PythonDomain.note_object
    _seen = {}

    def _dedup(self, name, objtype, node_id, aliased=False, location=None):
        if not aliased and name in _seen:
            return _orig(self, name, objtype, node_id, aliased=True,
                         location=location)
        if not aliased:
            _seen[name] = self.env.docname
        return _orig(self, name, objtype, node_id, aliased, location)

    PythonDomain.note_object = _dedup

# -- HTML output -------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_title = 'pylorenzmie'
html_static_path = ['_static']
html_css_files = ['nyu.css']

html_theme_options = {
    'github_url': 'https://github.com/davidgrier/pylorenzmie',
    'show_toc_level': 2,
    'navigation_with_keys': True,
    'show_nav_level': 2,
    'navbar_end': ['navbar-icon-links', 'theme-switcher'],
    'footer_start': ['copyright'],
    'footer_end': ['sphinx-version'],
}

html_sidebars = {'**': []}
