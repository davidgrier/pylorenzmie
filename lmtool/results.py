'''Save the results of the fits of an lmtool session.'''

import json
import logging
from pathlib import Path
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


def save_results(results: pd.DataFrame,
                 metadata: pd.Series,
                 filename: str) -> str:
    '''Write a table of results, one row per particle, and what goes with it.

    The format follows the suffix of the file name: ``.json``, ``.csv``, or
    HDF5 for anything else.  HDF5 needs PyTables; without it the results go to
    a JSON file instead, and the name that was written is returned.

    Parameters
    ----------
    results : pandas.DataFrame
        One row per particle.
    metadata : pandas.Series
        Parameters that the fits held fixed, their settings and the name of
        the data file.
    filename : str
        Where to write.

    Returns
    -------
    filename : str
        The file that was written.
    '''
    path = Path(filename).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        results.to_csv(path, index_label='particle')
        return str(path)
    if suffix != '.json':
        try:
            _save_hdf(results, metadata, path)
            return str(path)
        except ImportError:
            logger.warning('tables not installed; saving as JSON instead')
            path = path.with_suffix('.json')
    _save_json(results, metadata, path)
    return str(path)


def _save_hdf(results: pd.DataFrame, metadata: pd.Series, path: Path) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                category=pd.errors.PerformanceWarning)
        results.to_hdf(path, key='results', mode='w')
        metadata.to_hdf(path, key='metadata')


def _save_json(results: pd.DataFrame, metadata: pd.Series,
               path: Path) -> None:
    document = dict(metadata=json.loads(metadata.to_json()),
                    results=json.loads(results.to_json(orient='records')))
    path.write_text(json.dumps(document, indent=4))
