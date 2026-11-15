'''Objects that Qt always makes, though the stubs of its bindings allow None.

Many getters of Qt, such as ``QMainWindow.statusBar()`` or
``QAbstractItemView.selectionModel()``, are typed as returning an object or
``None``, because the C++ function can return a null pointer.  In lmtool the
object exists, and :func:`required` says so, and fails with a message if it
does not.
'''

from typing import TypeVar

T = TypeVar('T')


def required(value: T | None) -> T:
    '''Return *value*, which Qt must have made.

    Parameters
    ----------
    value : object or None
        What a getter of Qt returned.

    Returns
    -------
    object
        *value*, typed as not ``None``.

    Raises
    ------
    RuntimeError
        If *value* is ``None``.
    '''
    if value is None:
        raise RuntimeError('Qt returned no object')
    return value
