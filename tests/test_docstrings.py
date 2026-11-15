'''The public API is documented, so that Sphinx can publish it.

Sphinx lists only what has a docstring (there is no ``:undoc-members:``),
so an undocumented public class, function, method or property silently
disappears from the documentation.  This test says which.

A method is documented if it, or the method of a base class that it
overrides, has a docstring: Sphinx inherits docstrings the same way.
Overrides of Qt event handlers (``*Event``) are exempt, and so are Qt
signals, which are documented in the docstring of their class.
'''
import importlib
import inspect
import pkgutil
import unittest

#: The packages, and single modules, whose public API must be documented.
PACKAGES = ('pylorenzmie.lib', 'pylorenzmie.theory',
            'pylorenzmie.analysis', 'pylorenzmie.utilities')
MODULES = ()


def modules(package: str):
    '''Yield the modules of a package, and the package itself.'''
    pkg = importlib.import_module(package)
    yield pkg
    for info in pkgutil.walk_packages(pkg.__path__, package + '.'):
        try:
            yield importlib.import_module(info.name)
        except ImportError:
            # needs an optional library (CuPy, JAX, Torch, Triton) that
            # this environment does not have
            continue


def own_doc(member) -> str:
    '''Return the docstring that a member has of its own.'''
    if isinstance(member, property):
        member = member.fget
    member = getattr(member, '__func__', member)
    return inspect.cleandoc(getattr(member, '__doc__', None) or '')


def documented(cls: type, name: str) -> bool:
    '''Whether a member of a class is documented here or in a base class
    that belongs to pylorenzmie.'''
    for base in cls.__mro__:
        if not getattr(base, '__module__', '').startswith('pylorenzmie'):
            continue
        member = base.__dict__.get(name)
        if member is not None and own_doc(member):
            return True
    return False


def is_member(value) -> bool:
    '''Whether a class attribute is something to document.'''
    unwrapped = getattr(value, '__func__', value)
    return (inspect.isfunction(unwrapped) or isinstance(value, property)
            or inspect.isclass(value))


def missing(module) -> list[str]:
    '''Return the public items of a module that have no docstring.'''
    found = []
    for name, obj in vars(module).items():
        if name.startswith('_') or getattr(obj, '__module__', None) \
                != module.__name__:
            continue
        where = f'{module.__name__}.{name}'
        if inspect.isfunction(obj):
            if not own_doc(obj):
                found.append(where)
        elif inspect.isclass(obj):
            if not (obj.__doc__ or '').strip():
                found.append(where)
            for attr, value in vars(obj).items():
                if attr.startswith('_') or attr.endswith('Event'):
                    continue
                if is_member(value) and not documented(obj, attr):
                    found.append(f'{where}.{attr}')
    return found


class TestDocstrings(unittest.TestCase):

    def test_the_public_api_is_documented(self):
        found = []
        for package in PACKAGES:
            for module in modules(package):
                found += missing(module)
        for name in MODULES:
            found += missing(importlib.import_module(name))
        self.assertEqual(
            found, [], 'public items without a docstring:\n  '
            + '\n  '.join(found))


if __name__ == '__main__':
    unittest.main()
