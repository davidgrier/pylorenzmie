'''The layout of arguments follows the rule of the project.

In a definition or a call, either all the arguments are on one line or each
has a line of its own, with a trailing comma if the closing parenthesis has
one.  A list that is wrapped with several arguments on a line is hard to
read and to change, because a change to one argument moves the others.  A
definition aligns its parameters under the first, as QVideo does; a call
may do that or hang.  ``self`` and ``cls`` may share a line with the first
parameter.  See ``CLAUDE.md``.
'''
import ast
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = ('htmlcov', 'build', '_build', 'devel')


def arguments(node: ast.AST) -> list[ast.AST]:
    '''Return the arguments of a call or a definition.'''
    if isinstance(node, ast.Call):
        return list(node.args) + list(node.keywords)
    a = node.args
    found = a.posonlyargs + a.args + a.kwonlyargs
    return found + [x for x in (a.vararg, a.kwarg) if x]


def definitions(source: str):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def crowded(source: str) -> list[int]:
    '''Return the lines of the argument lists that put several arguments on
    a line but are not wholly on one.'''
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Call, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        items = arguments(node)
        if items and getattr(items[0], 'arg', None) in ('self', 'cls'):
            items = items[1:]
        lines = [a.lineno for a in items]
        if len(lines) < 2 or min(lines) == max(lines):
            continue
        if any(n > 1 for n in Counter(lines).values()):
            found.append(node.lineno)
    return found


def hanging(source: str) -> list[int]:
    '''Return the lines of the definitions that wrap their parameters but
    start them on a line after the name.'''
    return [
        node.lineno
        for node in definitions(source)
        if len(arguments(node)) > 1
        and min(a.lineno for a in arguments(node)) > node.lineno
    ]


SKIP_ANY = set(SKIP)    # a scratchpad or a build, at any depth


def sources():
    for path in sorted(ROOT.rglob('*.py')):
        relative = path.relative_to(ROOT).parts
        if relative[0].startswith('.') or SKIP_ANY.intersection(relative):
            continue
        yield path, path.read_text()


class TestTheDetector(unittest.TestCase):

    def test_one_line_is_fine(self):
        self.assertEqual(crowded('f(a, b, c)\n'), [])

    def test_one_to_a_line_is_fine(self):
        self.assertEqual(crowded('f(\n    a,\n    b,\n)\n'), [])

    def test_aligned_is_fine(self):
        self.assertEqual(crowded('f(a,\n  b,\n  c)\n'), [])

    def test_several_to_a_line_is_found(self):
        self.assertEqual(crowded('f(a, b,\n  c)\n'), [1])

    def test_a_definition_is_found_too(self):
        self.assertEqual(crowded('def f(a, b,\n      c):\n    pass\n'), [1])

    def test_one_argument_is_fine(self):
        self.assertEqual(crowded('f(\n    a,\n)\n'), [])

    def test_self_may_share_a_line_with_the_first_parameter(self):
        text = 'def f(self, a,\n      b):\n    pass\n'
        self.assertEqual(crowded(text), [])

    def test_a_hanging_definition_is_found(self):
        text = 'def f(\n    a,\n    b,\n):\n    pass\n'
        self.assertEqual(hanging(text), [1])

    def test_an_aligned_definition_is_not_hanging(self):
        text = 'def f(a,\n      b):\n    pass\n'
        self.assertEqual(hanging(text), [])

    def test_a_definition_on_one_line_is_not_hanging(self):
        self.assertEqual(hanging('def f(a, b):\n    pass\n'), [])

    def test_a_hanging_call_is_allowed(self):
        self.assertEqual(crowded('f(\n    a,\n    b,\n)\n'), [])


class TestTheCode(unittest.TestCase):

    def check(self, finder, message):
        found = [
            f'{path.relative_to(ROOT)}:{line}'
            for path, source in sources()
            for line in finder(source)
        ]
        self.assertEqual(
            found,
            [],
            message + ':\n  ' + '\n  '.join(found),
        )

    def test_arguments_are_on_one_line_or_one_to_a_line(self):
        self.check(crowded, 'argument lists with several arguments on a line')

    def test_definitions_align_their_parameters(self):
        self.check(
            hanging,
            'definitions that hang their parameters (align them under '
            'the first)',
        )


if __name__ == '__main__':
    unittest.main()
