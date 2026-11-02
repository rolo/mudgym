import subprocess
import sys


def test_static_notebook_helpers_do_not_import_anywidget():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class RejectWidgetImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'anywidget', 'ipywidgets'}:
            raise AssertionError(f'Unexpected widget dependency: {fullname}')

sys.meta_path.insert(0, RejectWidgetImports())
from mudgym.notebooks import show_room_map, show_ansi
assert 'Road' in show_room_map([('Road', 'north', 'Hut')]).data
assert 'Room' in show_ansi(b'Room').data
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
