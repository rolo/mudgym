"""MudGym imports without its native engine, the way a browser kernel loads it."""

import subprocess
import sys
import textwrap


def test_shared_imports_do_not_load_native_backends():
    source = textwrap.dedent("""
        import importlib.abc
        import sys

        forbidden = ("pexpect", "ptyprocess", "pty", "wasmtime", "docker")

        class RejectNativeImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in forbidden:
                    raise AssertionError("shared import attempted to load " + fullname)

        sys.meta_path.insert(0, RejectNativeImports())
        from mudgym import make_env, make_parallel_env
        from mudgym.connections.connection import MudConnection
        from mudgym.connections.provider import ConnectionProvider
        from mudgym.connections.termination import has_game_over_prompt, is_permadeath
        from mudgym.connections.wasm.engine_contract import CommandTicketStatus, WorldStatus
        from mudgym.connections.wasm import WasmProvider, create_connection
        from mudgym.notebooks import show_game_tabs

        assert callable(make_env) and callable(make_parallel_env)
        assert int(CommandTicketStatus.COMPLETED) == 0
        assert int(WorldStatus.ACTIVE) == 1
        assert has_game_over_prompt(b"Cheerio!\\r\\n")
        assert not has_game_over_prompt(b"say Cheerio!\\r\\n")
        assert not is_permadeath(b"Cheerio!\\r\\n")
        assert "game-tabs" in show_game_tabs({"raw_bytes": b"A room.\\r\\n", "render_bytes": b"A room.\\r\\n"}).data
        # The provider is what a browser kernel runs, so it must load without wasmtime too.
        assert callable(create_connection) and WasmProvider is not None
        assert not any(name.split(".")[0] in forbidden for name in sys.modules)
    """)
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_default_connection_still_reaches_wasmtime_on_use():
    source = textwrap.dedent("""
        import sys
        from mudgym.connections import registry

        assert "wasmtime" not in sys.modules
        assert registry.default_connection is registry.connections["wasm"]
        from mudgym.connections.wasm import WasmtimeRuntime, create_connection
        assert "wasmtime" in sys.modules
        assert callable(create_connection) and WasmtimeRuntime is not None
    """)
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
