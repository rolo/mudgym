"""The command-line client exposes the installed WASM engine."""

import os
import subprocess
import sys


def test_cli_help_works_with_legacy_connection_settings():
    result = subprocess.run(
        [sys.executable, "-m", "mudgym.cli", "--help"],
        env={
            **os.environ,
            "AVAILABLE_CONNECTIONS": "docker_run,docker_exec",
            "MUDGYM_DEFAULT_CONNECTION": "docker_exec",
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert "wasm" in result.stdout
    assert "docker" not in result.stdout.lower()
