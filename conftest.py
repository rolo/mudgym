"""
Streaming the mudgym logger during a test run is opt-in via the MUDGYM_LOG_LEVEL environment variable (the same
variable setup_logging() honours). For example:

    MUDGYM_LOG_LEVEL=DEBUG pytest src/mudgym/connections/tests/test_connection.py -vv

shows the per-command debug output (sm.send, sm.send_command.complete with seen_prompts/terminated/truncated, ...).
With the variable unset the run stays quiet, since the library defaults to WARNING with a null handler. An explicit
--log-cli-level on the command line always takes precedence.
"""

import os

import pytest
from tabulate import tabulate

pytest_plugins = ["tests.fixtures"]


@pytest.fixture(scope="session")
def tea_results(request):
    """Collect test_tea.py timings per connection type for the end-of-session comparison table."""
    request.config._tea_results = {}
    return request.config._tea_results


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Tea timing is session-local and intentionally serial; under pytest-xdist workers keep their
    # own config state, so the controller summary may be empty.
    tea_results = getattr(config, "_tea_results", None)
    if not tea_results:
        return

    stages = list(next(iter(tea_results.values())).keys())
    rows = [
        [connection, *(f"{results[stage]['total']:.3f}" if stage in results else "N/A" for stage in stages)]
        for connection, results in tea_results.items()
    ]

    terminalreporter.write_sep("-", "time to tea")
    terminalreporter.write_line(tabulate(rows, headers=["Connection", *stages], tablefmt="simple"))


def pytest_configure(config):
    level = os.environ.get("MUDGYM_LOG_LEVEL")
    if level and config.option.log_cli_level is None:
        config.option.log_cli_level = level
