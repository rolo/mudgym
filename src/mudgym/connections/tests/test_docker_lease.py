import os
import subprocess

import pytest

from mudgym.connections.docker_lease import LEASE_TIMEOUT_ENV, ContainerLease, lease_watchdog_command


def test_lease_timeout_leaves_time_for_renewal():
    with pytest.raises(ValueError, match="at least 6"):
        ContainerLease(5)


@pytest.mark.parametrize("ignore_term", [False, True])
def test_watchdog_stops_its_process_after_expiry(tmp_path, ignore_term):
    pid_file = tmp_path / "pid"
    trap = 'trap "" TERM; ' if ignore_term else ""
    script = lease_watchdog_command(
        f"sh -c '{trap}echo $$ > {pid_file}; exec sleep 30'",
        lease_path=str(tmp_path / "lease"),
        check_interval_seconds=0.1,
        stop_grace_seconds=1,
    )

    result = subprocess.run(
        ["/bin/sh", "-c", script],
        env={**os.environ, LEASE_TIMEOUT_ENV: "1"},
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
