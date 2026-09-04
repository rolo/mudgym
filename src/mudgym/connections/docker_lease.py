import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from uuid import uuid4

LEASE_PATH = "/tmp/mudgym-owner-lease"
LEASE_TIMEOUT_ENV = "MUDGYM_LEASE_TIMEOUT_SECONDS"


class ContainerLease:
    """Renew owned containers while their owner lives. Their watchdogs enforce expiry."""

    def __init__(self, timeout_seconds: int) -> None:
        if timeout_seconds < 6:
            raise ValueError("lease_timeout_seconds must be at least 6.")
        self.timeout_seconds = timeout_seconds
        self.renewal_interval_seconds = min(30, timeout_seconds / 6)
        self.command_timeout_seconds = min(10, self.renewal_interval_seconds)
        self.identifier = uuid4().hex
        self.renewers: list[subprocess.Popen] = []

    def docker_run_arguments(self) -> list[str]:
        return [
            "--label",
            "mudgym.role=shared-world",
            "--label",
            f"mudgym.lease.id={self.identifier}",
            "--label",
            f"mudgym.lease.owner-pid={os.getpid()}",
            "--label",
            f"mudgym.lease.timeout-seconds={self.timeout_seconds}",
            "--env",
            f"{LEASE_TIMEOUT_ENV}={self.timeout_seconds}",
        ]

    def boot_command(self, slots: int) -> str:
        return lease_watchdog_command(
            f"/app/bin/boot -n {slots} -f -k",
            lease_path=LEASE_PATH,
            check_interval_seconds=self.renewal_interval_seconds,
            stop_grace_seconds=5,
        )

    def add_container(self, container_id: str) -> None:
        # Processes avoid adding threads around pexpect's forkpty-based logins.
        self.renewers.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("docker_lease_guardian.py")),
                    str(os.getpid()),
                    container_id,
                    LEASE_PATH,
                    str(self.renewal_interval_seconds),
                    str(self.command_timeout_seconds),
                ]
            )
        )

    def close(self) -> None:
        for renewer in self.renewers:
            renewer.terminate()
        for renewer in self.renewers:
            try:
                renewer.wait(timeout=self.command_timeout_seconds + 1)
            except subprocess.TimeoutExpired:
                renewer.kill()
                renewer.wait()


def lease_watchdog_command(
    supervised_command: str,
    *,
    lease_path: str,
    check_interval_seconds: float,
    stop_grace_seconds: int,
) -> str:
    """Return a watchdog that sends TERM on expiry, then KILL after the grace period."""
    return dedent(
        f"""\
        touch {lease_path}
        {supervised_command} &
        supervised_pid=$!
        stop_supervised() {{
            kill -TERM "$supervised_pid" 2>/dev/null || true
            deadline=$(($(date +%s) + {stop_grace_seconds}))
            while kill -0 "$supervised_pid" 2>/dev/null && [ "$(date +%s)" -lt "$deadline" ]; do
                sleep 0.25
            done
            kill -KILL "$supervised_pid" 2>/dev/null || true
            wait "$supervised_pid" 2>/dev/null || true
            exit 0
        }}
        trap stop_supervised INT TERM
        while kill -0 "$supervised_pid" 2>/dev/null; do
            sleep {check_interval_seconds} & wait $! || true
            kill -0 "$supervised_pid" 2>/dev/null || break
            [ -e {lease_path} ] || stop_supervised
            now=$(date +%s) || stop_supervised
            renewed=$(stat -c %Y {lease_path}) || stop_supervised
            [ "$((now - renewed))" -lt "${LEASE_TIMEOUT_ENV}" ] || stop_supervised
        done
        wait "$supervised_pid"
        """
    )
