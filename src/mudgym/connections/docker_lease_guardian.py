"""Renew one container's lease until the allocating process exits."""

import os
import signal
import subprocess
import sys
import time


def stop(signum, frame):
    # SystemExit also makes subprocess.run kill and reap a renewal still in progress.
    sys.exit(0)


def main():
    owner_pid, container_id, lease_path, interval, command_timeout = sys.argv[1:]
    signal.signal(signal.SIGTERM, stop)
    while os.getppid() == int(owner_pid):
        try:
            subprocess.run(
                ["docker", "exec", container_id, "touch", lease_path],
                check=True,
                capture_output=True,
                timeout=float(command_timeout),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"MudGym lease renewal failed for {container_id}: {exc}", file=sys.stderr)
        time.sleep(float(interval))


if __name__ == "__main__":
    main()
