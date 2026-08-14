set shell := ["bash", "-euo", "pipefail", "-c"]

@_:
    just --list

install:
	uv sync --frozen

upgrade:
    uv lock --upgrade

lint:
    uv run ruff check .

format-check:
    uv run ruff format --check .

fix:
    uv run ruff check --fix .
    uv run ruff format .

format:
    uv run ruff format .

test:
    uv run pytest --tb=short --durations=20 --cov=src --cov-branch \
        --cov-report=xml --cov-report=html --cov-report=term-missing

# Build the sdist and wheel from scratch. Deliberately no `just publish`:
# production publishing is CI and OIDC only, so a local mistake cannot reach PyPI.
build:
    rm -rf dist
    uv build

check-dist: build
    uvx twine check --strict dist/*
    uvx check-wheel-contents dist/*.whl

# validate, version, commit, tag, and push a release; GitHub Actions publishes it via OIDC
release version:
    uv run python scripts/release.py {{version}}

# play the docs examples against the live game to record connection calls in docs/recordings/
docs-record *names:
    uv run python docs/record.py {{names}}

# replay committed connection captures through the env stack to rewrite the docs displayed fragments
docs-derive *names:
    uv run python docs/record.py --derive-only {{names}}

# both docs recipes re-derive the fragments first (fast, no Docker), so editing pages or the
# fragment-writing code in docs/code/ never needs a separate derive step
docs: docs-derive
    uv run zensical serve

# serve the docs while watching docs/code/: saving an example re-derives its fragments (recording it
# live first when its commands changed or it is new), and zensical reloads the browser with the result
docs-watch: docs-derive
    #!/usr/bin/env bash
    set -euo pipefail
    # the watcher runs without the `uv run` wrapper (the docs-derive dependency already synced .venv)
    # so the trap's kill reaches the actual process instead of orphaning it when serve exits
    .venv/bin/python docs/record.py --watch &
    watcher_pid=$!
    trap 'kill "$watcher_pid" 2>/dev/null || true' EXIT
    uv run zensical serve

# docs/build.py wraps `zensical build --strict` and then scans the built pages: a failed exec code
# block renders its traceback into the page instead of failing zensical, so the scan fails it loudly
build-docs: docs-derive
    uv run python docs/build.py
