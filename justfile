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

# Build the sdist and wheel from scratch. Deliberately no `just publish`: production publishing is CI and OIDC only, so
# a local mistake cannot reach PyPI.
build:
    rm -rf dist
    uv build

check-dist: build
    uvx twine check --strict dist/*
    uvx check-wheel-contents dist/*.whl

# show version guidance, or validate, version, commit, tag, and push an explicit release
release version="":
    uv run python scripts/release.py {{version}}

# retry a failed unpublished release from the corrected main commit, keeping the same version
release-retry version:
    uv run python scripts/release.py {{version}} --retry

# play the docs examples against the live game to record connection calls in docs/recordings/
docs-record *names:
    uv run python docs/record.py {{names}}

# replay committed connection captures and convert the recorded notebooks into the ignored pages the docs use
docs-derive *names:
    uv run python docs/record.py --derive-only {{names}}
    uv run python docs/notebooks.py

# re-run the docs notebooks against the live game, recording their outputs, then derive their pages
docs-notebooks *names:
    uv run --group notebooks python docs/notebooks.py --execute {{names}}

# build the browser workspace where the docs notebooks run in JupyterLite, which each notebook page links to
docs-lite:
    uv run python docs/jupyterlite.py

# Regenerate the displayed fragments and the browser notebooks before previewing.
docs: docs-derive docs-lite
    uv run zensical serve

# serve the docs while watching docs/code/: saving an example re-derives its fragments (recording it live first when its
# commands changed or it is new), and zensical reloads the browser with the result
docs-watch: docs-derive docs-lite
    #!/usr/bin/env bash
    set -euo pipefail
    # the watcher runs without the `uv run` wrapper (the docs-derive dependency already synced .venv) so the trap's kill
    # reaches the actual process instead of orphaning it when serve exits
    .venv/bin/python docs/record.py --watch &
    watcher_pid=$!
    trap 'kill "$watcher_pid" 2>/dev/null || true' EXIT
    uv run zensical serve

# docs/build.py wraps `zensical build --strict` and then scans the built pages: a failed exec code block renders its
# traceback into the page instead of failing zensical, so the scan fails it loudly
build-docs:
    uv run python docs/build.py

# run every docs notebook in Chromium from the built site and fail on any cell error, as the docs workflow does
check-docs-lite: build-docs
    uv run --with playwright==1.58.0 python docs/check_lite.py
