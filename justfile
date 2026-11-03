set shell := ["bash", "-euo", "pipefail", "-c"]

@_:
    just --list

# sync the locked dependencies
install:
    uv sync --locked

# check ruff's lint rules and formatting
lint:
    uv run ruff check .
    uv run ruff format --check .

# apply ruff's fixes and formatting
fix:
    uv run ruff check --fix .
    uv run ruff format .

# run the tests with coverage
test:
    uv run pytest --tb=short --durations=20 --cov=src --cov-branch \
        --cov-report=xml --cov-report=html --cov-report=term-missing

# There's deliberately no `just publish`: publishing is CI and OIDC only, so a local mistake can't reach PyPI.
# build the sdist and wheel from scratch
build:
    rm -rf dist
    uv build

# check the built sdist and wheel as PyPI would
check-dist: build
    uvx twine check --strict dist/*
    uvx check-wheel-contents dist/*.whl

# show version guidance, or validate, version, commit, tag, and push an explicit release
release version="":
    uv run python scripts/release.py {{version}}

# retry a failed unpublished release from the corrected main commit, keeping the same version
release-retry version:
    uv run python scripts/release.py {{version}} --retry

# JupyterLab sits on top rather than in .venv, as its server brings cffi, which the tests check MudGym never needs.
# open the docs notebooks in JupyterLab, with the modules they import beside them, `just lab all` for every interface
lab interfaces="local":
    uv run --group notebooks --with jupyterlab jupyter lab docs/notebooks --ip={{ if interfaces == "all" { "0.0.0.0" } else if interfaces == "local" { "localhost" } else { error("just lab takes `all`, or nothing for this machine only") } }}

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

# Saving an example re-derives its fragments, recording it live first when its commands changed or it's new, and
# zensical reloads the browser with the result.
# serve the docs, refreshing whichever example in docs/code/ you save
docs: docs-derive docs-lite
    #!/usr/bin/env bash
    set -euo pipefail
    # the watcher runs without the `uv run` wrapper (the docs-derive dependency already synced .venv) so the trap's kill
    # reaches the actual process instead of orphaning it when serve exits
    .venv/bin/python docs/record.py --watch &
    watcher_pid=$!
    trap 'kill "$watcher_pid" 2>/dev/null || true' EXIT
    uv run zensical serve

# A failed exec code block renders its traceback into the page instead of failing zensical, so the scan fails it.
# build the docs strictly and fail on any traceback in a page
docs-build:
    uv run python docs/build.py

# run every docs notebook in Chromium from the built site and fail on any cell error, as the docs workflow does
docs-check: docs-build
    uv run --with playwright==1.58.0 python docs/check_lite.py
