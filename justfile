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
    uv run pytest --tb=short --durations=20 --cov=src --cov-branch --cov-report=xml --cov-report=html --cov-report=term-missing

# Build the sdist and wheel from scratch. Deliberately no `just publish`:
# production publishing is CI and OIDC only, so a local mistake cannot reach PyPI.
build:
    rm -rf dist
    uv build

check-dist: build
    uvx twine check --strict dist/*
    uvx check-wheel-contents dist/*.whl

# run the docs code snippets against the live game to record sessions in docs/recordings/
docs-record *names:
    uv run python docs/record.py {{names}}

# replay committed session captures through the env stack to rewrite the docs displayed fragments
docs-derive *names:
    uv run python docs/record.py --derive-only {{names}}

docs:
    uv run zensical serve

build-docs:
    uv run zensical build --strict
