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

docs:
    uv run zensical serve

build-docs:
    uv run zensical build

# The results are committed, so building the docs itself does not need Docker.
# Re-capture the docs transcripts by running the game. Requires Docker.
docs-transcripts:
    uv run python scripts/generate_docs_transcripts.py
