"""Build the browser workspace where the docs notebooks run with nothing to install.

Every notebook in `docs/notebooks/` opens in JupyterLite, running on Python in the browser through Pyodide, with
MudGym built from this checkout and the engine wheel uv.lock names. A notebook's browser version is the recorded one
with its outputs cleared and the cells of its overlay in `docs/jupyterlite/browser/`, if it has one, swapped in by
id. Only what differs in the browser lives there, such as an intro that says nothing needs installing, or weights
offered for download rather than written beside the notebook.

The workspace is written to `docs/pages/lite/`, which Git ignores, so the docs build publishes it with the pages and
each notebook page links to its browser version there.

Run through the justfile:

    just docs-lite
"""

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPOSITORY_ROOT / "docs" / "notebooks"
LITE_SOURCE_DIR = REPOSITORY_ROOT / "docs" / "jupyterlite"
OVERLAYS_DIR = LITE_SOURCE_DIR / "browser"
OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "pages" / "lite"
ENGINE_PACKAGE = "mudgym-wasm-engine"
JUPYTERLITE_VERSION = "0.8.3"
# Naming the browser's kernel lets JupyterLite open a notebook without asking which kernel to start.
PYODIDE_KERNELSPEC = {"display_name": "Python (Pyodide)", "language": "python", "name": "python"}


def cleared(cell: dict) -> dict:
    """A copy of the cell ready for a fresh run, with no recorded outputs."""
    cell = {key: value for key, value in cell.items() if key not in ("outputs", "execution_count")}
    if cell["cell_type"] == "code":
        cell |= {"outputs": [], "execution_count": None}
    return cell


def browser_notebook(source: dict, overlay: dict) -> dict:
    """Swap the overlay's cells into the source notebook by id and clear every output."""
    replacements = {cell["id"]: cell for cell in overlay["cells"]}
    unknown = sorted(set(replacements) - {cell["id"] for cell in source["cells"]})
    if unknown:
        raise ValueError(f"the overlay names cells the notebook does not have: {unknown}")
    cells = [cleared(replacements.get(cell["id"], cell)) for cell in source["cells"]]
    metadata = {"kernelspec": PYODIDE_KERNELSPEC, "language_info": {"name": "python"}}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def write_browser_notebooks(contents_dir: Path) -> None:
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    orphans = sorted({path.name for path in OVERLAYS_DIR.glob("*.ipynb")} - {path.name for path in notebooks})
    if orphans:
        raise SystemExit(f"docs/jupyterlite/browser has overlays for notebooks docs/notebooks lacks: {orphans}")
    contents_dir.mkdir(parents=True)
    for path in notebooks:
        overlay_path = OVERLAYS_DIR / path.name
        overlay = json.loads(overlay_path.read_text()) if overlay_path.is_file() else {"cells": []}
        notebook = browser_notebook(json.loads(path.read_text()), overlay)
        (contents_dir / path.name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def engine_version() -> str:
    """The engine wheel version uv.lock resolved, which the native tests run."""
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())
    return next(package["version"] for package in lock["package"] if package["name"] == ENGINE_PACKAGE)


def build_wheel(wheels_dir: Path) -> Path:
    subprocess.run(["uv", "build", "--wheel", str(REPOSITORY_ROOT), "-o", str(wheels_dir)], check=True)
    (wheel,) = wheels_dir.glob("mudgym-*.whl")
    return wheel


def configure(lite_dir: Path, wheel: Path) -> None:
    """Resolve mudgym to this checkout's wheel, which PyPI's release of the same version is not, and pin the engine."""
    pins = lite_dir / "pins.txt"
    pins.write_text(f"mudgym @ {wheel.as_uri()}\n{ENGINE_PACKAGE}=={engine_version()}\n")
    config_path = lite_dir / "jupyter_lite_config.json"
    config = json.loads(config_path.read_text())
    config["PyodideLockAddon"] |= {
        "wheels": [str(wheel)],
        "pyodide_lock_uv_options": {"extra_uv_args": [f"--overrides={pins}"]},
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def build_workspace(workspace: Path, lite_dir: Path, contents_dir: Path) -> None:
    # Jupyter's build database and this machine's Jupyter configuration stay in the scratch workspace.
    environment = os.environ | {
        f"JUPYTER_{name}_DIR": str(workspace / name.lower()) for name in ("CONFIG", "DATA", "RUNTIME")
    }
    command = [
        "uvx",
        "--from",
        f"jupyterlite-core[contents]=={JUPYTERLITE_VERSION}",
        "--with",
        f"jupyterlite-pyodide-kernel[lock]=={JUPYTERLITE_VERSION}",
        "jupyter",
        "lite",
        "build",
        "--lite-dir",
        str(lite_dir),
        "--contents",
        str(contents_dir),
        "--output-dir",
        str(OUTPUT_DIR),
        "--apps",
        "notebooks",
        "--apps",
        "tree",
        "--apps",
        "lab",
        "--no-sourcemaps",
    ]
    subprocess.run(command, cwd=workspace, env=environment, check=True)
    # JupyterLite also copies its frontend bundler configuration, which browsers never load.
    for bundler_config in OUTPUT_DIR.glob("rspack.config*.js"):
        bundler_config.unlink()


def main() -> None:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="mudgym-lite-") as workspace_name:
        workspace = Path(workspace_name)
        lite_dir = workspace / "lite"
        shutil.copytree(LITE_SOURCE_DIR, lite_dir, ignore=shutil.ignore_patterns("browser"))
        contents_dir = workspace / "files"
        write_browser_notebooks(contents_dir)
        configure(lite_dir, build_wheel(workspace / "wheels"))
        build_workspace(workspace, lite_dir, contents_dir)
    print(f"built the browser notebooks into {OUTPUT_DIR.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
