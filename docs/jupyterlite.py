"""Build the JupyterLite site the docs notebooks run in, with `just docs-lite`.

It uses this checkout's MudGym wheel and the engine version uv.lock has. A notebook's browser version is the recorded
one with its outputs cleared.
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
OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "pages" / "lite"
ENGINE_PACKAGE = "mudgym-wasm-engine"
JUPYTERLITE_VERSION = "0.8.3"
# Naming the kernel saves JupyterLite asking which one to start.
PYODIDE_KERNELSPEC = {"display_name": "Python (Pyodide)", "language": "python", "name": "python"}
# Read the kernel list straight away as the refresh poll doesn't run in a background tab.
KERNEL_LIST_REFRESH = "r&&r.changed.connect(()=>{s.refreshSpecs()})"
KERNEL_LIST_READ = "r&&r.changed.connect(()=>{s.requestSpecs()})"


def cleared(cell: dict) -> dict:
    cell = {key: value for key, value in cell.items() if key not in ("outputs", "execution_count")}
    if cell["cell_type"] == "code":
        cell |= {"outputs": [], "execution_count": None}
    return cell


def browser_notebook(source: dict) -> dict:
    cells = [cleared(cell) for cell in source["cells"]]
    metadata = {"kernelspec": PYODIDE_KERNELSPEC, "language_info": {"name": "python"}}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def write_browser_notebooks(contents_dir: Path) -> None:
    contents_dir.mkdir(parents=True)
    for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        notebook = browser_notebook(json.loads(path.read_text()))
        (contents_dir / path.name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    # The notebooks import the modules beside them.
    for helper in NOTEBOOKS_DIR.glob("*.py"):
        shutil.copyfile(helper, contents_dir / helper.name)


def engine_version() -> str:
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())
    return next(package["version"] for package in lock["package"] if package["name"] == ENGINE_PACKAGE)


def build_wheel(wheels_dir: Path) -> Path:
    subprocess.run(["uv", "build", "--wheel", str(REPOSITORY_ROOT), "-o", str(wheels_dir)], check=True)
    (wheel,) = wheels_dir.glob("mudgym-*.whl")
    return wheel


def configure(lite_dir: Path, wheel: Path) -> None:
    """PyPI's mudgym of the same version isn't this checkout, so point it at the wheel and pin the engine."""
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
    # Keep Jupyter's config and build database in the scratch workspace, away from this machine's own.
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
    for bundler_config in OUTPUT_DIR.glob("rspack.config*.js"):
        bundler_config.unlink()


def read_kernel_list_when_hidden() -> None:
    patched = 0
    for chunk in (OUTPUT_DIR / "build").glob("*.js"):
        source = chunk.read_text(encoding="utf-8")
        if KERNEL_LIST_REFRESH in source:
            chunk.write_text(source.replace(KERNEL_LIST_REFRESH, KERNEL_LIST_READ), encoding="utf-8")
            patched += 1
    if not patched:
        raise SystemExit(f"JupyterLite {JUPYTERLITE_VERSION} has no refreshSpecs call to patch.")


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
    read_kernel_list_when_hidden()
    print(f"built the browser notebooks into {OUTPUT_DIR.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
