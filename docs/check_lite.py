"""Run every docs notebook in Chromium from the built site, as a visitor would, and fail on any cell error.

A browser workspace that cannot install or run MudGym still builds a clean static site, so this serves `site/` over
HTTP and checks it two ways. It installs the built wheel in stock Pyodide and plays an episode, letting the wheel's
metadata resolve its dependencies from PyPI. Then it runs every notebook in `site/lite/` with PyPI blocked, so a
package the build failed to bundle fails too.

Run through the justfile, which builds the docs first:

    just check-docs-lite
"""

import functools
import http.server
import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import Browser, sync_playwright

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPOSITORY_ROOT / "site"
NOTEBOOKS_DIR = REPOSITORY_ROOT / "docs" / "notebooks"
KERNEL_START_SECONDS = 180
NOTEBOOK_RUN_SECONDS = 900
KERNEL_STATUS = "() => window.jupyterapp?.shell.currentWidget?.sessionContext?.session?.kernel?.status ?? null"
# Returning the command's promise would make evaluate wait for the whole run, so the deadline could never fire.
RUN_ALL_CELLS = """() => {
    window.runAllError = null;
    window.jupyterapp.commands.execute("notebook:run-all-cells").catch((error) => {
        window.runAllError = String(error);
    });
}"""
# Cells left unrun mean the run is still going, or stopped at an error.
RUN_STATE = """() => {
    const panel = window.jupyterapp.shell.currentWidget;
    const cells = panel.content.widgets.filter((cell) => cell.model.type === "code");
    const errors = cells.flatMap((cell) => cell.model.outputs.toJSON()
        .filter((output) => output.output_type === "error")
        .map((output) => `${output.ename}: ${output.evalue}`));
    if (window.runAllError) errors.push(window.runAllError);
    const ran = cells.filter((cell) => cell.model.executionCount !== null).length;
    return {errors, ran, total: cells.length, idle: panel.sessionContext.session.kernel.status === "idle"};
}"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def serve_site() -> Iterator[str]:
    handler = functools.partial(QuietHandler, directory=SITE_DIR)
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()


def run_notebook(browser: Browser, base_url: str, name: str) -> None:
    page = browser.new_page()
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    # JupyterLite's service worker fetches the kernel's requests itself, and only context routes see those.
    page.context.route("https://pypi.org/**", lambda route: route.abort())
    page.context.route("https://files.pythonhosted.org/**", lambda route: route.abort())
    try:
        page.goto(f"{base_url}/lite/notebooks/index.html?path={name}")
        # Cells run before the kernel is up are skipped without an error.
        page.wait_for_function(f"({KERNEL_STATUS})() === 'idle'", timeout=KERNEL_START_SECONDS * 1000)
        deadline = time.monotonic() + NOTEBOOK_RUN_SECONDS
        page.evaluate(RUN_ALL_CELLS)
        while True:
            state = page.evaluate(RUN_STATE)
            if state["errors"]:
                raise AssertionError(f"{name} failed after {state['ran']} cells: {state['errors']}")
            if state["idle"] and state["ran"] == state["total"]:
                break
            if time.monotonic() > deadline:
                raise AssertionError(f"{name} ran {state['ran']} of {state['total']} cells in {NOTEBOOK_RUN_SECONDS}s")
            page.wait_for_timeout(1000)
        if page_errors:
            raise AssertionError(f"{name} raised in the page: {page_errors}")
    finally:
        page.close()


def check_wheel_install(browser: Browser, base_url: str) -> None:
    """Install the built wheel in stock Pyodide, resolving dependencies without the docs requirements or lock."""
    wheels = SITE_DIR / "lite" / "static" / "pyodide-lock"
    (wheel,) = wheels.glob("mudgym-*.whl")
    lock = json.loads((wheels / "pyodide-lock.json").read_text())
    # Use the same Pyodide release as the notebooks, with its original package list.
    pyodide_url = urljoin(lock["packages"]["micropip"]["file_name"], "pyodide.mjs")
    wheel_url = f"{base_url}/{wheel.relative_to(SITE_DIR).as_posix()}"
    source = f"import micropip\nawait micropip.install({wheel_url!r})\n"
    source += (REPOSITORY_ROOT / "scripts" / "smoke.py").read_text()
    page = browser.new_page()
    try:
        page.goto(base_url)
        page.evaluate(
            """({pyodideUrl, source}) => {
                const workerSource = `
                    self.onmessage = async ({data}) => {
                        try {
                            const {loadPyodide} = await import(data.pyodideUrl)
                            const pyodide = await loadPyodide()
                            await pyodide.loadPackage("micropip")
                            await pyodide.runPythonAsync(data.source)
                            self.postMessage({passed: true})
                        } catch (error) {
                            self.postMessage({error: String(error)})
                        }
                    }
                `
                const url = URL.createObjectURL(new Blob([workerSource], {type: "text/javascript"}))
                const worker = new Worker(url, {type: "module"})
                worker.onmessage = ({data}) => {
                    window.installResult = data
                    worker.terminate()
                    URL.revokeObjectURL(url)
                }
                worker.onerror = (error) => {
                    window.installResult = {error: error.message}
                    worker.terminate()
                    URL.revokeObjectURL(url)
                }
                worker.postMessage({pyodideUrl, source})
            }""",
            {"pyodideUrl": pyodide_url, "source": source},
        )
        page.wait_for_function("window.installResult !== undefined", timeout=KERNEL_START_SECONDS * 1000)
        result = page.evaluate("window.installResult")
        if not result.get("passed"):
            raise AssertionError(f"The built wheel failed in a fresh Pyodide: {result.get('error', result)}")
    finally:
        page.close()
    print("Built wheel: fresh Pyodide install and WASM gameplay passed", flush=True)


def main() -> int:
    if not (SITE_DIR / "lite" / "notebooks" / "index.html").is_file():
        raise SystemExit("site/lite has no notebook app. Build the docs first with `just build-docs`.")
    names = [path.name for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb"))]
    with serve_site() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            check_wheel_install(browser, base_url)
            for name in names:
                started = time.monotonic()
                run_notebook(browser, base_url, name)
                print(f"{name}: every cell ran in the browser in {time.monotonic() - started:.0f}s", flush=True)
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
