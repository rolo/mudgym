"""Build the docs with `just docs-build`, failing if a code block's traceback ended up in a page.

zensical's strict mode misses a failed `exec="true"` block and renders its traceback as the output, so we look for one
in the built HTML, with the tags stripped since pygments splits it over spans.
"""

import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPOSITORY_ROOT / "site"
ERROR_MARKER = "Traceback (most recent call last):"


def pages_with_rendered_errors(pages: list[Path]) -> list[Path]:
    offending = []
    for path in pages:
        text = re.sub(r"<[^>]+>", "", path.read_text(encoding="utf-8"))
        if ERROR_MARKER in text:
            offending.append(path)
    return offending


def main() -> None:
    for derive_command in (["docs/record.py", "--derive-only"], ["docs/jupyterlite.py"]):
        derive = subprocess.run([sys.executable, *derive_command], cwd=REPOSITORY_ROOT, check=False)
        if derive.returncode:
            raise SystemExit(derive.returncode)

    build = subprocess.run([sys.executable, "-m", "zensical", "build", "--strict"], cwd=REPOSITORY_ROOT, check=False)
    if build.returncode:
        raise SystemExit(build.returncode)

    pages = sorted(SITE_DIR.rglob("*.html"))
    if not pages:
        raise SystemExit("zensical reported success but built no pages. Check zensical.toml.")
    offending = pages_with_rendered_errors(pages)
    if offending:
        listing = ", ".join(str(path.relative_to(REPOSITORY_ROOT)) for path in offending)
        raise SystemExit(f"A code block failed during the build; its traceback is rendered into: {listing}")
    print(f"No rendered code block errors in {len(pages)} pages.")


if __name__ == "__main__":
    main()
