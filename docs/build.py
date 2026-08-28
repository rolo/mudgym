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
    build = subprocess.run([sys.executable, "-m", "zensical", "build", "--strict"], cwd=REPOSITORY_ROOT)
    if build.returncode:
        raise SystemExit(build.returncode)

    pages = sorted(SITE_DIR.rglob("*.html"))
    offending = pages_with_rendered_errors(pages)
    if offending:
        listing = ", ".join(str(path.relative_to(REPOSITORY_ROOT)) for path in offending)
        raise SystemExit(f"A code block failed during the build; its traceback is rendered into: {listing}")
    print(f"No rendered code block errors in {len(pages)} pages.")


if __name__ == "__main__":
    main()
