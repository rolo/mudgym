import re

from mudgym.notebooks.style import THEME, token


def test_every_theme_token_falls_back_to_literal_values():
    for name in THEME:
        assert not re.search(r"var\(--notebook-[a-z-]+\)", token(name)), name
