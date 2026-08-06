# LLM players

The `text` observation and `text` action space put a model in the loop with no wrappers. Full notebook: [`examples/llms.py`](https://github.com/rolo/mudgym/blob/main/examples/llms.py).

Models like to explain themselves, and the environment takes one line, so keep the first non-empty one.

```python
--8<-- "examples/llms.py:llm-command"
```

## Playing an episode

```python
from openai import OpenAI

from mudgym import make_env
from mudgym.notebooks import show_game_tabs

--8<-- "examples/llms.py:llm-episode"
```

The endpoint is any OpenAI-compatible server, seeded from `MUDGYM_LLM_BASE_URL`, `MUDGYM_LLM_MODEL`, and `MUDGYM_LLM_API_KEY`.
