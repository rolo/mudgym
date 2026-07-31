"""
Inject game output into docs.

    <!-- transcript: observation-parsed -->
    ...generated, do not edit by hand...
    <!-- /transcript: observation-parsed -->

Run with ``just docs-transcripts``

Write a function decorated with ``@transcript`` and drop the matching marker pair into a page.
"""

import re
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mudgym import make_env, make_parallel_env
from mudgym.notebooks import show_ansi

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# A marker pair delimits one generated region. The name ties the region to the
# transcript function that fills it.
MARKER_PATTERN = re.compile(
    r"<!-- transcript: (?P<name>[a-z0-9-]+) -->"
    r".*?"
    r"<!-- /transcript: (?P=name) -->",
    re.DOTALL,
)

TRANSCRIPTS: dict[str, Callable[[], str]] = {}


def transcript(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
    """Register a function as the source for the ``name`` marker region."""

    def register(builder: Callable[[], str]) -> Callable[[], str]:
        if name in TRANSCRIPTS:
            raise ValueError(f"Duplicate transcript name: {name!r}")
        TRANSCRIPTS[name] = builder
        return builder

    return register


def code_block(body: str, language: str = "text") -> str:
    """Wrap text in a fenced code block, trimming trailing whitespace per line."""

    lines = [line.rstrip() for line in body.strip().splitlines()]
    return f"```{language}\n" + "\n".join(lines) + "\n```"


def game_screen(info: Mapping[str, Any]) -> str:
    """Render the true game screen, ANSI colours and all, as inline HTML."""

    return show_ansi(info["render_bytes"]).data


def format_value(value: Any) -> str:
    """Render one observation value compactly enough for a docs table cell."""

    if isinstance(value, np.ndarray):
        return f"`{np.array2string(value, separator=', ', threshold=24)}`"
    if isinstance(value, tuple):
        return ", ".join(f"`{item}`" for item in value) if value else "_empty_"
    return f"`{value}`"


def observation_table(observation: Mapping[str, Any], skip: Iterable[str] = ("text",)) -> str:
    """Render an observation dict as a markdown key/value table."""

    skipped = set(skip)
    rows = [f"| `{key}` | {format_value(value)} |" for key, value in observation.items() if key not in skipped]
    return "\n".join(["| Key | Value |", "|---|---|", *rows])


@transcript("quickstart-single")
def quickstart_single() -> str:
    """The single-agent quickstart on the landing page, printed output and screen."""

    env = make_env(observation="parsed")
    try:
        env.reset()
        observation, reward, _terminated, _truncated, info = env.step("howl")
        printed = code_block(f"{observation['room_name']} {reward}")
        return "\n\n".join([printed, game_screen(info)])
    finally:
        env.close()


@transcript("quickstart-multiagent")
def quickstart_multiagent() -> str:
    """The two-agent quickstart: what each agent saw for the same tick."""

    env = make_parallel_env(agents=2)
    try:
        env.reset()
        actions = dict.fromkeys(env.agents, "yodel")
        observations, rewards, _terminations, _truncations, infos = env.step(actions)
        printed = code_block(
            "\n".join(f"{agent} {observations[agent]['room_name']} {rewards[agent]}" for agent in sorted(observations))
        )
        screens = [f"**{agent}**\n\n{game_screen(infos[agent])}" for agent in sorted(infos)]
        return "\n\n".join([printed, *screens])
    finally:
        env.close()


@transcript("observation-text")
def observation_text() -> str:
    env = make_env(observation="text")
    try:
        observation, info = env.reset()
        return "\n\n".join([code_block(observation["text"]), game_screen(info)])
    finally:
        env.close()


@transcript("observation-parsed")
def observation_parsed() -> str:
    env = make_env(observation="parsed")
    try:
        observation, info = env.reset()
        return "\n\n".join([observation_table(observation), game_screen(info)])
    finally:
        env.close()


@transcript("observation-cheats")
def observation_cheats() -> str:
    env = make_env(observation="cheats")
    try:
        observation, info = env.reset()
        return "\n\n".join([observation_table(observation), game_screen(info)])
    finally:
        env.close()


@transcript("observation-bytes")
def observation_bytes() -> str:
    env = make_env(observation="bytes")
    try:
        observation, info = env.reset()
        returned_bytes = bytes(observation["raw_bytes"][: info["bytes_length"]])
        return "\n\n".join([code_block(repr(returned_bytes[:320]) + "\n...", "python"), game_screen(info)])
    finally:
        env.close()


@transcript("actions-directions")
def actions_directions() -> str:
    """Show the discrete vocabulary alongside the exit mask that indexes it."""

    env = make_env(observation="parsed", actions="directions")
    try:
        observation, _info = env.reset()
        commands = env.commands  # type: ignore[attr-defined]
        mask = observation["available_exits"]
        rows = [
            f"| {index} | `{command}` | {'yes' if bit else 'no'} |"
            for index, (command, bit) in enumerate(zip(commands, mask, strict=True))
        ]
        return "\n".join(["| Index | Command | Available here |", "|---|---|---|", *rows])
    finally:
        env.close()


@transcript("actions-step")
def actions_step() -> str:
    """A single text action and the screen it produced."""

    env = make_env(observation="parsed")
    try:
        env.reset()
        observation, reward, terminated, truncated, info = env.step("look")
        summary = code_block(
            f"room_name  {observation['room_name']}\n"
            f"reward     {reward}\n"
            f"terminated {terminated}\n"
            f"truncated  {truncated}",
        )
        return "\n\n".join([summary, game_screen(info)])
    finally:
        env.close()


def inject_into_page(path: Path, generated: Mapping[str, str]) -> list[str]:
    """Replace every marker region in one page. Returns the names it filled."""

    source = path.read_text()
    filled: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in generated:
            raise KeyError(
                f"{path.name} has a marker for unknown transcript {name!r}. "
                f"Known transcripts: {', '.join(sorted(generated))}"
            )
        filled.append(name)
        return f"<!-- transcript: {name} -->\n{generated[name]}\n<!-- /transcript: {name} -->"

    updated = MARKER_PATTERN.sub(replace, source)
    if updated != source:
        path.write_text(updated)
    return filled


def main() -> int:
    generated = {}
    for name, builder in TRANSCRIPTS.items():
        print(f"Capturing {name} ...", flush=True)
        generated[name] = builder()

    filled: list[str] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        filled.extend(inject_into_page(path, generated))

    unused = sorted(set(generated) - set(filled))
    if unused:
        # a transcript with no marker is worth noticing but should not break a docs build.
        print(f"warning: no marker found for: {', '.join(unused)}", file=sys.stderr)

    print(f"Injected {len(filled)} transcript(s) into {DOCS_DIR}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
