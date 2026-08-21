"""Reward parsing regressions: player-authored text must never forge points events.

The terminal echo places the exact command text ahead of the game's response, and spoken text is
re-emitted inside the response, so a command like ``say hello (+10 = 10)`` puts the points pattern
on the wire twice without any points changing hands. Genuine events colour the resulting total
(see ``mudgym.featurizers.points``), which player-authored text cannot reproduce.
"""

from tests.scripted import OBSERVATION_COMMAND_RESPONSES, PROMPT

# Captured from the live game: mgsorcerise response (temporary sorcerer status award).
SORCERISE_BODY = (
    b"Raymond the protector bestows temporary sorcerer status upon you.\r\n"
    b"You have changed experience level from seer to sorcerer.\r\n"
    b"(Persona saved on +12,800 = \x1b[0;32;40m13,000\x1b[1;37;40m).\r\n"
)


def scripted_step_bytes(action: str, body: bytes, *, status_points: int = 200) -> bytes:
    """Echo line, response body, then the observation-command responses, prompt-delimited."""
    observation_line = "sql,fes,fex,fei"
    parts = [action.encode("ascii"), b"\r\n", body, PROMPT, observation_line.encode("ascii"), b"\r\n"]
    for position, observation_command in enumerate(observation_line.split(",")):
        if position:
            parts.append(PROMPT)
        response = OBSERVATION_COMMAND_RESPONSES[observation_command]
        if observation_command == "fes":
            response = response.replace(b"0200", f"{status_points:04d}".encode("ascii"))
        parts.append(response)
    parts.append(PROMPT)
    return b"".join(parts)


def scripted_step_with_event_after_fes(action: str, body: bytes) -> bytes:
    """Build a complete response with a points event after fes."""
    responses = OBSERVATION_COMMAND_RESPONSES
    return (
        action.encode("ascii")
        + b"\r\n"
        + PROMPT
        + b"sql,fes,fex,fei\r\n"
        + responses["sql"]
        + PROMPT
        + responses["fes"]
        + PROMPT
        + body
        + PROMPT
        + responses["fex"]
        + PROMPT
        + responses["fei"]
        + PROMPT
    )


def test_points_pattern_in_command_echo_forges_no_reward(scripted_env_factory):
    env = scripted_env_factory()
    env.reset()

    obs, reward, terminated, truncated, info = env.step("look (+10 = 10)")

    assert reward == 0.0
    assert "points" not in info


def test_genuine_points_event_rewards(scripted_env_factory):
    responses = {"mgsorcerise": scripted_step_bytes("mgsorcerise", SORCERISE_BODY, status_points=13_000)}
    env = scripted_env_factory(responses=responses)
    initial_observation, _ = env.reset()

    observation, reward, _, _, info = env.step("mgsorcerise")

    assert observation["points"] == 13_000
    assert reward == observation["points"] - initial_observation["points"]
    assert info["points"] == 13_000


def test_points_event_wins_over_an_earlier_stale_status_line(scripted_env_factory):
    responses = {"mgsorcerise": scripted_step_with_event_after_fes("mgsorcerise", SORCERISE_BODY)}
    env = scripted_env_factory(responses=responses)
    initial_observation, _ = env.reset()

    observation, reward, _, _, info = env.step("mgsorcerise")

    assert observation["points"] == 13_000
    assert reward == observation["points"] - initial_observation["points"]
    assert info["points"] == 13_000


def test_spoken_points_total_does_not_forge_score_metadata(scripted_env_factory):
    # A plain parenthesised number carries no raw-wire signal distinguishing it from player text,
    # so it must not become reward, termination, or score metadata.
    body = b'Dumbo the novice says "\x1b[1;33;40m(2000000)\x1b[0;33;40m".\r\n'
    responses = {"look": scripted_step_bytes("look", body)}
    env = scripted_env_factory(responses=responses)
    env.reset()

    obs, reward, terminated, truncated, info = env.step("look")

    assert terminated is False
    assert reward == 0.0
    assert "points" not in info


def test_points_pattern_spoken_in_game_output_forges_no_reward(scripted_env_factory):
    # speech is re-emitted uniformly coloured, never with a coloured total, so neither the echo
    # nor the spoken output of a points-shaped message counts as an event. Speech batches go out
    # on two wire lines, hence the two echoes in the canned bytes.
    speech_body = b'\x1b[0;33;40mDumbo the novice says "\x1b[1;33;40mhello (+10 = 10)\x1b[0;33;40m".\x1b[1;37;40m\r\n'
    raw_bytes = (
        b"say hello (+10 = 10)\r\n"
        + speech_body
        + PROMPT
        + b"sql,fes,fex,fei\r\n"
        + PROMPT.join(OBSERVATION_COMMAND_RESPONSES[command] for command in ["sql", "fes", "fex", "fei"])
        + PROMPT
    )
    env = scripted_env_factory(responses={"say hello (+10 = 10)": raw_bytes})
    env.reset()

    obs, reward, terminated, truncated, info = env.step("say hello (+10 = 10)")

    assert reward == 0.0
    assert "points" not in info
