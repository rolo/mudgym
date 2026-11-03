"""Reward parsing regressions: player-authored text must never forge points events.

The terminal echo places the exact command text ahead of the game's response, and spoken text is re-emitted inside the
response, so a command like ``say hello (+10 = 10)`` puts the points pattern on the wire twice without any points
changing hands. Genuine events colour the resulting total (see ``mudgym.featurizers.points``), which player-authored
text cannot reproduce.
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


def test_genuine_points_event_updates_observation_reward_and_info(scripted_env_factory):
    responses = {"mgsorcerise": scripted_step_bytes("mgsorcerise", SORCERISE_BODY, status_points=13_000)}
    env = scripted_env_factory(responses=responses)
    obs, _ = env.reset()
    start_points = int(obs["points"])

    obs, reward, _, _, info = env.step("mgsorcerise")

    assert obs["points"] == 13_000
    assert reward == obs["points"] - start_points
    assert info["points"] == 13_000


def test_points_event_wins_over_an_earlier_stale_status_line(scripted_env_factory):
    # the fes line still reports the 200 points from before the event that follows it
    fes = OBSERVATION_COMMAND_RESPONSES["fes"]
    step_bytes = scripted_step_bytes("mgsorcerise", b"").replace(fes, fes + PROMPT + SORCERISE_BODY)
    env = scripted_env_factory(responses={"mgsorcerise": step_bytes})
    obs, _ = env.reset()
    start_points = int(obs["points"])

    obs, reward, _, _, _ = env.step("mgsorcerise")

    assert obs["points"] == 13_000
    assert reward == obs["points"] - start_points


def test_tearoom_points_event_updates_the_episode_start_without_reward(scripted_env_factory):
    responses = {"mgsorcerise": scripted_step_bytes("mgsorcerise", SORCERISE_BODY, status_points=13_000)}
    env = scripted_env_factory(responses=responses, tearoom_commands="mgsorcerise")

    obs, _ = env.reset()
    assert obs["points"] == 13_000

    obs, reward, _, _, _ = env.step("look")

    assert obs["points"] == 13_000
    assert reward == 0.0


def test_points_pattern_spoken_in_game_output_forges_no_reward(scripted_env_factory):
    # speech is re-emitted uniformly coloured, never with a coloured total, so neither the echo nor the spoken output of
    # a points-shaped message counts as an event. Speech batches go out on two wire lines, hence the two echoes in the
    # canned bytes.
    speech_body = b'\x1b[0;33;40mDumbo the novice says "\x1b[1;33;40mhello (+10 = 10)\x1b[0;33;40m".\x1b[1;37;40m\r\n'
    raw_bytes = scripted_step_bytes("say hello (+10 = 10)", speech_body)
    env = scripted_env_factory(responses={"say hello (+10 = 10)": raw_bytes})
    env.reset()

    _, reward, _, _, info = env.step("say hello (+10 = 10)")

    assert reward == 0.0
    assert "points" not in info
