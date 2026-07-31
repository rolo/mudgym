"""Speech commands and the auto command batch.

A speech verb consumes the rest of its wire line, so the env's session sends the auto commands on
their own line for speech actions. The step must come back complete, with the field responses
claimed positionally and both echo lines removed from the text observation.
"""

import pytest


def test_say_step_completes_with_fields_populated(scripted_env_factory):
    env = scripted_env_factory()
    env.reset()

    obs, reward, terminated, truncated, info = env.step("say hello")

    assert truncated is False
    assert terminated is False
    connection = env.unwrapped.session.connection
    assert connection.sent_lines[-1] == ["say hello", "sql,fes,fex,fei"]
    # the auto command responses were claimed into fields, not swallowed into the speech
    assert list(obs["available_exits"]).count(1) > 0
    assert "say hello" not in obs["text"]
    assert "sql,fes,fex,fei" not in obs["text"]


def test_speech_response_lands_in_the_text_observation(scripted_env_factory):
    env = scripted_env_factory()
    env.reset()

    obs, reward, terminated, truncated, info = env.step("say hello")

    # the speech line's re-emitted body arrives ahead of the autos line and must stay in the text
    assert 'says "hello"' in obs["text"]


def test_converse_actions_are_rejected(scripted_env_factory):
    # converse mode would turn every later line into speech, autos included, desynchronising the
    # transport for the rest of the episode
    env = scripted_env_factory()
    env.reset()

    with pytest.raises(ValueError, match="converse"):
        env.step("converse rat")
    with pytest.raises(ValueError, match="converse"):
        env.step("get sword,converse rat")


def test_plain_commands_still_use_the_single_line_wire_format(scripted_env_factory):
    env = scripted_env_factory()
    env.reset()

    obs, reward, terminated, truncated, info = env.step("look")

    assert truncated is False
    connection = env.unwrapped.session.connection
    assert connection.sent_lines[-1] == ["look,sql,fes,fex,fei"]
    assert connection.commands[-1] == "look,sql,fes,fex,fei"
