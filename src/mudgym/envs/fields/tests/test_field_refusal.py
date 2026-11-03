import pytest

from mudgym.envs.fields import FEScoreField
from mudgym.envs.fields.tests.payloads import FORD_COLLAPSE_BYTES
from mudgym.featurizers.responses import split_on_echo_lines


@pytest.fixture
def observation_response():
    _, response = split_on_echo_lines(FORD_COLLAPSE_BYTES, ["sql,fes,fex,fei"])
    return response


def test_a_refusal_followed_by_other_game_output_is_not_a_refusal(observation_response):
    assert not FEScoreField().is_refusal(observation_response)


def test_recognises_a_wake_up_with_crlf_line_endings():
    assert FEScoreField().is_refusal(b"You wake up!\r\nYour stamina is 32.\r\n")
