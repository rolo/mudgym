import pytest

from mudgym.envs.fields import FEScoreField
from mudgym.envs.fields.tests.payloads import FORD_COLLAPSE_BYTES, VAMPIRE_BLIND_BYTES
from mudgym.featurizers.responses import split_on_echo, split_on_prompt


@pytest.fixture(
    params=[
        pytest.param(FORD_COLLAPSE_BYTES, id="unconscious"),
        pytest.param(VAMPIRE_BLIND_BYTES, id="blind"),
    ]
)
def observation_response(request):
    _, response = split_on_echo(request.param, "sql,fes,fex,fei")
    return response


def test_recognises_a_refusal_from_real_game_bytes(observation_response):
    refusal, *_ = split_on_prompt(observation_response)

    assert FEScoreField().is_refusal(refusal)


def test_a_refusal_followed_by_other_game_output_is_not_a_refusal(observation_response):
    assert not FEScoreField().is_refusal(observation_response)


def test_other_field_responses_are_not_refusals(observation_response):
    _, *responses = split_on_prompt(observation_response)

    for response in responses:
        assert not FEScoreField().is_refusal(response)
