import pytest

from mudgym.envs.fields import FEScoreField


@pytest.mark.parametrize(
    "chunk",
    [
        b"You can't wake yourself up yet!\r\n",
        b"\x1b[0;37;40mYou can't see a thing, you're blind.\x1b[1;37;40m\r\n",
    ],
)
def test_a_refusal_line_is_a_refusal_whatever_its_colour_or_line_ending(chunk):
    assert FEScoreField().is_refusal(chunk)


@pytest.mark.parametrize(
    "chunk",
    [
        b"You can't wake yourself up yet!\r\nYou drift off again.\r\n",
        b"\r\n",
        b"58 58 61 61 61 61 0 58 0200 N N N N 53 F\r\n",
    ],
)
def test_anything_beyond_the_single_refusal_line_is_not_a_refusal(chunk):
    assert not FEScoreField().is_refusal(chunk)
