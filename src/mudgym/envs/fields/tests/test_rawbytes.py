import numpy as np
import pytest

from mudgym.envs.fields.rawbytes import RawBytesField
from mudgym.envs.fields.tests.helpers import assert_valid_observation
from mudgym.envs.specs import BYTE_DTYPE


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(b"abc", b"abc" + b"\x00" * 7, id="shorter-than-capacity"),
        pytest.param(b"0123456789", b"0123456789", id="exact-capacity"),
        pytest.param(b"0123456789abcdef", b"0123456789", id="longer-than-capacity-keeps-prefix"),
    ],
)
def test_extracts_prefix_and_zero_pads_to_capacity(raw, expected):
    field = RawBytesField(max_bytes=10)
    obs = field.extract([raw])

    np.testing.assert_equal(obs, {"raw_bytes": np.frombuffer(expected, dtype=BYTE_DTYPE)})
    assert_valid_observation(field, obs)
