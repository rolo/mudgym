import numpy as np
import pytest

from mudgym.envs.fields.rawbytes import DEFAULT_MAX_BYTES, RawBytesField
from mudgym.envs.fields.tests.helpers import assert_valid_observation
from mudgym.envs.specs import BYTE_DTYPE

RAW_BYTES_PAYLOAD = (
    b"dance,fes,fex,fei\r\n\x1b[0;33;40mOK, Janet the protector \x1b[1;33;40mdances.\x1b[0;33;40m\x1b[1;37;40m\r\n"
)


def test_default_capacity_is_16384_zero_bytes():
    field = RawBytesField()

    assert DEFAULT_MAX_BYTES == 16384
    assert field.space()["raw_bytes"].shape == (16384,)
    np.testing.assert_equal(field.empty(), {"raw_bytes": np.zeros(16384, dtype=BYTE_DTYPE)})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(b"", b"\x00" * 10, id="empty"),
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


def test_preserves_wire_bytes_and_zeroes_the_entire_padding_suffix():
    field = RawBytesField()
    obs = field.extract([RAW_BYTES_PAYLOAD])

    np.testing.assert_equal(
        obs["raw_bytes"][: len(RAW_BYTES_PAYLOAD)], np.frombuffer(RAW_BYTES_PAYLOAD, dtype=BYTE_DTYPE)
    )
    np.testing.assert_equal(obs["raw_bytes"][len(RAW_BYTES_PAYLOAD) :], 0)
    assert_valid_observation(field, obs)
