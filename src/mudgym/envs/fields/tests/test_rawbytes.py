import numpy as np
import numpy.testing as npt

from mudgym.envs.fields.rawbytes import DEFAULT_MAX_BYTES, RawBytesField
from mudgym.envs.specs import BYTE_DTYPE

RAW_BYTES_PAYLOAD = (
    b"dance,fes,fex,fei\r\n\x1b[0;33;40mOK, Janet the protector \x1b[1;33;40mdances.\x1b[0;33;40m\x1b[1;37;40m\r\n"
)


def test_default_capacity_is_16384_bytes():
    assert DEFAULT_MAX_BYTES == 16384
    assert RawBytesField().space()["raw_bytes"].shape == (16384,)


def test_pads_and_preserves_prefix():
    out = RawBytesField().extract([RAW_BYTES_PAYLOAD])
    arr = out["raw_bytes"]

    assert arr.dtype == BYTE_DTYPE
    assert arr.shape == (DEFAULT_MAX_BYTES,)

    npt.assert_array_equal(arr[: len(RAW_BYTES_PAYLOAD)], np.frombuffer(RAW_BYTES_PAYLOAD, dtype=BYTE_DTYPE))
    assert arr[len(RAW_BYTES_PAYLOAD)] == 0


def test_truncates_long_input():
    out = RawBytesField(max_bytes=10).extract([b"A" * 100])
    arr = out["raw_bytes"]

    assert arr.shape == (10,)
    assert (arr == ord("A")).all()


def test_empty_returns_valid_defaults():
    defaults = RawBytesField().empty()

    assert defaults["raw_bytes"].dtype == BYTE_DTYPE
    assert defaults["raw_bytes"].shape == (DEFAULT_MAX_BYTES,)
    assert np.all(defaults["raw_bytes"] == 0)
