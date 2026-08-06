from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from mudgym.envs.specs import BYTE_DTYPE
from mudgym.logs import get_logger

from .field import ObservationField

DEFAULT_MAX_BYTES = 16384

logger = get_logger(__name__)


class RawBytesField(ObservationField):
    """
    Returns the raw bytes from the game response, including ANSI escape codes, prompt markers
    and line breaks as a fixed-size uint8 numpy array.
    """

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES, include_keys: Sequence[str] | None = None):
        # max_bytes shapes space(), so it must be set before the base validates include_keys
        self.max_bytes = max_bytes
        super().__init__(include_keys=include_keys)

    def full_space(self) -> dict[str, spaces.Space]:
        return {
            "raw_bytes": spaces.Box(0, 255, shape=(self.max_bytes,), dtype=BYTE_DTYPE),
        }

    def full_empty(self) -> dict[str, Any]:
        return {
            "raw_bytes": np.zeros(self.max_bytes, dtype=BYTE_DTYPE),
        }

    def full_extract(self, chunks: Sequence[bytes], **context: Any) -> dict[str, Any]:
        # raw bytes wants the exact wire output, so re-join the per-command chunks
        payload = b"".join(chunks)

        if len(payload) > self.max_bytes:
            logger.warning(
                "field.raw_bytes.truncated",
                payload_bytes=len(payload),
                max_bytes=self.max_bytes,
            )
            payload = payload[: self.max_bytes]

        raw_array = np.zeros(self.max_bytes, dtype=BYTE_DTYPE)
        raw_array[: len(payload)] = np.frombuffer(payload, dtype=BYTE_DTYPE)
        return {"raw_bytes": raw_array}
