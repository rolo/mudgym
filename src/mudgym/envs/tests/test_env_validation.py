"""Field-space validation contract.

`validate_field_spaces` runs during MudEnv construction and is also usable standalone in tests and tooling. It
checks that each field's space() and empty() agree on their key set, and that no two fields provide the same
observation key after include_keys filtering.
"""

from typing import Any

import pytest
from gymnasium import spaces

from mudgym.envs.fields import MGCheatsField, ObservationField, SuperQuickLookField
from mudgym.envs.validation import validate_field_spaces


class MismatchedKeysField(ObservationField):
    """Deliberately broken field whose space() and empty() disagree on their key set."""

    def full_space(self) -> dict[str, spaces.Space]:
        return {"declared": spaces.Discrete(2)}

    def full_empty(self) -> dict[str, Any]:
        return {"other": 0}

    def full_extract(self, raw_bytes: bytes) -> dict[str, Any]:
        return {}


def test_validate_field_spaces_rejects_mismatched_space_and_empty_keys():
    with pytest.raises(ValueError, match=r"space\(\) keys.*empty\(\) keys"):
        validate_field_spaces([MismatchedKeysField()])


def test_validate_field_spaces_rejects_duplicate_keys_across_fields():
    with pytest.raises(ValueError, match="provided by both"):
        validate_field_spaces([SuperQuickLookField, MGCheatsField])
