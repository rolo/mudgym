import pytest

from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.envs.fields.fescore import FEScoreField
from mudgym.envs.fields.fexits import FEXitsField
from mudgym.envs.fields.mgcheats import MGCheatsField
from mudgym.envs.fields.rawbytes import RawBytesField
from mudgym.envs.fields.superquicklook import SuperQuickLookField
from mudgym.envs.validation import validate_field_spaces

ALL_FIELDS = [
    MGCheatsField(),
    FEInventoryField(),
    FEScoreField(),
    FEXitsField(),
    RawBytesField(),
    RawBytesField(max_bytes=100),
    SuperQuickLookField(),
]


@pytest.mark.parametrize("field", ALL_FIELDS, ids=lambda f: f.__class__.__name__)
def test_space_keys_match_empty_keys(field):
    """Every field's space() keys must match its empty() keys."""
    validate_field_spaces([field])


@pytest.mark.parametrize("field", ALL_FIELDS, ids=lambda f: f.__class__.__name__)
def test_empty_values_satisfy_spaces(field):
    """Every field's empty() values must be valid for their declared spaces."""
    spaces = field.space()
    defaults = field.empty()

    for key, space in spaces.items():
        value = defaults[key]
        assert space.contains(value), f"{key}: {value!r} not in {space}"


FES_RESPONSE = b"58 58 61 61 61 61 0 58 0200 N N N N 53 F"


def test_space_empty_and_extract_apply_include_keys():
    field = FEScoreField(include_keys=("points",))
    assert set(field.space()) == {"points"}
    assert set(field.empty()) == {"points"}
    assert set(field.extract([FES_RESPONSE])) == {"points"}


def test_full_methods_keep_every_parser_key():
    field = FEScoreField(include_keys=("points",))
    assert set(field.full_space()) > {"points"}
    assert field.full_space().keys() == field.full_empty().keys()
    assert field.full_extract([FES_RESPONSE]).keys() == field.full_space().keys()


def test_space_empty_and_extract_match_full_capability_without_include_keys():
    field = FEScoreField()
    assert field.space().keys() == field.full_space().keys()
    assert field.empty().keys() == field.full_empty().keys()
    assert field.extract([FES_RESPONSE]).keys() == field.full_extract([FES_RESPONSE]).keys()


def test_empty_include_keys_contributes_nothing():
    field = FEScoreField(include_keys=())
    assert field.space() == {}
    assert field.empty() == {}
    assert field.extract([FES_RESPONSE]) == {}
