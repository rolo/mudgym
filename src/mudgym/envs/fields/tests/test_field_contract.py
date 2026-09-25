from functools import partial

import numpy as np
import pytest

from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.envs.fields.fescore import FEScoreField
from mudgym.envs.fields.fexits import FEXitsField
from mudgym.envs.fields.mgcheats import MGCheatsField
from mudgym.envs.fields.rawbytes import RawBytesField
from mudgym.envs.fields.superquicklook import SuperQuickLookField
from mudgym.envs.fields.tests.helpers import assert_valid_observation


@pytest.mark.parametrize(
    "field_constructor",
    [
        MGCheatsField,
        FEInventoryField,
        FEScoreField,
        FEXitsField,
        RawBytesField,
        pytest.param(partial(RawBytesField, max_bytes=100), id="RawBytesField-custom-capacity"),
        SuperQuickLookField,
    ],
)
def test_empty_observation_satisfies_field_contract(field_constructor):
    field = field_constructor()
    assert_valid_observation(field, field.empty())


@pytest.mark.parametrize(
    ("kwargs", "expected_keys"),
    [
        pytest.param({}, {"vitals", "flags", "reset_minutes", "weather", "weather_index"}, id="omitted"),
        pytest.param(
            {"include_keys": None}, {"vitals", "flags", "reset_minutes", "weather", "weather_index"}, id="default"
        ),
        pytest.param({"include_keys": ("vitals",)}, {"vitals"}, id="one-key"),
        pytest.param({"include_keys": ()}, set(), id="no-keys"),
    ],
)
def test_include_keys_filters_public_methods_but_not_full_methods(kwargs, expected_keys):
    # Every field inherits the filtering interface unchanged from ObservationField.
    field = FEScoreField(**kwargs)
    unfiltered = FEScoreField()
    chunks = [b"58 78 41 61 32 52 7 68 0200 N N N N 53 F"]

    assert field.full_space() == unfiltered.space()
    np.testing.assert_equal(field.full_empty(), unfiltered.empty())
    np.testing.assert_equal(field.full_extract(chunks), unfiltered.extract(chunks))

    assert field.space() == {key: space for key, space in unfiltered.space().items() if key in expected_keys}
    np.testing.assert_equal(
        field.empty(), {key: value for key, value in unfiltered.empty().items() if key in expected_keys}
    )
    np.testing.assert_equal(
        field.extract(chunks), {key: value for key, value in unfiltered.extract(chunks).items() if key in expected_keys}
    )


def test_include_keys_must_be_space_keys():
    with pytest.raises(ValueError, match="not in full_space"):
        FEScoreField(include_keys=("vitals", "bogus"))
