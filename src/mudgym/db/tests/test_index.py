import pytest

from mudgym.db.index import member_to_index


@pytest.mark.parametrize(("member", "expected_index"), [("fair", 1), ("raining", 2)])
def test_known_members_use_one_based_indices(member, expected_index):
    assert member_to_index(["fair", "raining"], member) == expected_index


@pytest.mark.parametrize("member", ["notamember", "", "Fair"])
def test_unrecognised_members_raise_instead_of_returning_zero(member):
    with pytest.raises(ValueError):
        member_to_index(["fair", "raining"], member)