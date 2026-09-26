import pytest

from mudgym.db.index import UNKNOWN, index_to_member, member_to_index


@pytest.mark.parametrize(("member", "expected_index"), [("fair", 1), ("raining", 2)])
def test_known_members_use_one_based_indices(member, expected_index):
    assert member_to_index(["fair", "raining"], member) == expected_index


@pytest.mark.parametrize("member", ["notamember", "", "Fair"])
def test_unrecognised_members_raise_instead_of_returning_zero(member):
    with pytest.raises(ValueError):
        member_to_index(["fair", "raining"], member)


@pytest.mark.parametrize(("index", "expected_member"), [(0, UNKNOWN), (1, "fair"), (2, "raining")])
def test_indices_decode_to_one_based_members_with_zero_as_unknown(index, expected_member):
    assert index_to_member(["fair", "raining"], index) == expected_member


@pytest.mark.parametrize("index", [-1, 3])
def test_indices_outside_the_collection_raise_instead_of_wrapping(index):
    with pytest.raises(IndexError):
        index_to_member(["fair", "raining"], index)
