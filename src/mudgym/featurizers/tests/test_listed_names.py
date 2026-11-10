from mudgym.featurizers.listed_names import split_listed_names


def test_a_list_splits_on_its_separator():
    assert split_listed_names("necklace0, road", ", ") == ["necklace0", "road"]
    assert split_listed_names("the coal,the door", ",") == ["the coal", "the door"]


def test_the_name_holding_a_comma_stays_whole_wherever_it_is_listed():
    assert split_listed_names("sturdy, giant door", ", ") == ["sturdy, giant door"]
    assert split_listed_names("wall, sturdy, giant door, sofa", ", ") == ["wall", "sturdy, giant door", "sofa"]
    assert split_listed_names("the wall, the sturdy, giant door", ",") == ["the wall", " the sturdy, giant door"]
