from gymnasium import spaces

from mudgym.notebooks import show_space


def test_a_dict_space_shows_a_row_per_key_without_text_character_sets():
    table = show_space(
        spaces.Dict(
            {
                "room_id_index": spaces.Discrete(1127),
                "here": spaces.Sequence(spaces.Text(96)),
            }
        )
    ).data

    assert "Discrete(1127)" in table
    assert "Sequence(Text(max_length=96))" in table
    assert "charset" not in table


def test_any_other_space_shows_as_one_row():
    assert "Discrete(14)" in show_space(spaces.Discrete(14)).data
