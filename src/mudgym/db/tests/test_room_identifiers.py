"""The room db pins the identifier policy: every real room id and name fits the declared spaces."""

from mudgym.db.rooms import ROOM_NAME_BY_ID
from mudgym.envs.specs import (
    IDENTIFIER_CHARSET,
    ROOM_ID_MAX_LENGTH,
    ROOM_NAME_MAX_LENGTH,
    SINGLE_LINE_CHARSET,
)


def test_every_room_id_fits_the_identifier_space():
    for room_id in ROOM_NAME_BY_ID:
        assert len(room_id) <= ROOM_ID_MAX_LENGTH, room_id
        assert set(room_id) <= set(IDENTIFIER_CHARSET), room_id


def test_every_room_name_fits_the_single_line_space():
    for room_name in ROOM_NAME_BY_ID.values():
        assert len(room_name) <= ROOM_NAME_MAX_LENGTH, room_name
        assert set(room_name) <= set(SINGLE_LINE_CHARSET), room_name
