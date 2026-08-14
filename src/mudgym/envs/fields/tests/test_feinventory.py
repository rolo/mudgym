import pytest

from mudgym.envs.fields.feinventory import INVENTORY_DIVIDER, FEInventoryField


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        (b"streetsign\r\n========\r\n", {"portables": ("streetsign",), "inventory": ()}),
        (b"========\r\nambrosia\r\n", {"portables": (), "inventory": ("ambrosia",)}),
        (b"necklace0\r\n========\r\n", {"portables": ("necklace0",), "inventory": ()}),
        (b"oo\r\n========\r\n", {"portables": (), "inventory": ()}),
    ],
)
def test_extracts_fei_chunk(chunk, expected):
    obs = FEInventoryField().extract([chunk])

    assert obs == expected
    divider = INVENTORY_DIVIDER.decode("ascii")
    assert all(divider not in item for item in obs["portables"])
    assert all(divider not in item for item in obs["inventory"])


def test_handles_no_separator():
    obs = FEInventoryField().extract([b"x"])
    assert obs == {"portables": (), "inventory": ()}


def test_dark_room_portables_dropped():
    # in the dark the room's portables show as the 'oo' marker; the divider and the player's own inventory
    # below it still show, so portables come back empty while inventory still parses.
    # the inventory value mirrors the real fei grammar: identifiers, never descriptive phrases
    raw = b"oo\r\n" + INVENTORY_DIVIDER + b"\r\nbroadsword\r\n"
    obs = FEInventoryField().extract([raw])
    assert obs["portables"] == ()
    assert obs["inventory"] == ("broadsword",)
    assert FEInventoryField().full_space()["inventory"].contains(obs["inventory"])


def test_empty_returns_valid_defaults():
    defaults = FEInventoryField().empty()

    assert defaults["portables"] == ()
    assert defaults["inventory"] == ()


def test_extracts_fei_from_real_captures(bytes_case):
    obs = FEInventoryField().extract(bytes_case["chunks"])
    expected = bytes_case["fei"]

    assert obs["portables"] == expected["portables"]
    assert obs["inventory"] == expected["inventory"]
