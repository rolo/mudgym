from mudgym.envs.fields.superquicklook import SuperQuickLookField

COAL_BUNKER_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40mcoal bunker\x1b[0;37;40m" contains '
    b"\x1b[1;36;40mthe coal\x1b[0;37;40m, \x1b[36mthe door\x1b[37m, \x1b[31mJuan the protector\x1b[37m "
    b"and \x1b[35m4 rats\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
    b"The large rat is carrying the following:\r\n"
    b"        nothing.\r\n"
    b"The rat is carrying the following:\r\n"
    b"        nothing.\r\n"
)

KEEP_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40mthird floor of keep\x1b[0;37;40m" contains '
    b"\x1b[1;36;40mthe manuscript\x1b[0;37;40m, \x1b[32mthe flickering haze\x1b[37m, "
    b"\x1b[32mthe wall\x1b[37m, \x1b[31mJuan the protector\x1b[37m and \x1b[36mthe mortar\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
    b"The mortar contains:\r\n"
    b"        powdered dragonblood.\r\n"
)


def test_classifies_coal_bunker_contents():
    obs = SuperQuickLookField().extract([COAL_BUNKER_CHUNK])
    assert obs["room_name"] == "coal bunker"
    assert obs["here"] == ("coal", "door", "Juan the protector", "4 rats")
    assert obs["portables"] == ("coal", "door")
    assert obs["players"] == ("Juan the protector",)
    assert obs["mobiles"] == ("4 rats",)
    assert obs["features"] == ()
    assert obs["inventory"] == ()


TWO_MORTALS_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40mcoal bunker\x1b[0;37;40m" contains '
    b"\x1b[36mthe door\x1b[37m, \x1b[31mDavid the sorcerer\x1b[37m, "
    b"\x1b[31mJessica the protector\x1b[37m, \x1b[35m4 rats\x1b[37m and "
    b"\x1b[1;36;40mthe coal\x1b[0;37;40m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
)

ARCANE_FOREST_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40marcane forest\x1b[0;37;40m" contains '
    b"\x1b[1;35;40mthe dragon\x1b[0;37;40m, \x1b[1;36;40mthe amulet\x1b[0;37;40m, "
    b"\x1b[31mMark the protector\x1b[37m and \x1b[32mthe arcane tree\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
    b"The dragon is carrying the following:\r\n"
    b"        the emerald.\r\n"
)

WIZ_PLAYER_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40mcoal bunker\x1b[0;37;40m" contains '
    b"\x1b[36mthe door\x1b[37m, \x1b[1;31;40mKeyser the archizard\x1b[0;37;40m and "
    b"\x1b[31mDumbo the protector\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
)


def test_classifies_keep_contents():
    obs = SuperQuickLookField().extract([KEEP_CHUNK])
    assert obs["room_name"] == "third floor of keep"
    assert obs["here"] == ("manuscript", "flickering haze", "wall", "Juan the protector", "mortar")
    assert obs["features"] == ("flickering haze", "wall")
    assert obs["portables"] == ("manuscript", "mortar")
    assert obs["players"] == ("Juan the protector",)
    assert obs["mobiles"] == ()


def test_classifies_both_mortals_as_players():
    obs = SuperQuickLookField().extract([TWO_MORTALS_CHUNK])
    assert obs["players"] == ("David the sorcerer", "Jessica the protector")
    assert obs["mobiles"] == ("4 rats",)
    assert obs["portables"] == ("door", "coal")


def test_classifies_every_category_in_one_line():
    obs = SuperQuickLookField().extract([ARCANE_FOREST_CHUNK])
    assert obs["room_name"] == "arcane forest"
    assert obs["mobiles"] == ("dragon",)
    assert obs["portables"] == ("amulet",)
    assert obs["players"] == ("Mark the protector",)
    assert obs["features"] == ("arcane tree",)


def test_wiz_player_folds_into_players():
    obs = SuperQuickLookField().extract([WIZ_PLAYER_CHUNK])
    assert obs["players"] == ("Keyser the archizard", "Dumbo the protector")
    assert obs["portables"] == ("door",)
    assert obs["mobiles"] == ()
