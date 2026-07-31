"""Captured real game bytes with documented expected parsed values."""

DALLY_LANE_BYTES = (
    b"move north,fes,fex,fei\r\nAs you step through the opening, you become swathed in a fine, gossamer mist. "
    b"The Elizabethan tearoom fades hazily away, and vague, new shapes begin to form around you. Their outlines "
    b"become more defined, their colours grow stronger, and the mist thins out into pale wisps, which gradually "
    b"disperse away to nothingness...\r\n\x1b[32mDally Lane\x1b[37m.\r\n\x1b[0;32;40mYou are standing on a dusty "
    b"road with rising ground both to the north and south. Though dilapidated and disused, the route north of "
    b"where you stand, with a building at the far end, looks as if it once formed a grand driveway. To the south, "
    b"the road twists up the hill where, at the summit, an ancient walled monastery dominates the scene. Open "
    b"fields lie to the west, and east is a flat area of lawn. \x1b[1;37;40m\x1b[0;32;40mIt is raining. "
    b"\x1b[1;37;40m\x1b[36mA streetsign has fallen here. \x1b[37m\r\n\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
    b"\x1b[1;37;40m\x1b[0;37;40m\x1b[1;32;40m71\x1b[0;37;40m \x1b[1;32;40m71\x1b[0;37;40m 51 51 58 58 0 71 0200 "
    b"N N N N 53 R\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40mup out "
    b"swampward southwest south west east north\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
    b"\x1b[1;37;40m\x1b[0;37;40mstreetsign\r\n========\r\n\x1b[1;37;40m\r\n\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
    b"\x1b[1;37;40m"
)

BADLY_PAVED_BYTES = (
    b"move north,fes,fex,fei\r\nAs you step through the opening, you become swathed in a fine, gossamer mist. "
    b"The Elizabethan tearoom fades hazily away, and vague, new shapes begin to form around you. Their outlines "
    b"become more defined, their colours grow stronger, and the mist thins out into pale wisps, which gradually "
    b"disperse away to nothingness...\r\n\x1b[32mBadly-paved road\x1b[37m.\r\n\x1b[0;32;40mYou are standing on a "
    b"badly-paved road between a mountain, to the north, and the doorway of a wayside inn, directly to the south. "
    b"East, the road fords a fast-flowing river, and west it continues. To the northeast is a ramshackle old "
    b"building, and southeast there seems to be a well of some kind. \x1b[1;37;40m\x1b[0;36;40mThe door is open. "
    b"\x1b[1;37;40m\r\n\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m\x1b[1;32;40m43\x1b[0;37;40m "
    b"\x1b[1;32;40m43\x1b[0;37;40m 69 69 68 68 0 43 0200 N N N N 53 F\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*"
    b"\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40mup in out swampward southwest south southeast northeast northwest west "
    b"east north\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m========\r\n"
    b"\x1b[1;37;40m\r\n\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
)

JANET_DANCE_BYTES = (
    b"dance,fes,fex,fei\r\n\x1b[0;33;40mOK, Janet the protector \x1b[1;33;40mdances.\x1b[0;33;40m\x1b[1;37;40m\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m\x1b[1;32;40m60\x1b[0;37;40m \x1b[1;32;40m60"
    b"\x1b[0;37;40m 53 61 56 59 0 60 0377 N N N N 46 F\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
    b"\x1b[1;37;40m\x1b[0;37;40mup down out swampward south southeast east north\r\n\x1b[1;37;40m\x1b[0;34;40m"
    b"\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40mcoracle\r\nvial1\r\nring0\r\nbrand39\r\ncoronet\r\n"
    b"========\r\nkey50\r\ncloth-of-gold\r\nbroadsword\r\n\x1b[1;37;40m\r\n\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
    b"\x1b[1;37;40m"
)

BYTES_CASES = {
    "dally_lane": {
        "raw": DALLY_LANE_BYTES,
        "fes": {
            "points": 200,
            "vitals": [71, 71, 51, 51, 58, 58, 0, 71],
            "flags": [0, 0, 0, 0],
            "reset_minutes": 53,
            "weather": "raining",
        },
        "fex": {
            "names": {"up", "out", "swampward", "southwest", "south", "west", "east", "north"},
        },
        "fei": {
            "portables": ("streetsign",),
            "inventory": (),
        },
    },
    "badly_paved": {
        "raw": BADLY_PAVED_BYTES,
        "fes": {
            "points": 200,
            "vitals": [43, 43, 69, 69, 68, 68, 0, 43],
            "flags": [0, 0, 0, 0],
            "reset_minutes": 53,
            "weather": "fair",
        },
        "fex": {
            "names": {
                "up",
                "in",
                "out",
                "swampward",
                "southwest",
                "south",
                "southeast",
                "northeast",
                "northwest",
                "west",
                "east",
                "north",
            },
        },
        "fei": {
            "portables": (),
            "inventory": (),
        },
    },
    "janet_dance": {
        "raw": JANET_DANCE_BYTES,
        "fes": {
            "points": 377,
            "vitals": [60, 60, 53, 61, 56, 59, 0, 60],
            "flags": [0, 0, 0, 0],
            "reset_minutes": 46,
            "weather": "fair",
        },
        "fex": {
            "names": {"up", "down", "out", "swampward", "south", "southeast", "east", "north"},
        },
        "fei": {
            "portables": ("coracle", "vial1", "ring0", "brand39", "coronet"),
            "inventory": ("key50", "cloth-of-gold", "broadsword"),
        },
    },
}
