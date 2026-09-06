"""Captured real game bytes with documented expected parsed values."""

# SQL refusals captured after collapsing in the flooded ford and being blinded by the vampire.
FORD_COLLAPSE_BYTES = (
    b"\x1b[1;37;40mmove swampward\r\n"
    b"\x1b[32mFord across river\x1b[37m.\r\n"
    b"\x1b[0;32;40mYou are standing on a ford across a fast-flowing river. To the west is a badly-paved "
    b"road, which carries on into the distance. Northwest is a ramshackle old building, and southwest "
    b"is some sort of well. South lies a forest, and north is the west bank of the river you now cross. "
    b"The ford goes beneath the water level to the east, but you can still go that way if you so "
    b"desire. \x1b[1;37;40m\x1b[0;32;40mIt is raining. \x1b[1;37;40m\r\n"
    b"Rain has swollen the river to a raging torrent! You fight your way across, but are constantly "
    b"buffeted and pounded all the way, causing you major injury!\r\n"
    b"You feel unbearably giddy.\r\n"
    b"You collapse, unconscious.\r\n"
    b"Your stamina has fallen from \x1b[0;33;40m11\x1b[1;37;40m to \x1b[31m1\x1b[37m.\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40msql,fes,fex,fei\r\n"
    b"You can't wake yourself up yet!\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"\x1b[1;31;40m1\x1b[0;37;40m \x1b[1;32;40m51\x1b[0;37;40m 33 47 39 52 0 51 075 N N N N 52 R\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"up in down out swampward southwest south southeast northeast northwest west east north\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"========\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
)

VAMPIRE_BLIND_BYTES = (
    b"\x1b[31mYou are wounded by the violence of a crafty, upward blow by the vampire.\r\n"
    b"Stamina=\x1b[33m51\x1b[31m/\x1b[32m73\x1b[31m.\x1b[37m\r\n"
    b"\x1b[31mYou ably graze the vampire with a punishing spurt.\r\nDamage: 8.\x1b[37m\r\n"
    b"\x1b[0;37;40mThe vampire looks strong.\r\n"
    b"\x1b[1;37;40m\x1b[36mA fabulous, gold ring with a fearful, bloodstone setting has been dropped here. \r\n"
    b"\x1b[37mThe vampire makes some magical gestures.\r\n"
    b"\x1b[31mYou have suddenly and magically gone blind!\x1b[37m\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
    b"west\r\n"
    b"You can't just leave in the middle of a fight! You have to flee!\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40msql,fes,fex,fei\r\n"
    b"You can't see a thing, you're blind.\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"\x1b[1;33;40m51\x1b[0;37;40m \x1b[1;32;40m73\x1b[0;37;40m 55 55 21 52 0 73 0200 Y N N N 52 F\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"--\r\n"
    b"========\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
)

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
