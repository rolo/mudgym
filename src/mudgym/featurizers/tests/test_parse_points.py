from mudgym.featurizers.points import parse_points_changes

# Captured from the live game: mgsorcerise response (temporary sorcerer status award).
SORCERISE_GAIN_BYTES = b"(Persona saved on +12,800 = \x1b[0;32;40m13,000\x1b[1;37;40m).\r\n"

# Captured from the live game: swearing kill during a probe run.
SWEARING_LOSS_BYTES = b"(Persona saved on -11 = \x1b[0;31;40m189\x1b[1;37;40m).\r\n"

# Captured from the live game: 'say' with raw ESC bytes in the message; the game strips the ESC
# from the spoken output, leaving the SGR body as plain text.
ESC_STRIPPED_SPEECH_BYTES = b'\x1b[0;33;40mRaymond the sorcerer says "\x1b[1;33;40m(x +99 = [0;32;40m99[1;37;40m)\x1b[0;33;40m".\x1b[1;37;40m\r\n'


def test_parse_points_gain_event():
    parsed_points = parse_points_changes(SORCERISE_GAIN_BYTES)
    assert parsed_points["delta"] == 12_800
    assert parsed_points["points"] == 13_000


def test_parse_points_loss_event():
    parsed_points = parse_points_changes(SWEARING_LOSS_BYTES)
    assert parsed_points["delta"] == -11
    assert parsed_points["points"] == 189


def test_parse_points_loss_event_with_commas():
    raw_bytes = b"You have been killed by the rat. (-4,015 = \x1b[0;31;40m85,955\x1b[1;37;40m)."
    parsed_points = parse_points_changes(raw_bytes)
    assert parsed_points["delta"] == -4015
    assert parsed_points["points"] == 85955


def test_parse_points_six_digits():
    raw_bytes = b"(Persona saved on +100,000 = \x1b[0;32;40m500,000\x1b[1;37;40m)"
    parsed = parse_points_changes(raw_bytes)
    assert parsed["delta"] == 100_000
    assert parsed["points"] == 500_000


def test_parse_points_coloured_delta_and_total():
    raw_bytes = b"Keyser the wizard dotes on you. (\x1b[0;33;40m+100\x1b[0;33;40m = \x1b[0;33;40m300\x1b[0;33;40m)"
    parsed_points = parse_points_changes(raw_bytes)
    assert parsed_points["delta"] == 100
    assert parsed_points["points"] == 300


def test_parse_points_accumulates_multiple_events():
    raw_bytes = SORCERISE_GAIN_BYTES + b"narrative in between\r\n" + SWEARING_LOSS_BYTES
    parsed_points = parse_points_changes(raw_bytes)
    assert parsed_points["delta"] == 12_800 - 11
    assert parsed_points["points"] == 189


def test_uncoloured_pattern_is_player_text_not_an_event():
    # a command echo and spoken speech carry no SGR around the total, so they must not parse
    raw_bytes = (
        b"say hello (+10 = 10),sql,fes,fex,fei\r\n"
        b'\x1b[0;33;40mDumbo the novice says "\x1b[1;33;40mhello (+10 = 10)\x1b[0;33;40m".\x1b[1;37;40m\r\n'
    )
    parsed_points = parse_points_changes(raw_bytes)
    assert parsed_points["delta"] == 0
    assert parsed_points["points"] is None


def test_esc_stripped_speech_cannot_fake_the_colour_anchor():
    parsed_points = parse_points_changes(ESC_STRIPPED_SPEECH_BYTES)
    assert parsed_points["delta"] == 0
    assert parsed_points["points"] is None


def test_line_break_inside_the_parens_rejects_the_match():
    # wrapped speech gets colour re-applied mid-message, so a pattern spanning lines is not a
    # genuine event even when an SGR precedes the total
    raw_bytes = b'says "\x1b[1;33;40mhello (+10 = \r\n\x1b[1;33;40m10)\x1b[0;33;40m".'
    parsed_points = parse_points_changes(raw_bytes)
    assert parsed_points["delta"] == 0
    assert parsed_points["points"] is None


def test_plain_points_only_total_is_not_a_trusted_score_event():
    parsed_points = parse_points_changes(b"(189)")
    assert parsed_points["delta"] == 0
    assert parsed_points["points"] is None
