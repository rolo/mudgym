# A real seagull death captured live (via session.py) from a jump off "Lovers' Leap" at the
# "Beaten track near cliff" room. This is the post-echo portion of the death step, verbatim:
# "\r\n" line endings, the experience-level line, the score wrapped in ANSI SGR codes inside the
# "(Persona saved ...)" line, and a final "Overall, you scored ..." line ending in a bare "\r"
# with no trailing prompt marker (the episode terminates here).
SEAGULL_DEATH_BYTES = (
    b"You are splattered over a very large area, or at least most of you is. "
    b"The rest of your remains are, even now, being eaten by the seagulls "
    b"(especially your eyes). If you'd have looked properly before you leaped, "
    b"you might have decided not to jump!\r\n"
    b"You have changed experience level from protector to novice.\r\n"
    b"(Persona saved on -11 = \x1b[0;31;40m189\x1b[1;37;40m).\r\n"
    b"Overall, you scored 189 points this game.\r"
)


def test_bytes_to_observation_handles_incomplete_auto_command_batch(scripted_env_factory):
    """Death and other early exits can return narrative without auto-command field chunks.

    Driven by a real death captured from a live session rather than a hand-written stand-in, so the
    parser meets the exact bytes the game produces: "\\r\\n" line endings, ANSI SGR codes mid-line,
    and a death message with no trailing prompt marker.
    """
    env = scripted_env_factory(observation="parsed")
    # the command is echoed, but the episode ends before any auto-command response arrives
    raw_bytes = b"jump,sql,fes,fex,fei\r\n" + SEAGULL_DEATH_BYTES
    info = {"last_command": "jump"}

    obs = env.bytes_to_observation(raw_bytes, info)

    # no auto-command responses arrived, so the structured fields stay at their empty defaults.
    # obs["points"] is sourced from the `fes` response, which never came, so it stays 0 here --
    # the death score is recovered separately, below.
    assert info["auto_command_chunks"] == []
    assert obs["room_name"] == ""
    assert obs["here"] == ()
    assert obs["points"] == 0

    # the narrative survives as text, with ANSI codes stripped and lines normalised
    assert "seagulls" in info["text"]
    assert "experience level from protector to novice" in info["text"]
    assert "(Persona saved on -11 = 189)." in info["text"]
    assert "Overall, you scored 189 points this game" in info["text"]
    assert "\x1b" not in info["text"]
