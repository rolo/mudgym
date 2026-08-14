# seagull death jump off cliff captured with no trailing prompt marker (the episode terminates).
SEAGULL_DEATH_BYTES = (
    b"You are splattered over a very large area, or at least most of you is. "
    b"The rest of your remains are, even now, being eaten by the seagulls "
    b"(especially your eyes). If you'd have looked properly before you leaped, "
    b"you might have decided not to jump!\r\n"
    b"You have changed experience level from protector to novice.\r\n"
    b"(Persona saved on -11 = \x1b[0;31;40m189\x1b[1;37;40m).\r\n"
    b"Overall, you scored 189 points this game.\r"
)


def test_bytes_to_observation_handles_incomplete_command_window(scripted_env_factory):
    """Death and other early exits can return narrative without observation-field chunks."""
    env = scripted_env_factory(observation="parsed")
    # the command is echoed, but the episode ends before any observation response arrives
    raw_bytes = b"jump\r\n" + SEAGULL_DEATH_BYTES

    obs, _, _ = env.bytes_to_observation(
        raw_bytes,
        sent_lines=["jump"],
        response_complete=False,
    )

    # no observation responses arrived, so the structured fields stay at their empty defaults.
    # obs["points"] is sourced from the `fes` response, which never came, so it stays 0 here --
    # the death score is recovered separately, below.
    assert obs["room_name"] == ""
    assert obs["here"] == ()
    assert obs["points"] == 0

    # the narrative survives as text, with ANSI codes stripped and lines normalised
    assert "seagulls" in obs["text"]
    assert "experience level from protector to novice" in obs["text"]
    assert "(Persona saved on -11 = 189)." in obs["text"]
    assert "Overall, you scored 189 points this game" in obs["text"]
    assert "\x1b" not in obs["text"]
