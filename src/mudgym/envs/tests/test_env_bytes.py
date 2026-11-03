CAPTURED_RESET_SWEEP = (
    b"\r\n+- Database 2 has finished initialising -+\r\n"
    b"\r\n+- Database 5 has finished initialising -+\r\n"
    b"\r\n+- Database 1 has finished initialising -+\r\n"
    b"fes\r\n"
    b"\x1b[0;37;40m\x1b[1;32;40m57\x1b[0;37;40m \x1b[1;32;40m57\x1b[0;37;40m 65 65 58 58 0 57 0200 N N N N 105 R\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
)


def test_database_reset_broadcasts_stay_out_of_text(scripted_env_factory):
    env = scripted_env_factory(observation="text")
    env.reset()
    obs, render_bytes, _ = env.unwrapped.bytes_to_observation(
        CAPTURED_RESET_SWEEP, sent_lines=["fes"], response_complete=True
    )
    assert "initialising" not in obs["text"]
    assert b"initialising" not in render_bytes
    assert obs["points"] == 200
