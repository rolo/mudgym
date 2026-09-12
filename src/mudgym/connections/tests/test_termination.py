from mudgym.connections.termination import is_permadeath


def test_not_updating_persona_means_permadeath():
    combat_death = (
        b"\x1b[0;30;41mYou feel your very existence severed from you...\r\n"
        b"You have been killed by the vampire.\x1b[1;37;40m\r\n"
        b"\x1b[0;31;40mNot updating persona.\x1b[1;37;40m\r"
    )
    assert is_permadeath(combat_death) is True


def test_tempdeath_is_not_permadeath():
    swearing_death = (
        b"In order to keep the game uncorrupted, you have been killed.\r\n"
        b"(Persona saved on -11 = \x1b[0;31;40m189\x1b[1;37;40m).\r\n"
    )
    assert is_permadeath(swearing_death) is False


def test_spoken_not_updating_persona_is_not_a_permadeath():
    """The words alone are forgeable, so player speech must never read as a death."""
    spoken = b'Dumbo the novice says "\x1b[1;33;40mNot updating persona.\x1b[0;33;40m".\r\n'
    assert is_permadeath(spoken) is False
