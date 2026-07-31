"""Wire line assembly: speech commands must not swallow the chained auto commands.

Speech verbs (say, shout, tell, ...) consume the rest of their input line as message text, so the
session puts the auto commands on a separate wire line for them; everything else stays one
comma-chained line.
"""

from mudgym.session import contains_speech_command, wire_lines

AUTO_COMMANDS = ["sql", "fes", "fex", "fei"]


def test_plain_commands_stay_on_one_chained_line():
    assert wire_lines("look", AUTO_COMMANDS) == ["look,sql,fes,fex,fei"]


def test_speech_command_takes_autos_on_a_separate_line():
    assert wire_lines("say hello", AUTO_COMMANDS) == ["say hello", "sql,fes,fex,fei"]


def test_say_shorthand_quote_is_speech():
    assert wire_lines('"hello', AUTO_COMMANDS) == ['"hello', "sql,fes,fex,fei"]
    assert wire_lines("'hello", AUTO_COMMANDS) == ["'hello", "sql,fes,fex,fei"]


def test_speech_mid_chain_splits_the_whole_batch():
    # the speech verb swallows everything after it on its line, so the player's own chain stays
    # intact on line one and only the autos move to line two
    assert wire_lines("get sword,say hi", AUTO_COMMANDS) == ["get sword,say hi", "sql,fes,fex,fei"]


def test_no_auto_commands_means_a_single_line():
    assert wire_lines("say hello", []) == ["say hello"]


def test_speech_detection_is_word_anchored():
    # commands merely starting with a speech verb's letters are not speech
    assert contains_speech_command("sayonara") is False
    assert contains_speech_command("tellurium") is False
    assert contains_speech_command("shout hello") is True
    assert contains_speech_command("SAY hello") is True
    assert contains_speech_command("wave") is False
