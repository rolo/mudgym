import re

ansi_escape_bytes = re.compile(
    rb"""
    (?:\x1B  # ESC
        (?:  # 7-bit C1 Fe (except CSI)
            [@-Z\\-_] 
        |    # CSI sequence
            \[ [0-?]* [ -/]* [@-~] 
        |    # OSC, PM, APC sequences
            \] .*? (?:\x07|\x1B\\)  
        |    # DCS sequences
            P .*? (?:\x07|\x1B\\)
        |    # SOS/PM/APC sequences
            [_^] .*? (?:\x07|\x1B\\)
        )
    )
    """,
    re.VERBOSE,
)


def strip_ansi(data: bytes) -> bytes:
    """Remove ANSI escape sequences from a byte string."""
    return ansi_escape_bytes.sub(b"", data)
