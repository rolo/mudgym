"""State transition lookup table for the connection state machine."""

from collections.abc import Callable
from time import monotonic, sleep
from typing import TYPE_CHECKING, NamedTuple

from mudgym.connections.persona import (
    UNUSED_PERSONA,
    generate_persona_name,
    generate_persona_sex,
    parse_persona_screen,
)
from mudgym.connections.prompts import Prompt, State
from mudgym.featurizers.ansi import strip_ansi
from mudgym.logs import get_logger

if TYPE_CHECKING:  # pragma: no cover - only needed for static analysis
    from mudgym.connections.state_machine import ConnectionState


logger = get_logger(__name__)


# enough to recognise a short echo line that straddles two matches, and no more
MENU_ECHO_WINDOW_BYTES = 512

# the least time we leave between two answers to the Option menu
MENU_ANSWER_MIN_INTERVAL_SECONDS = 0.02


def menu_answer_was_echoed(output: bytes, answer: bytes) -> bool:
    """The answer came back on a line of its own, rather than its bytes turning up in passing."""
    lines = output.replace(b"\r\n", b"\n").split(b"\n")
    return any(strip_ansi(line).strip() == answer for line in lines)


TransitionAction = Callable[["ConnectionState"], None]


class Transition(NamedTuple):
    """Definition of a single state transition."""

    next_state: State
    action: TransitionAction | None = None


def T(
    next_state: State,
    action: TransitionAction | None = None,
) -> Transition:
    """Helper to keep table entries concise."""

    return Transition(next_state=next_state, action=action)


def handle_supersede(sm: "ConnectionState") -> None:
    logger.info("transition.supersede", account_id=sm.account_id)
    # blowing up here as I think supporting supercede may allow hard to find bugs with connections that aren't closed properly.
    raise RuntimeError("Account already in use.")


def send_db_slot(sm: "ConnectionState") -> None:
    slot = sm.default_db_slot if sm.default_db_slot is not None else 0
    answer = b"p%d" % slot

    # only answer again once the previous answer has been echoed back.
    pending = sm.pending_menu_answer
    if pending is not None and not (sm.menu_answer_echoed or menu_answer_was_echoed(sm.get_buffer(), pending)):
        logger.debug(
            "transition.skip_db_slot",
            reason="previous answer not consumed",
            pending=sm.pending_menu_answer,
            state=sm.state.name,
        )
        return

    # stop flooding spam retries
    waited = monotonic() - sm.last_menu_answer_at
    if waited < MENU_ANSWER_MIN_INTERVAL_SECONDS:
        sleep(MENU_ANSWER_MIN_INTERVAL_SECONDS - waited)

    logger.debug("transition.send_db_slot", slot=slot, state=sm.state.name)
    sm.send(answer)
    sm.last_menu_answer_at = monotonic()
    sm.pending_menu_answer = answer
    sm.output_since_menu_answer = b""
    sm.menu_answer_echoed = False


def send_db_slot_if_ours(sm: "ConnectionState") -> None:
    our_slot = sm.default_db_slot if sm.default_db_slot is not None else 0
    matched = sm.child.after if hasattr(sm.child, "after") else b""
    if isinstance(matched, bytes):
        matched = matched.decode("latin-1")

    if our_slot == 0 and "The database has" in matched or f"Database {our_slot} has" in matched:
        send_db_slot(sm)
    else:
        logger.debug("transition.skip_db_slot", reason="other_slot", matched=matched[:50])


def choose_or_create_persona(sm: "ConnectionState") -> None:
    """Transition action: automatically select existing persona or create new one."""
    # Parse available personas from buffer
    current_buffer = sm.get_buffer()
    personas = parse_persona_screen(current_buffer)
    actual_personas = {k: v for k, v in personas.items() if v != UNUSED_PERSONA}

    if actual_personas:
        # Select existing persona
        preferred_slot = sm.default_persona_slot
        if preferred_slot is None or preferred_slot not in actual_personas:
            slot = min(actual_personas.keys())
        else:
            slot = preferred_slot
        logger.info("transition.persona.select_existing", slot=slot, persona=actual_personas[slot])
        sm.send(str(slot))
    else:
        # Create new persona - select first slot or generate name
        if personas:  # We have unused slots listed
            slot = 1 if 1 in personas else min(personas.keys())
            logger.info("transition.persona.select_unused_slot", slot=slot)
            sm.send(str(slot))
        else:
            # No slots shown, send new name directly
            name_gen = sm.default_name_generator or generate_persona_name
            name = name_gen()
            logger.info("transition.persona.create_new", name=name)
            sm.send(name)


def send_persona_name(sm: "ConnectionState") -> None:
    """Transition action: generate and send a persona name."""
    name_gen = sm.default_name_generator or generate_persona_name
    name = name_gen()
    logger.info("transition.persona.send_name", name=name)
    sm.send(name)


def send_persona_sex(sm: "ConnectionState") -> None:
    """Transition action: answer the persona sex question."""
    sex = generate_persona_sex()
    logger.info("transition.persona.send_sex", sex=sex)
    sm.send(sex)


def sip_tea(sm: "ConnectionState") -> None:
    """Transition action: automatically sip tea in the tearoom."""
    logger.debug("transition.sip_tea")
    sm.send(b"sip t")


GAME_OVER_TRANSITIONS: dict[Prompt, Transition] = {
    Prompt.GAME_OVER_QUIT_CHEERIO: T(State.GAME_OVER),
    Prompt.GAME_OVER_EPISODE_POINTS: T(State.GAME_OVER),
    Prompt.GAME_OVER_NOT_UPDATING_PERSONA: T(State.GAME_OVER),
    Prompt.GAME_OVER_KILLED_FOR_SWEARING: T(State.GAME_OVER),
}

GLOBAL_TRANSITIONS: dict[Prompt, Transition] = {
    Prompt.OPTION: T(State.OPTION, send_db_slot),
    Prompt.EOF: T(State.DEAD),
    Prompt.RESET_IN_PROGRESS: T(State.RESETTING),
    Prompt.DATABASE_NOT_INITIALIZED: T(State.RESETTING),
    Prompt.PERSONA_AVAILABLE: T(State.PERSONA_SELECT, choose_or_create_persona),
    Prompt.PERSONA_NAME: T(State.PERSONA_NAME_INPUT, send_persona_name),
    Prompt.TEA_SIPPED: T(State.TEA_SIPPED),
    Prompt.PAGER: T(State.OPTION, lambda c: c.send(b"Q")),
    Prompt.LIBRARY: T(State.OPTION, lambda c: c.send(b"Q")),
    Prompt.EXAMINE: T(State.OPTION, lambda c: c.send(b"Q")),
    **GAME_OVER_TRANSITIONS,
}

# these mean the same thing wherever we are, so answer them from every state still in the dialogue
DIALOGUE_TRANSITIONS: dict[Prompt, Transition] = {
    Prompt.PERSONA_SEX: T(State.PERSONA_SEX_INPUT, send_persona_sex),
    Prompt.TEAROOM: T(State.TEAROOM, sip_tea),
}

DIALOGUE_STATES = (
    State.INITIAL,
    State.LOGIN,
    State.OPTION,
    State.PERSONA_SELECT,
    State.PERSONA_NAME_INPUT,
    State.PERSONA_SEX_INPUT,
    State.RESETTING,
)

STATE_TRANSITIONS: dict[State, dict[Prompt, Transition]] = {
    State.INITIAL: {
        Prompt.SUPERSEDE: T(State.LOGIN, handle_supersede),
        Prompt.SESSION_DYING: T(State.LOGIN),
        Prompt.BOOT_COMPLETE: T(State.LOGIN, lambda sm: sm.send_cr()),
    },
    State.GAME: {
        Prompt.ENTERED_LAND: T(State.GAME),
    },
    State.OPTION: {
        Prompt.BOOT_COMPLETE: T(State.OPTION),
    },
    State.PERSONA_SELECT: {
        Prompt.GAME: T(State.GAME),
    },
    State.PERSONA_SEX_INPUT: {
        Prompt.GAME: T(State.GAME, sip_tea),
    },
    State.CLOSING: {
        # Q backs out of persona selection to the Option menu, and Q at the
        # Option menu logs the account out of mudlogin (EOF follows).
        Prompt.PERSONA_AVAILABLE: T(State.CLOSING, lambda sm: sm.send(b"Q")),
        Prompt.OPTION: T(State.CLOSING, lambda sm: sm.send(b"Q")),
    },
    State.RESETTING: {
        Prompt.BOOT_COMPLETE: T(State.LOGIN, lambda sm: sm.send_cr()),
        # the only state waiting on a database - stay put and let the next real prompt place us
        Prompt.DATABASE_FINISHED_INITIALIZING: T(State.RESETTING, send_db_slot_if_ours),
    },
    State.TEAROOM: {
        Prompt.ENTERED_LAND: T(State.GAME),
    },
    State.TEA_SIPPED: {
        Prompt.ENTERED_LAND: T(State.GAME),
    },
}

TRANSITIONS = {
    state: GLOBAL_TRANSITIONS
    | (DIALOGUE_TRANSITIONS if state in DIALOGUE_STATES else {})
    | STATE_TRANSITIONS.get(state, {})
    for state in State
}
