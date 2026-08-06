"""State transition lookup table for the connection state machine."""

from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, NamedTuple

from mudgym.connections.persona import generate_persona_name, parse_persona_screen
from mudgym.connections.prompts import Prompt, State
from mudgym.logs import get_logger

if TYPE_CHECKING:  # pragma: no cover - only needed for static analysis
    from mudgym.connections.state_machine import ConnectionState


logger = get_logger(__name__)


TransitionAction = Callable[["ConnectionState"], None]


class Transition(NamedTuple):
    """Definition of a single state transition."""

    next_state: State
    action: TransitionAction | None = None
    expected_prompt: Prompt | None = None


def T(
    next_state: State,
    action: TransitionAction | None = None,
    expected_prompt: Prompt | None = None,
) -> Transition:
    """Helper to keep table entries concise."""

    return Transition(next_state=next_state, action=action, expected_prompt=expected_prompt)


def handle_supersede(sm: "ConnectionState") -> None:
    logger.info("transition.supersede", account_id=sm.account_id)
    # blowing up here as I think supporting supercede may allow hard to find bugs with connections that aren't closed properly.
    raise RuntimeError("Account already in use.")


def send_db_slot(sm: "ConnectionState") -> None:
    slot = sm.default_db_slot if sm.default_db_slot is not None else 0
    answer = f"p{slot}"

    # The menu redraws its Option prompt around interstitials (eg the MAIL screens after mgquit)
    # before consuming an answer we already sent. The game queues typed input, so answering the
    # redraw too would leave a stray line that the persona dialogue later swallows as a name,
    # desynchronising every answer after it. The echo is the consumption signal: only answer
    # again once the previous answer has been echoed back.
    if sm.pending_menu_answer is not None and sm.pending_menu_answer.encode("ascii") not in sm.get_buffer():
        logger.debug(
            "transition.skip_db_slot",
            reason="previous answer not consumed",
            pending=sm.pending_menu_answer,
            state=sm.state.name,
        )
        return

    logger.debug("transition.send_db_slot", slot=slot, state=sm.state.name)
    sm.send(answer)
    sm.pending_menu_answer = answer


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
    actual_personas = {k: v for k, v in personas.items() if v != "Unused"}

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


def sip_tea(sm: "ConnectionState") -> None:
    """Transition action: automatically sip tea in the tearoom."""
    logger.debug("transition.sip_tea")
    sm.send(b"sip t")


GAME_OVER_TRANSITIONS: Mapping[Prompt, Transition] = {
    Prompt.GAME_OVER_QUIT_CHEERIO: T(State.GAME_OVER),
    Prompt.GAME_OVER_EPISODE_POINTS: T(State.GAME_OVER),
    Prompt.GAME_OVER_NOT_UPDATING_PERSONA: T(State.GAME_OVER),
}

LOGIN_TRANSITIONS: dict[Prompt, Transition] = {
    Prompt.SUPERSEDE: T(State.LOGIN, handle_supersede),
    Prompt.SESSION_DYING: T(State.LOGIN),  # Wait for session to die, OPTION comes next
    Prompt.BOOT_COMPLETE: T(State.LOGIN, lambda c: c.send_cr()),
}

GLOBAL_TRANSITIONS: dict[Prompt, Transition] = {
    Prompt.OPTION: T(State.OPTION, send_db_slot),
    Prompt.EOF: T(State.DEAD),
    Prompt.RESET_IN_PROGRESS: T(State.RESETTING),
    Prompt.DATABASE_NOT_INITIALIZED: T(State.RESETTING),
    Prompt.DATABASE_FINISHED_INITIALIZING: T(State.OPTION, send_db_slot_if_ours),
    Prompt.PERSONA_AVAILABLE: T(State.PERSONA_SELECT, choose_or_create_persona),
    Prompt.PERSONA_NAME: T(State.PERSONA_NAME_INPUT, send_persona_name),
    Prompt.TEA_SIPPED: T(State.TEA_SIPPED),
    Prompt.PAGER: T(State.OPTION, lambda c: c.send(b"Q")),
    Prompt.LIBRARY: T(State.OPTION, lambda c: c.send(b"Q")),
    Prompt.EXAMINE: T(State.OPTION, lambda c: c.send(b"Q")),
    **GAME_OVER_TRANSITIONS,
}


TRANSITIONS: MutableMapping[State, dict[Prompt, Transition]] = {
    State.INITIAL: {
        Prompt.TEAROOM: T(State.TEAROOM, sip_tea),
        Prompt.EOF: T(State.DEAD),
        **LOGIN_TRANSITIONS,
    },
    State.GAME: {
        Prompt.ENTERED_LAND: T(State.GAME),
        # answer the silent Option menu on leaving the game, as the login tables do
        Prompt.OPTION: T(State.OPTION, send_db_slot),
        **GAME_OVER_TRANSITIONS,
    },
    State.OPTION: {
        Prompt.OPTION: T(State.OPTION, send_db_slot),
        Prompt.PERSONA_AVAILABLE: T(State.PERSONA_SELECT, choose_or_create_persona),
        Prompt.PERSONA_NAME: T(State.PERSONA_NAME_INPUT, send_persona_name),
        Prompt.PERSONA_SEX: T(State.PERSONA_SEX_INPUT),
        Prompt.RESET_IN_PROGRESS: T(State.RESETTING),
        Prompt.DATABASE_NOT_INITIALIZED: T(State.RESETTING),
        Prompt.BOOT_COMPLETE: T(State.OPTION),
    },
    State.PERSONA_SELECT: {
        Prompt.PERSONA_NAME: T(State.PERSONA_NAME_INPUT, send_persona_name),
        Prompt.PERSONA_AVAILABLE: T(State.PERSONA_SELECT, choose_or_create_persona),
        Prompt.PERSONA_SEX: T(State.PERSONA_SEX_INPUT, lambda c: c.send(b"m")),
        Prompt.TEAROOM: T(State.TEAROOM, sip_tea),
        Prompt.GAME: T(State.GAME),
        Prompt.OPTION: T(State.OPTION, send_db_slot),
        Prompt.RESET_IN_PROGRESS: T(State.RESETTING),
        Prompt.DATABASE_NOT_INITIALIZED: T(State.RESETTING),
    },
    State.PERSONA_NAME_INPUT: {
        Prompt.PERSONA_SEX: T(State.PERSONA_SEX_INPUT, lambda c: c.send(b"m")),
        Prompt.PERSONA_NAME: T(State.PERSONA_NAME_INPUT, send_persona_name),
        Prompt.TEAROOM: T(State.TEAROOM, sip_tea),
        Prompt.OPTION: T(State.OPTION, send_db_slot),
        Prompt.RESET_IN_PROGRESS: T(State.RESETTING),
        Prompt.DATABASE_NOT_INITIALIZED: T(State.RESETTING),
    },
    State.PERSONA_SEX_INPUT: {
        Prompt.PERSONA_SEX: T(State.PERSONA_SEX_INPUT, lambda c: c.send(b"m")),
        Prompt.TEAROOM: T(State.TEAROOM, sip_tea),
        Prompt.GAME: T(State.GAME, sip_tea),
        Prompt.OPTION: T(State.OPTION, send_db_slot),
    },
    State.CLOSING: {
        # Q backs out of persona selection to the Option menu, and Q at the
        # Option menu logs the account out of mudlogin (EOF follows).
        Prompt.PERSONA_AVAILABLE: T(State.CLOSING, lambda c: c.send(b"Q")),
        Prompt.OPTION: T(State.CLOSING, lambda c: c.send(b"Q")),
    },
    State.RESETTING: {
        Prompt.RESET_IN_PROGRESS: T(State.RESETTING),
        Prompt.DATABASE_NOT_INITIALIZED: T(State.RESETTING),
        Prompt.BOOT_COMPLETE: T(State.LOGIN, lambda c: c.send_cr()),
        # After reset completes, Option: may come without DATABASE_FINISHED_INITIALIZING
        Prompt.OPTION: T(State.OPTION, send_db_slot),
        Prompt.PERSONA_AVAILABLE: T(State.PERSONA_SELECT, choose_or_create_persona),
        Prompt.TEAROOM: T(State.TEAROOM, sip_tea),
    },
}

TEAROOM_TRANSITIONS: dict[Prompt, Transition] = {
    Prompt.ENTERED_LAND: T(State.GAME),
    # answer the silent Option menu on arrival, as State.GAME does
    Prompt.OPTION: T(State.OPTION, send_db_slot),
    **GAME_OVER_TRANSITIONS,
    Prompt.TEA_SIPPED: T(State.TEA_SIPPED),
}

TEA_SIPPED_TRANSITIONS: dict[Prompt, Transition] = {
    Prompt.ENTERED_LAND: T(State.GAME),
    # answer the silent Option menu on arrival, as State.GAME does
    Prompt.OPTION: T(State.OPTION, send_db_slot),
    **GAME_OVER_TRANSITIONS,
}

TRANSITIONS[State.TEAROOM] = TEAROOM_TRANSITIONS
TRANSITIONS[State.TEA_SIPPED] = TEA_SIPPED_TRANSITIONS


for state in State:
    specific = TRANSITIONS.get(state, {})
    TRANSITIONS[state] = {**GLOBAL_TRANSITIONS, **specific}
