import logging as stdlib_logging
import os
import sys
import time
from threading import RLock
from typing import Any, TextIO

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

LOGGER_NAME = "mudgym"

stdlib_logging.getLogger(LOGGER_NAME).addHandler(stdlib_logging.NullHandler())


class RelativeTimeStamper:
    """Add relative timestamp that auto-scales from ms -> s -> m -> h."""

    def __init__(self, key: str = "elapsed") -> None:
        self._start = time.perf_counter()
        self._key = key

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        elapsed = time.perf_counter() - self._start

        if elapsed < 1:
            value = f"{elapsed * 1000:.0f}ms"
        elif elapsed < 60:
            value = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            value = f"{elapsed / 60:.1f}m"
        else:
            value = f"{elapsed / 3600:.1f}h"

        event_dict[self._key] = value
        return event_dict


def _format_value(value: object) -> str:
    """Render values compactly while keeping spaces/containers unambiguous."""
    if isinstance(value, str):
        if not value or any(char.isspace() for char in value):
            return repr(value)
        return value
    return repr(value)


class CompactConsoleRenderer:
    """Compact console renderer: time [LEVEL] message key=val ... t=elapsed"""

    COLORS = {
        "debug": "\033[36m",  # cyan
        "info": "\033[32m",  # green
        "warning": "\033[33m",  # yellow
        "error": "\033[31m",  # red
        "critical": "\033[31;1m",  # bold red
        "exception": "\033[31m",  # red
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self, *, colors: bool = True) -> None:
        self._colors = colors

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> str:
        timestamp = event_dict.pop("timestamp", "")
        elapsed = event_dict.pop("elapsed", "")
        level = event_dict.pop("level", method_name)
        event = event_dict.pop("event", "")
        exception = event_dict.pop("exception", None)

        level_lower = str(level).lower()
        color = self.COLORS.get(level_lower, "") if self._colors else ""
        reset = self.RESET if color else ""
        bold = self.BOLD if self._colors else ""
        level_str = str(level).upper()

        parts: list[str] = []

        if timestamp:
            parts.append(str(timestamp))

        parts.append(f"{color}[{level_str}]{reset}")

        if event:
            parts.append(f"{bold}{event}{reset}")

        if event_dict:
            kvs = " ".join(f"{k}={_format_value(v)}" for k, v in event_dict.items())
            parts.append(kvs)

        if elapsed:
            parts.append(f"t={elapsed}")

        result = " ".join(parts)

        if exception:
            result += "\n" + str(exception)

        return result


_LOCK = RLock()
_configured = False
_configured_key: tuple[int, int] | None = None
_mudgym_handler: stdlib_logging.Handler | None = None
_processors: tuple[Processor, ...] = ()


def _library_processors() -> tuple[Processor, ...]:
    """Render safely through stdlib logging when setup_logging() has not run."""
    return (
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
        structlog.processors.KeyValueRenderer(key_order=["event", "level", "logger"]),
    )


def _process_event(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> Any:
    with _LOCK:
        processors = _processors

    event: Any = event_dict
    for processor in processors:
        event = processor(logger, method_name, event)
    return event


_processors = _library_processors()


def _level_to_int(level: str | int | None, *, default: int = stdlib_logging.WARNING) -> int:
    """Return a logging level constant."""
    if level is None:
        return default
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        clean_level = level.strip()
        if clean_level.isdigit():
            return int(clean_level)

        resolved = stdlib_logging.getLevelNamesMapping().get(clean_level.upper())
        if resolved is not None:
            return resolved

    raise ValueError(f"Unknown logging level: {level!r}")


def _stream_supports_color(stream: TextIO) -> bool:
    is_tty = getattr(stream, "isatty", lambda: False)
    return bool(is_tty()) and "NO_COLOR" not in os.environ


def setup_logging(
    *,
    level: str | int | None = None,
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """Configure compact console logging for the mudgym package.

    This installs a real handler on the ``mudgym`` logger so CLIs, training
    scripts, and notebooks get formatted output without configuring the root
    logger. Library users can ignore this function and use their application
    logging setup instead.

    Args:
        level: Desired logging level (name or int). Defaults to MUDGYM_LOG_LEVEL env var or WARNING.
        stream: Where to write logs. Defaults to stderr so stdout stays clean.
        force: Reconfigure even if the (level, stream) pair is unchanged.
    """
    global _configured, _configured_key, _mudgym_handler, _processors

    stream = stream if stream is not None else sys.stderr

    env_level = os.getenv("MUDGYM_LOG_LEVEL")
    effective_level = level if level is not None else env_level
    processor_level = _level_to_int(effective_level)
    config_key = (processor_level, id(stream))

    with _LOCK:
        if not force and _configured and _configured_key == config_key:
            return

        relative_stamper = RelativeTimeStamper()

        # Use a simple time format: HH:MM:SS.ffffff
        timestamper = structlog.processors.TimeStamper(fmt="%H:%M:%S.%f")

        # set_exc_info promotes a live exception onto the event; format_exc_info
        # then renders it to text for the compact renderer.
        exception_processors: list[Processor] = [
            structlog.dev.set_exc_info,
            structlog.processors.format_exc_info,
        ]

        renderer = CompactConsoleRenderer(colors=_stream_supports_color(stream))

        # foreign_pre_chain renders records from plain stdlib loggers (non-structlog)
        # and should mirror the structlog chain for consistent output.
        foreign_pre_chain: list[Processor] = [
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.ExtraAdder(),
            timestamper,
            relative_stamper,
            structlog.processors.StackInfoRenderer(),
            *exception_processors,
        ]

        processor_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=foreign_pre_chain,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )

        handler = stdlib_logging.StreamHandler(stream)
        handler.setFormatter(processor_formatter)

        lib = stdlib_logging.getLogger(LOGGER_NAME)
        if _mudgym_handler is not None:
            lib.removeHandler(_mudgym_handler)
            _mudgym_handler.close()
        lib.addHandler(handler)
        lib.setLevel(processor_level)
        lib.propagate = False  # don't double-log via root
        _mudgym_handler = handler

        _processors = (
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            relative_stamper,
            structlog.processors.StackInfoRenderer(),
            *exception_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        )

        _configured = True
        _configured_key = config_key


def get_logger(name: str | None = None) -> Any:
    """
    Get a structlog logger for the given name.

    Always use this in mudgym modules instead of structlog.get_logger().
    This is intentionally side-effect free: call setup_logging() from an
    entrypoint when compact console logs are desired.

    Args:
        name: Logger name (typically __name__)

    Returns:
        A structlog BoundLogger instance
    """
    return structlog.wrap_logger(
        stdlib_logging.getLogger(name or LOGGER_NAME),
        processors=[_process_event],
        context_class=dict,
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
