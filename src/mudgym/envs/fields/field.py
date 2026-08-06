import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from gymnasium import spaces

from mudgym.featurizers.ansi import strip_ansi
from mudgym.featurizers.strings import decode_text_bytes


class ObservationField(ABC):
    """
    A self-contained, pure (no side effects) observation field.

    A field declares:
    - the command that produces its bytes, if any;
    - the observation-space keys it owns;
    - empty defaults for those keys;
    - an extractor from response chunks to values.

    A `chunk` is the game's output bytes for a single game event delimited by a game prompt marker.

    The abstract methods (``full_space()``, ``full_empty()``, ``full_extract()``) declare the parser's
    full capability; ``space()``, ``empty()``, and ``extract()`` restrict that capability to
    ``include_keys`` and are what consumers read.
    """

    command: str | None = None

    # When this field's command ends the auto-command batch, this pattern identifies its response bytes as the step's
    # end of turn marker.
    # None (the default) means the response is not distinctive enough to trust for end of step marking duty.
    end_of_turn_marker: re.Pattern[bytes] | None = None

    # When True (the default), the chunk this field claims is considered consumed and not included in the observation `text` key.
    remove_on_match: bool = True

    # Messages the game emits in place of a command's real output when the persona cannot act
    # (unconscious, asleep, ...). A refusal claims the field's slot but carries no data.
    # Unknown responses keep failing loudly in bytes_to_observation.
    PLAYER_STATE_REFUSALS: tuple[bytes, ...] = (
        b"You can't wake yourself up yet!",
        b"You can't see a thing, you're blind.",
    )

    def __init__(self, include_keys: Sequence[str] | None = None):
        """
        Args:
            include_keys: restrict this field's observation contribution to the given space() keys.
            None (default) keeps every key. An empty sequence keeps none, useful for using as an end of step marker.
        """
        if include_keys is None:
            self.include_keys: tuple[str, ...] | None = None
            return
        self.include_keys = tuple(include_keys)
        unknown_keys = set(self.include_keys) - set(self.full_space())
        if unknown_keys:
            raise ValueError(
                f"{self.__class__.__name__} include_keys {sorted(unknown_keys)} are "
                f"not in full_space() keys {sorted(self.full_space())}"
            )

    def filter_keys(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Restrict a full_space/full_empty/full_extract mapping to this field's include_keys."""
        if self.include_keys is None:
            return dict(values)
        return {key: value for key, value in values.items() if key in self.include_keys}

    def space(self) -> dict[str, spaces.Space]:
        """The observation-space slice this field contributes: ``full_space()`` restricted to include_keys."""
        return self.filter_keys(self.full_space())

    def empty(self) -> dict[str, Any]:
        """Default values for the contributed keys: ``full_empty()`` restricted to include_keys."""
        return self.filter_keys(self.full_empty())

    def extract(self, chunks: Sequence[bytes], **context: Any) -> dict[str, Any]:
        """This field's observation contribution for a turn: ``full_extract()`` restricted to include_keys."""
        return self.filter_keys(self.full_extract(chunks, **context))

    @abstractmethod
    def full_space(self) -> dict[str, spaces.Space]:
        """Every observation-space key this field's parser can produce."""
        ...

    @abstractmethod
    def full_empty(self) -> dict[str, Any]:
        """Default values for every parser key when nothing matches (dtypes/shapes match ``full_space()``)."""
        ...

    @abstractmethod
    def full_extract(self, chunks: Sequence[bytes], **context: Any) -> dict[str, Any]:
        """Parse the turn's response ``chunks`` into every parser key. Pure: a function of its inputs alone.

        ``context`` carries observer facts the env supplies each call (currently ``persona``, the
        observing persona's bare name); a parser names what it consumes and ignores the rest.
        """
        ...

    def matches(self, chunk: bytes) -> bool:
        """Whether `chunk` is a valid output of this field's command.

        By default we just return True, but subclasses can override this to make matching more robust, in which case this
        method should return True for every output the command can really produce, including edge cases (eg, when dark, asleep,
        blind, etc)
        """
        return True

    def is_refusal(self, chunk: bytes) -> bool:
        """Whether ``chunk`` is a player-state refusal instead of this command's real output."""
        return strip_ansi(bytes(chunk)).strip() in self.PLAYER_STATE_REFUSALS

    def decode(self, raw_bytes: bytes) -> str:
        """Strip ANSI escape codes from the bytes and decode to text."""
        return decode_text_bytes(strip_ansi(bytes(raw_bytes)))

    def find_last_line(self, regex: re.Pattern[str], chunks: Sequence[bytes]) -> re.Match[str] | None:
        """Return the last line matching ``regex`` across the response chunks (or ``None``)."""
        match: re.Match[str] | None = None
        for chunk in chunks:
            for line in self.decode(chunk).splitlines():
                candidate = regex.match(line.strip())
                if candidate is not None:
                    match = candidate
        return match
