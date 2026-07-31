from collections.abc import Sequence

from mudgym.envs.fields import FieldSpec, ObservationField, instantiate_field


def validate_field_spaces(field_parsers: Sequence[FieldSpec]) -> None:
    """
    Validate a set of observation fields as a group: each field's full_space() and full_empty() must agree on their
    key set, and no two fields may provide the same observation key (after include_keys filtering). Runs during
    MudEnv construction and is also usable standalone in tests and tooling.
    """
    key_owners: dict[str, ObservationField] = {}
    for spec in field_parsers:
        field = instantiate_field(spec)
        space = field.full_space()
        empty = field.full_empty()
        missing_empty = set(space) - set(empty)
        extra_empty = set(empty) - set(space)
        if missing_empty or extra_empty:
            raise ValueError(
                f"{field.__class__.__name__}: full_space() keys do not match full_empty() keys "
                f"(missing={sorted(missing_empty)}, extra={sorted(extra_empty)})"
            )
        for key in field.space():
            if key in key_owners:
                raise ValueError(
                    f"observation key {key!r} is provided by both {key_owners[key].__class__.__name__} "
                    f"and {field.__class__.__name__}; use include_keys to choose which field provides it"
                )
            key_owners[key] = field
