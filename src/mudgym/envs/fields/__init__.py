from collections.abc import Callable

from .feinventory import FEInventoryField
from .fescore import FEScoreField
from .fexits import FEXitsField
from .field import ObservationField
from .mgcheats import MGCheatsField
from .rawbytes import RawBytesField
from .superquicklook import SuperQuickLookField

FieldSpec = ObservationField | type[ObservationField] | Callable[[], ObservationField]


def instantiate_field(spec: FieldSpec) -> ObservationField:
    field = spec if isinstance(spec, ObservationField) else spec()
    if not isinstance(field, ObservationField):
        raise TypeError(f"observation field spec {spec!r} produced {field!r}, not an ObservationField")
    return field


__all__ = [
    "ObservationField",
    "FieldSpec",
    "FEInventoryField",
    "FEScoreField",
    "FEXitsField",
    "MGCheatsField",
    "RawBytesField",
    "instantiate_field",
    "SuperQuickLookField",
]
