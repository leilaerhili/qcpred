from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

# A circuit generator returns (circuit_object, metadata_dict)
GeneratorFn = Callable[..., Tuple[Any, dict]]

_REGISTRY: Dict[str, GeneratorFn] = {}


def register(family: str, fn: GeneratorFn) -> None:
    if not family or not isinstance(family, str):
        raise ValueError("family must be a non-empty string")
    if family in _REGISTRY:
        raise ValueError(f"Generator already registered for family='{family}'")
    _REGISTRY[family] = fn


def get_generator(family: str) -> GeneratorFn:
    try:
        return _REGISTRY[family]
    except KeyError as e:
        raise KeyError(f"Unknown circuit family '{family}'. Available: {sorted(_REGISTRY.keys())}") from e


def available_families() -> list[str]:
    return sorted(_REGISTRY.keys())
