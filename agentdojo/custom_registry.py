from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RegisterEntry:
    name: str
    factory: Callable[..., Any]
    description: str | None = None


_OPENROUTER_DEFENSES: dict[str, RegisterEntry] = {}


def register_openrouter_defense(
    name: str,
    factory: Callable[..., Any] | None = None,
    *,
    description: str | None = None,
):
    """Register a custom defense entry for run_openrouter_benchmark.py.

    Usage:
        @register_openrouter_defense("my_defense")
        def my_defense_factory(...):
            ...

    The factory should return an AgentPipeline instance.
    """

    def _register(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _OPENROUTER_DEFENSES:
            raise ValueError(f"Defense '{name}' is already registered")
        _OPENROUTER_DEFENSES[name] = RegisterEntry(name=name, factory=fn, description=description)
        return fn

    if factory is not None:
        return _register(factory)
    return _register


def get_openrouter_defense(name: str) -> RegisterEntry | None:
    return _OPENROUTER_DEFENSES.get(name)


def list_openrouter_defenses() -> list[str]:
    return sorted(_OPENROUTER_DEFENSES.keys())
