"""Training dispatcher package.

Provides a schema/registry/dispatch seam so CLI, API, and GUI can
call one canonical training path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .registry import ModeSpec, get_mode_registry
from .schema import (
    TrainingJobConfig,
    load_training_config,
    load_training_config_raw,
    materialize_dispatch_payload,
)

if TYPE_CHECKING:
    from .dispatch import DispatchContext

__all__ = [
    "DispatchContext",
    "ModeSpec",
    "TrainingJobConfig",
    "build_dispatch_context",
    "get_mode_registry",
    "load_training_config",
    "load_training_config_raw",
    "materialize_dispatch_payload",
    "run_training",
]


def __getattr__(name: str) -> Any:
    if name in {"DispatchContext", "build_dispatch_context", "run_training"}:
        from .dispatch import DispatchContext, build_dispatch_context, run_training

        exports = {
            "DispatchContext": DispatchContext,
            "build_dispatch_context": build_dispatch_context,
            "run_training": run_training,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
