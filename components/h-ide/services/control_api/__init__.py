"""HTTP control-plane service package.

The HTTP application is imported lazily so maintenance submodules such as the
offline administrator CLI do not construct an application or open its default
database merely because Python imported this package.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"app", "create_app"}:
        from .main import app, create_app

        return {"app": app, "create_app": create_app}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "create_app"]
