"""Test-only observations and provider doubles.

This package is outside ``src`` and therefore cannot be installed or selected
by the H-IDE Vision product runtime.
"""

from .observation import fixture_bundle, fixture_rgb
from .providers import StubQwenProvider, StubSam2Provider

__all__ = [
    "StubQwenProvider",
    "StubSam2Provider",
    "fixture_bundle",
    "fixture_rgb",
]
