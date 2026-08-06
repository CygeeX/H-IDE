"""Filesystem-layout compatibility wrapper for the importable control API.

The service directory follows the deployment name from the implementation
plan (``control-api``), while Python imports use the underscore package in
``services.control_api``.  Keep one implementation so both layouts stay in
sync.
"""

from services.control_api.main import app, create_app

__all__ = ["app", "create_app"]
