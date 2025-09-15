"""Dashboard package initializer.

This makes the `Dashboard` directory a proper Python package and ensures that
absolute imports used inside submodules (e.g., `from config import Config` in
`extensions.py`) resolve to the package's own modules.

No changes to existing modules are required.
"""

import sys as _sys

# Import package-local modules and alias them under top-level names so that
# absolute imports inside submodules keep working when importing as a package.
from . import config as _dashboard_config  # noqa: F401
from . import database as _dashboard_database  # noqa: F401

# Register aliases so `from config import ...` and `from database import ...`
# inside package modules resolve to `Dashboard.config` and `Dashboard.database`.
_sys.modules.setdefault("config", _dashboard_config)
_sys.modules.setdefault("database", _dashboard_database)

# Public re-exports (optional convenience)
try:  # Safe optional re-exports; avoid import errors during tooling
    from .config import Config  # noqa: F401
    from .database import DatabaseManager  # noqa: F401
except Exception:  # pragma: no cover - avoid hard failure during partial loads
    pass

__all__ = [
    "Config",
    "DatabaseManager",
]


