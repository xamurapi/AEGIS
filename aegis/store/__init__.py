"""Versioned, atomic persistence shared by the new contours (spec §3.2)."""
from aegis.store.migrations import (
    CURRENT_VERSION, MigrationError, migrate, read_store, version_of, write_store,
)

__all__ = ["CURRENT_VERSION", "MigrationError", "migrate", "read_store",
           "version_of", "write_store"]
