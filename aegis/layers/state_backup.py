"""State Backup — scheduled, emergency and checkpoint-based state persistence."""
import json
import gzip
import time
import shutil
import logging
from pathlib import Path
from collections import deque
from aegis.clock import CLOCK

logger = logging.getLogger("aegis.state_backup")


class StateBackup:
    """Flexible backup/restore system with compression, rotation and emergency snapshots."""

    def __init__(self, backup_dir: Path | None = None, max_backups: int = 20):
        self.backup_dir = backup_dir or Path(__file__).parent.parent.parent / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        self.backup_history: deque = deque(maxlen=100)
        self.last_backup_time = 0.0
        self.backup_count = 0
        self.restore_count = 0
        self.failed_backups = 0

    def save_state(self, state: dict, backup_type: str = "scheduled") -> dict:
        """Save compressed state snapshot."""
        ts = CLOCK.now()
        # Nanosecond stamp so two backups of the same type in the same second
        # don't collide and overwrite each other.
        filename = f"aegis_{backup_type}_{time.time_ns()}.json.gz"
        filepath = self.backup_dir / filename

        try:
            data = json.dumps(state, default=str).encode("utf-8")
            compressed = gzip.compress(data, compresslevel=6)
            filepath.write_bytes(compressed)

            record = {
                "time": ts,
                "type": backup_type,
                "file": filename,
                "size_raw": len(data),
                "size_compressed": len(compressed),
                "compression_ratio": round(len(compressed) / max(len(data), 1) * 100, 1),
                "success": True,
            }
            self.backup_history.append(record)
            self.backup_count += 1
            self.last_backup_time = ts
            self._rotate_backups()
            return record
        except Exception as e:
            self.failed_backups += 1
            record = {"time": ts, "type": backup_type, "success": False, "error": str(e)}
            self.backup_history.append(record)
            return record

    @staticmethod
    def _ns_stamp(f: Path) -> int:
        """Extract the nanosecond stamp from ``aegis_{type}_{ns}.json.gz``.

        Sorting by the whole filename is WRONG: the ``{type}`` field precedes
        the timestamp, so "scheduled" > "emergency" lexicographically and an old
        scheduled backup would outrank a fresh emergency one. Ordering by this
        stamp (falling back to mtime) restores true chronological order."""
        base = f.name
        if base.endswith(".json.gz"):
            base = base[: -len(".json.gz")]
        last = base.rsplit("_", 1)[-1]
        try:
            return int(last)
        except ValueError:
            try:
                return f.stat().st_mtime_ns
            except OSError:
                return -1

    def restore_latest(self, backup_type: str | None = None) -> dict | None:
        """Restore the most recent backup, optionally filtered by type."""
        files = sorted(self.backup_dir.glob("aegis_*.json.gz"),
                       key=self._ns_stamp, reverse=True)
        for f in files:
            if backup_type and backup_type not in f.name:
                continue
            try:
                compressed = f.read_bytes()
                data = gzip.decompress(compressed)
                state = json.loads(data.decode("utf-8"))
                self.restore_count += 1
                return state
            except Exception:
                continue
        return None

    def emergency_backup(self, state: dict) -> dict:
        """Immediate emergency backup with highest priority."""
        return self.save_state(state, backup_type="emergency")

    def _rotate_backups(self):
        """Keep only the most recent max_backups files PER TYPE.

        Rotating across all types together let a burst of one type (e.g.
        scheduled) evict the emergency snapshots that are meant to survive."""
        by_type: dict[str, list[Path]] = {}
        for f in self.backup_dir.glob("aegis_*.json.gz"):
            parts = f.name.split("_")
            btype = parts[1] if len(parts) > 2 else "unknown"
            by_type.setdefault(btype, []).append(f)
        for files in by_type.values():
            files.sort()  # oldest first (nanosecond stamp sorts chronologically)
            for old in files[: max(0, len(files) - self.max_backups)]:
                try:
                    old.unlink()
                except Exception:
                    logger.debug("Could not rotate old backup %s", old, exc_info=True)

    def list_backups(self) -> list[dict]:
        """List available backup files with metadata."""
        result = []
        for f in sorted(self.backup_dir.glob("aegis_*.json.gz"),
                        key=self._ns_stamp, reverse=True):
            parts = f.stem.replace(".json", "").split("_")
            result.append({
                "file": f.name,
                "size_bytes": f.stat().st_size,
                "type": parts[1] if len(parts) > 1 else "unknown",
                "timestamp": parts[2] if len(parts) > 2 else "unknown",
            })
        return result[:20]

    def status(self) -> dict:
        return {
            "backup_count": self.backup_count,
            "restore_count": self.restore_count,
            "failed_backups": self.failed_backups,
            "last_backup_time": self.last_backup_time,
            "available_backups": len(list(self.backup_dir.glob("aegis_*.json.gz"))),
            "backup_dir": str(self.backup_dir),
            "recent_history": [
                {"type": h["type"], "success": h["success"], "time": h["time"]}
                for h in list(self.backup_history)[-5:]
            ],
        }
