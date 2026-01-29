from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional, TextIO


class TelemetryLogger:
    """Structured JSONL logger for rover telemetry.

    Thread-safe, append-only logging of dict records, one JSON object per line.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._fp: Optional[TextIO] = open(self.path, "a", encoding="utf-8")

    def log_step(self, record: Dict[str, Any]) -> None:
        """Append a single telemetry record to the JSONL file."""
        if self._fp is None:
            return
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._fp.write(line + "\n")
            self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

