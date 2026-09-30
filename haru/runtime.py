"""Portable deadlines and atomic metadata writes for local training."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Deadline:
    def __init__(self, unix_deadline=None, seconds=None, clock=time.monotonic):
        self.clock = clock
        duration = float("inf") if seconds is None else seconds
        if unix_deadline is not None:
            duration = min(duration, unix_deadline - time.time())
        self.end = clock() + max(0, duration)
        self.requested = False

    def expired(self):
        return self.requested or self.clock() >= self.end

    def request_stop(self, *_):
        self.requested = True


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            # Windows indexers can briefly hold a JSON file without delete sharing.
            time.sleep(min(0.5, 0.05 * (attempt + 1)))
