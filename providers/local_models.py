"""Request-scoped local specialist selection and safe idle unload hooks."""
from __future__ import annotations

import threading
import time
from typing import Callable


class LocalModelManager:
    def __init__(self, mapping: dict[str, str] | None = None, *, unload: Callable[[str], bool] | None = None, idle_minutes: int = 15, clock: Callable[[], float] = time.time):
        self.mapping = dict(mapping or {})
        self._unload = unload
        self.idle_seconds = max(0, idle_minutes) * 60
        self.clock = clock
        self.loaded_model: str | None = None
        self.last_used = 0.0
        self.active_requests = 0
        self._lock = threading.RLock()

    def load_local_model(self, capability: str) -> str:
        model = self.mapping.get(capability)
        if not model:
            raise KeyError(f"No local model configured for {capability}")
        with self._lock:
            self.loaded_model, self.last_used = model, self.clock()
        return model

    def request_started(self) -> None:
        with self._lock: self.active_requests += 1; self.last_used = self.clock()

    def request_finished(self) -> None:
        with self._lock: self.active_requests = max(0, self.active_requests - 1); self.last_used = self.clock()

    def unload_local_model(self) -> bool:
        with self._lock:
            if self.active_requests or not self.loaded_model: return False
            model = self.loaded_model
            if self._unload and not self._unload(model): return False
            self.loaded_model = None
            return True

    def unload_if_idle(self) -> bool:
        with self._lock:
            idle = bool(self.loaded_model and not self.active_requests and self.idle_seconds and self.clock() - self.last_used >= self.idle_seconds)
        return self.unload_local_model() if idle else False


__all__ = ["LocalModelManager"]
