import os
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional

from fastapi.responses import JSONResponse

# PUBLIC_INTERFACE
def error_response(status_code: int, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> JSONResponse:
    """Create a consistent JSON error payload."""
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


class _SemaphorePool:
    """Simple global semaphore to limit OCR concurrency."""
    def __init__(self, limit: int):
        import threading
        self.sem = threading.Semaphore(limit)

    @contextmanager
    def acquire(self):
        self.sem.acquire()
        try:
            yield
        finally:
            self.sem.release()


# PUBLIC_INTERFACE
class OcrConcurrencyLimiter:
    """Global OCR concurrency limiter controlled via env OCR_MAX_CONCURRENCY (default 2)."""
    def __init__(self):
        limit = 2
        try:
            limit = max(1, int(os.getenv("OCR_MAX_CONCURRENCY", "2")))
        except Exception:
            pass
        self.pool = _SemaphorePool(limit)

    @contextmanager
    def acquire(self):
        with self.pool.acquire():
            yield


# PUBLIC_INTERFACE
class CleanupController:
    """
    Periodic cleanup controller driven by env:
      FILE_RETENTION_SECONDS: int, default 86400 (1 day)
      ENABLE_CLEANUP: "true"/"false", default "false"
      CLEANUP_INTERVAL_SECONDS: int, default 600
    """
    def __init__(self, storage, enable: bool):
        self.storage = storage
        self.enable = enable
        self.thread = None
        self._stop = threading.Event()

    def start(self):
        if not self.enable:
            return
        interval = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "600"))
        retention = int(os.getenv("FILE_RETENTION_SECONDS", "86400"))

        def _runner():
            while not self._stop.is_set():
                try:
                    self.storage.cleanup(retention)
                    # silently continue; could log if a logger is configured
                except Exception:
                    pass
                self._stop.wait(interval)

        self.thread = threading.Thread(target=_runner, name="cleanup-thread", daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread:
            self._stop.set()
            self.thread.join(timeout=2)
