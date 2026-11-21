"""Background task runner: keeps long operations (fitting, inference, extraction) off the GUI thread."""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class TaskSignals(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str, str)      # message, traceback


class Task(QRunnable):
    """Run ``fn(progress=..., cancel=..., **kwargs)`` in the global thread pool."""

    def __init__(self, fn: Callable[..., Any], **kwargs):
        super().__init__()
        self.fn, self.kwargs = fn, kwargs
        self.signals = TaskSignals()
        self._cancel = False
        self.setAutoDelete(False)

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        def emit(sig, *args):
            try:
                sig.emit(*args)
            except RuntimeError:      # receiver/window already destroyed (e.g. app closing)
                self._cancel = True

        try:
            out = self.fn(progress=lambda f, m="": emit(self.signals.progress, float(f), str(m)),
                          cancel=self.is_cancelled, **self.kwargs)
        except Exception as e:  # report to the GUI instead of crashing the worker thread
            emit(self.signals.failed, f"{type(e).__name__}: {e}", traceback.format_exc())
            return
        emit(self.signals.finished, out)


_running: set[Task] = set()


def _guard(cb):
    """Ignore callbacks that arrive after their widgets were destroyed (window closed while busy)."""
    def wrapper(*args):
        try:
            cb(*args)
        except RuntimeError as e:
            if "already deleted" not in str(e):
                raise
    return wrapper


def start(fn: Callable[..., Any], on_done=None, on_error=None, on_progress=None, **kwargs) -> Task:
    t = Task(fn, **kwargs)
    _running.add(t)
    if on_done:
        t.signals.finished.connect(_guard(on_done))
    if on_error:
        t.signals.failed.connect(_guard(on_error))
    if on_progress:
        t.signals.progress.connect(_guard(on_progress))
    t.signals.finished.connect(lambda *_: _running.discard(t))
    t.signals.failed.connect(lambda *_: _running.discard(t))
    QThreadPool.globalInstance().start(t)
    return t


def shutdown(timeout_ms: int = 10000) -> bool:
    """Ask every running task to stop (cooperative cancel) and wait for the pool, at most ``timeout_ms``.

    Called when the main window closes and after the Qt event loop ends, so that Python does not
    finalise its modules while worker threads are still running (that produced harmless but noisy
    ``torch.library`` finaliser TypeErrors, and could cut a checkpoint write short). Returns True if
    all workers finished in time."""
    for t in list(_running):
        t.cancel()
    return QThreadPool.globalInstance().waitForDone(timeout_ms)
