"""Clean shutdown of background tasks and atomic checkpoint writes (maintenance 2026-11-21)."""

import time

import pytest

pytest.importorskip("pytestqt")


def test_task_shutdown_cancels_and_waits(qtbot):
    from app.workers import task

    state = {"cancelled": False}

    def job(progress, cancel):
        t0 = time.time()
        while time.time() - t0 < 30:          # would run 30 s unless cancelled
            if cancel():
                state["cancelled"] = True
                return None
            time.sleep(0.01)
        return "finished"

    task.start(job)
    time.sleep(0.1)
    t0 = time.time()
    assert task.shutdown(5000) is True
    assert state["cancelled"] and time.time() - t0 < 5


def test_checkpoint_write_is_atomic(synthetic_index, tmp_path, monkeypatch, deep):
    from core.inference import model_registry as MR
    from core.inference.session import LabSession
    monkeypatch.setattr(MR, "FITTED", tmp_path / "fitted")
    s = LabSession("padim")
    s.bind(synthetic_index)
    real_save = s.method.save_state

    def broken_save(path):                     # write half a file, then fail (e.g. app killed mid-save)
        path.write_bytes(b"partial")
        raise OSError("simulated interruption")
    monkeypatch.setattr(s.method, "save_state", broken_save)
    with pytest.raises(OSError):
        s.fit(max_images=4)
    assert not list((tmp_path / "fitted").glob("*")), "no truncated .pt or leftover .partial may remain"
    monkeypatch.setattr(s.method, "save_state", real_save)
    s2 = LabSession("padim")
    s2.bind(synthetic_index)
    s2.fit(max_images=4)
    assert [p.suffix for p in (tmp_path / "fitted").glob("*.pt")] == [".pt"]
    assert not list((tmp_path / "fitted").glob("*.partial"))
