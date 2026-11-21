"""SQLite experiment store with explicit result provenance categories.

Result categories are never mixed in queries or in the UI:

* ``author_reported``   – numbers quoted from a paper (must cite the publication id)
* ``reproduced``        – full official protocol re-run locally
* ``local_educational`` – local experiment on a teaching subset / non-official protocol
* ``synthetic_demo``    – computed on SYNTHETIC data; illustrative only
"""

from __future__ import annotations

import json
import platform
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

from core import paths

CATEGORIES = ("author_reported", "reproduced", "local_educational", "synthetic_demo")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('author_reported','reproduced','local_educational','synthetic_demo')),
    method_id TEXT NOT NULL,
    model_version TEXT,
    dataset_id TEXT,
    dataset_category TEXT,
    protocol TEXT,
    params TEXT,
    seed INTEGER,
    environment TEXT,
    metrics TEXT,
    notes TEXT,
    publication_id TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sample_id TEXT NOT NULL,
    split TEXT,
    label TEXT,
    score REAL,
    latency_ms REAL,
    map_path TEXT,
    PRIMARY KEY (run_id, sample_id)
);
"""


def environment_snapshot() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "cv2", "torch", "torchvision", "anomalib", "sklearn", "skimage", "PySide6", "open3d"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
            env["cuda_runtime"] = torch.version.cuda
    except Exception:
        pass
    return env


class ExperimentStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or paths.EXPERIMENTS / "experiments.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        return c

    def create_run(self, *, category: str, method_id: str, model_version: str = "", dataset_id: str | None = None,
                   dataset_category: str | None = None, protocol: str = "", params: dict | None = None,
                   seed: int | None = None, notes: str = "", publication_id: str | None = None,
                   metrics: dict | None = None) -> str:
        if category not in CATEGORIES:
            raise ValueError(f"unknown result category {category!r}")
        if category == "author_reported" and not publication_id:
            raise ValueError("author-reported results must cite a publication_id")
        run_id = uuid.uuid4().hex[:12]
        with closing(self._conn()) as c, c:
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (run_id, time.time(), category, method_id, model_version, dataset_id,
                       dataset_category, protocol, json.dumps(params or {}), seed,
                       json.dumps(environment_snapshot()), json.dumps(metrics or {}), notes, publication_id))
        return run_id

    def add_predictions(self, run_id: str, rows: list[dict]) -> None:
        with closing(self._conn()) as c, c:
            c.executemany("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?)",
                          [(run_id, r["sample_id"], r.get("split"), r.get("label"), r.get("score"),
                            r.get("latency_ms"), r.get("map_path")) for r in rows])

    def set_metrics(self, run_id: str, metrics: dict) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE runs SET metrics=? WHERE run_id=?", (json.dumps(metrics), run_id))

    def runs(self, category: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM runs", ()
        if category:
            q, a = q + " WHERE category=?", (category,)
        with closing(self._conn()) as c:
            rows = [dict(r) for r in c.execute(q + " ORDER BY created DESC", a)]
        for r in rows:
            for k in ("params", "environment", "metrics"):
                r[k] = json.loads(r[k] or "{}")
        return rows

    def predictions(self, run_id: str) -> list[dict]:
        with closing(self._conn()) as c:
            return [dict(r) for r in c.execute("SELECT * FROM predictions WHERE run_id=?", (run_id,))]
