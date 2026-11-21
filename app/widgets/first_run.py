"""First-run assistant (Student Edition): welcome, overview, what this computer can do, what is
installed, mode selection and a recommended first exercise. Never downloads anything."""

from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QDialog, QHBoxLayout, QPushButton, QRadioButton, QVBoxLayout)

from app.i18n import tr
from app.widgets.common import markdown_browser


def capability_summary() -> dict:
    """Cheap, offline facts about this computer and installation (no downloads, no model loading)."""
    import importlib.util
    from core.datasets import registry as dreg
    from core.inference import model_registry as MR
    info = {"torch": importlib.util.find_spec("torch") is not None, "cuda": None, "gpu": None,
            "anomalib": importlib.util.find_spec("anomalib") is not None,
            "open3d": importlib.util.find_spec("open3d") is not None}
    if info["torch"]:
        try:
            import torch
            info["cuda"] = torch.cuda.is_available()
            info["gpu"] = torch.cuda.get_device_name(0) if info["cuda"] else None
        except Exception:  # noqa: BLE001
            info["cuda"] = False
    info["datasets"] = {d: dreg.installed_categories(d) for d in ("SYN", "D05", "D06", "D01", "D03", "D04")}
    info["stored_results"] = len(MR.load_verified())
    info["fitted"] = sum(1 for e in MR.scan() if e.manifest)
    return info


def summary_markdown(info: dict) -> str:
    yes, no = "✓", "—"
    ds = info["datasets"]
    real = {k: v for k, v in ds.items() if k != "SYN" and v}
    gpu = (f"{yes} {info['gpu']}" if info["cuda"] else
           (f"{no} PyTorch installed, no CUDA GPU — deep methods run on the CPU (slower)" if info["torch"]
            else f"{no} PyTorch not installed (core edition): classical methods, Atlas, Library, Learning Center and "
                 "stored experiment results work; install the 'deep' profile for neural methods"))
    return "\n".join([
        "## " + tr("fr.welcome", "Welcome to IDD Lab — Student Edition"),
        tr("fr.overview", "IDD Lab is an educational desktop application for industrial defect detection, visual "
                          "anomaly detection, 3D inspection and multimodal sensing. It works offline."),
        "", "### " + tr("fr.hw", "This computer"),
        f"- GPU / CUDA: {gpu}",
        f"- Anomalib (PaDiM, PatchCore, EfficientAD, AnomalyDINO, WinCLIP): {yes if info['anomalib'] else no}",
        f"- 3D (Open3D, PyVista): {yes if info['open3d'] else no}",
        "", "### " + tr("fr.data", "Data and models"),
        f"- {tr('fr.synthetic', 'Synthetic Educational Data (not a real benchmark)')}: {yes if ds.get('SYN') else no}",
        f"- {tr('fr.real', 'Real research datasets')}: " + (", ".join(f"{k}: {', '.join(v)}" for k, v in real.items())
                                                             if real else no + " " + tr("fr.real_none", "none yet — optional; see DATASETS.md for the official sources")),
        f"- {tr('fr.stored', 'Stored verified experiment results')}: {info['stored_results']}",
        f"- {tr('fr.fitted', 'Fitted models on this computer')}: {info['fitted']}",
        "", "### " + tr("fr.next", "Recommended first exercise"),
        tr("fr.next_text", "Open **Start Learning → 1 · Introduction**, then **Exercise 1 — Classical Defect "
                           "Detection** on the synthetic data. No dataset or GPU is needed."),
    ])


class FirstRunDialog(QDialog):
    def __init__(self, main, info: dict | None = None):
        super().__init__(main)
        self.main = main
        self.setWindowTitle(tr("fr.title", "IDD Lab — first run"))
        self.resize(760, 640)
        lay = QVBoxLayout(self)
        self.text = markdown_browser(summary_markdown(info or capability_summary()))
        lay.addWidget(self.text, 1)
        self.student = QRadioButton(tr("fr.mode_student", "Student Guided Mode — learning content, synthetic demos, "
                                                          "stored results; real data and models when available"))
        self.instructor = QRadioButton(tr("fr.mode_instructor", "Instructor Workshop Mode — full verified installation "
                                                                "(real datasets and prepared checkpoints required)"))
        self.student.setChecked(True)
        g = QButtonGroup(self)
        for b in (self.student, self.instructor):
            g.addButton(b)
            lay.addWidget(b)
        row = QHBoxLayout()
        self.start_btn = QPushButton("★ " + tr("fr.start", "Start Learning"))
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(lambda: self._finish(True))
        close = QPushButton(tr("fr.close", "Close"))
        close.clicked.connect(lambda: self._finish(False))
        row.addStretch(1)
        row.addWidget(close)
        row.addWidget(self.start_btn)
        lay.addLayout(row)

    def _finish(self, go_learning: bool):
        from core import settings
        s = settings.get()
        s.first_run_done = True
        s.profile = "instructor" if self.instructor.isChecked() else "student"
        s.workshop_mode = self.instructor.isChecked()
        settings.save(s)
        if go_learning:
            self.main.navigate("learning", topic="T1")
        self.accept()


def should_show() -> bool:
    from core import profile, settings
    return not settings.get().first_run_done and profile.resolve() == "student"
