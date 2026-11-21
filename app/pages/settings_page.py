"""Settings: preferences, Dataset Manager, models, diagnostics, readiness, experiment log."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMessageBox,
                               QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from app.i18n import tr
from app.widgets.common import PageHeader, amp, muted
from app.workers.task import start
from core import settings
from core.datasets import manager, registry as dreg
from core.datasets.manager import GB


class GeneralTab(QWidget):
    def __init__(self, main):
        super().__init__()
        s = settings.get()
        f = QFormLayout(self)
        self.lang = QComboBox()
        self.lang.addItem("English", "en")
        self.lang.addItem("Español", "es")
        self.lang.setCurrentIndex(self.lang.findData(s.language))
        self.theme = QComboBox()
        self.theme.addItem("Dark", "dark")
        self.theme.addItem("Light", "light")
        self.theme.setCurrentIndex(self.theme.findData(s.theme))
        self.device = QComboBox()
        for d in ("auto", "cuda", "cpu"):
            self.device.addItem(d, d)
        self.device.setCurrentIndex(self.device.findData(s.device))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.setValue(s.seed)
        self.margin = QDoubleSpinBox()
        self.margin.setRange(1, 500)
        self.margin.setValue(s.min_free_space_gb)
        self.root = QLineEdit(str(s.data_root_path))
        browse = QPushButton(tr("set.browse", "Browse…"))
        browse.clicked.connect(self._browse)
        rr = QHBoxLayout()
        rr.addWidget(self.root, 1)
        rr.addWidget(browse)
        f.addRow(tr("set.language", "Language (restart)"), self.lang)
        f.addRow(tr("set.theme", "Theme"), self.theme)
        f.addRow(tr("set.device", "Compute device"), self.device)
        f.addRow(tr("set.seed", "Random seed"), self.seed)
        f.addRow(tr("set.margin", "Keep free on data drive (GB)"), self.margin)
        f.addRow(tr("set.data_root", "Data root (may be on another drive)"), rr)
        from PySide6.QtWidgets import QCheckBox
        self.workshop = QCheckBox(tr("set.workshop", "Workshop offline demo profile"))
        self.workshop.setToolTip(tr("set.workshop_tip", "Methods Laboratory and Compare Methods use Workshop Verified models "
                                                        "and ask before any training. Also: start.ps1 -Workshop"))
        self.workshop.setChecked(s.workshop_mode)
        f.addRow(tr("set.profile", "Profile"), self.workshop)
        from core import profile as _profile
        self.edition = QComboBox()
        for key, label in (("auto", tr("set.ed_auto", "Automatic (instructor if verified checkpoints are present)")),
                           ("student", tr("set.ed_student", "Student Edition")),
                           ("instructor", tr("set.ed_instructor", "Instructor Edition"))):
            self.edition.addItem(label, key)
        self.edition.setCurrentIndex(max(0, self.edition.findData(getattr(s, "profile", "auto"))))
        self.edition.setToolTip(tr("set.ed_tip", "Currently resolved: {p}. Changes readiness checks and the first-run "
                                                 "assistant; never deletes data or models.").format(p=_profile.resolve()))
        f.addRow(tr("set.edition", "Edition"), self.edition)
        save = QPushButton(tr("set.save", "Save"))
        save.setObjectName("Primary")
        save.clicked.connect(lambda: self._save(main))
        f.addRow("", save)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Data root", self.root.text())
        if d:
            self.root.setText(d)

    def _save(self, main):
        s = settings.get()
        lang_changed = s.language != self.lang.currentData()
        s.language, s.theme, s.device = self.lang.currentData(), self.theme.currentData(), self.device.currentData()
        s.seed, s.min_free_space_gb = self.seed.value(), self.margin.value()
        s.workshop_mode = self.workshop.isChecked()
        s.profile = self.edition.currentData()
        s.data_root = "" if Path(self.root.text()) == settings.Settings().data_root_path else self.root.text()
        settings.save(s)
        from PySide6.QtWidgets import QApplication
        from app.ui import theme
        theme.apply(QApplication.instance(), s.theme)
        main.statusBar().showMessage(tr("set.saved", "Settings saved") + (" — restart to apply language" if lang_changed else ""), 8000)


class DatasetsTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QVBoxLayout(self)
        lay.addWidget(muted(tr("dm.intro", "IDD Lab never redistributes datasets. Download them from the official sources "
                                "(links in the Atlas), then import the archive here; only the selected categories are extracted. "
                                "Transfers > 1 GB always require your explicit approval.")))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Dataset", "Status", "Installed categories", "Local folder", "Access"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.import_btn = QPushButton(tr("dm.import", "Import downloaded archive…"))
        self.folder_btn = QPushButton(tr("dm.folder", "Use existing extracted folder…"))
        self.mulsen_btn = QPushButton(tr("dm.mulsen", "Fetch MulSen-AD categories (HTTP range)…"))
        self.syn_btn = QPushButton(tr("dm.synthetic", "Generate SYNTHETIC demo data"))
        self.refresh_btn = QPushButton(tr("dm.refresh", "Refresh"))
        for b in (self.import_btn, self.folder_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        row2 = QHBoxLayout()                       # second row: long (translated) labels fit 1280-px screens
        for b in (self.mulsen_btn, self.syn_btn, self.refresh_btn):
            row2.addWidget(b)
        row2.addStretch(1)
        lay.addLayout(row2)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.msg = muted("")
        self.cancel = QPushButton(tr("lab.cancel", "Cancel"))
        self.cancel.setEnabled(False)
        r2 = QHBoxLayout()
        r2.addWidget(self.prog, 1)
        r2.addWidget(self.cancel)
        lay.addLayout(r2)
        lay.addWidget(self.msg)
        self.import_btn.clicked.connect(self._import)
        self.folder_btn.clicked.connect(self._folder)
        self.mulsen_btn.clicked.connect(self._mulsen)
        self.syn_btn.clicked.connect(self._synthetic)
        self.refresh_btn.clicked.connect(self.refresh)
        self.task = None
        self.cancel.clicked.connect(lambda: self.task and self.task.cancel())
        self.refresh()

    def refresh(self):
        dreg._INDEX_CACHE.clear()
        ids = [d for d in dreg.adapters()]
        self.table.setRowCount(len(ids))
        for i, did in enumerate(ids):
            rec = self.main.catalog.datasets.get(did, {})
            vals = [f"{did} {rec.get('acronym', 'SYNTHETIC demo')}", dreg.installation_status(did),
                    ", ".join(dreg.installed_categories(did)), str(dreg.dataset_root(did)), rec.get("access_method", "generated locally")]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setToolTip(str(v))
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()

    def _pick_dataset(self) -> str | None:
        ids = [d for d in dreg.adapters() if d != "SYN"]
        labels = [f"{d} {self.main.catalog.datasets.get(d, {}).get('acronym', d)}" for d in ids]
        lab, ok = QInputDialog.getItem(self, "Dataset", "Which dataset is this archive/folder?", labels, 0, False)
        return ids[labels.index(lab)] if ok else None

    def _progress(self, f, m):
        self.prog.setValue(int(1000 * f))
        self.msg.setText(m)

    def _done(self, res):
        self.cancel.setEnabled(False)
        self.msg.setText(f"✓ {json.dumps(res, default=str)[:300]}")
        self.refresh()

    def _err(self, m, tb):
        self.cancel.setEnabled(False)
        print(tb)
        self.main.show_error("Dataset Manager", m, "Check that the archive matches the selected dataset and is not corrupted.")

    def _import(self):
        did = self._pick_dataset()
        if not did:
            return
        f, _ = QFileDialog.getOpenFileName(self, "Archive", str(Path.home() / "Downloads"),
                                           "Archives (*.zip *.tar *.tar.gz *.tgz *.tar.xz *.txz)")
        if not f:
            return
        arch = Path(f)
        known = self.main.catalog.datasets[did].get("categories") or []
        try:
            found = manager.detect_categories(manager.list_archive(arch), known)
        except Exception as e:
            self.main.show_error("Import", str(e), "Unsupported or corrupted archive.")
            return
        if not found:
            self.main.show_error("Import", "No known categories found in archive.", f"Expected one of: {', '.join(known)}")
            return
        cats = sorted(found)
        sel, ok = QInputDialog.getText(self, "Categories", f"Categories found: {', '.join(cats)}\n"
                                       "Enter the categories to extract (comma-separated), or leave empty for all:")
        if not ok:
            return
        chosen = [c.strip() for c in sel.split(",") if c.strip()] or None
        dest = dreg.dataset_root(did)
        good, msg = manager.check_space(dest, arch.stat().st_size * 2, settings.get().min_free_space_gb)
        if not good:
            self.main.show_error("Disk space", msg, "Choose a data root on another drive in General settings.")
            return
        self.cancel.setEnabled(True)
        self.task = start(manager.import_archive, archive=arch, dataset_root=dest, categories=chosen,
                          known_categories=known, on_done=self._done, on_error=self._err, on_progress=self._progress)

    def _folder(self):
        did = self._pick_dataset()
        if not did:
            return
        d = QFileDialog.getExistingDirectory(self, "Dataset folder (containing the category folders)")
        if not d:
            return
        a = dreg.get(did)
        if not a.list_categories(Path(d)):
            self.main.show_error("Folder", f"No {did} categories recognised in {d}.", "Select the folder that directly contains the category folders.")
            return
        s = settings.get()
        s.recent.setdefault("dataset_roots", {})[did] = d
        settings.save(s)
        self.refresh()

    def _mulsen(self):
        from scripts.prepare_datasets import MULSEN_URL
        from core.datasets import remote_zip as rz
        cats, ok = QInputDialog.getText(self, "MulSen-AD", "Categories (comma-separated):", text="capsule, nut, flat_pad")
        if not ok:
            return
        cats = [c.strip() for c in cats.split(",") if c.strip()]

        def plan_job(progress, cancel):
            rf, zf = rz.list_remote(MULSEN_URL)
            infos = rz.select(zf, [f"MulSen_AD/{c}" for c in cats])
            return rf, infos, rz.plan(infos)

        def planned(out):
            rf, infos, p = out
            dest = dreg.dataset_root("D06")
            good, msg = manager.check_space(dest, p["uncompressed_bytes"], settings.get().min_free_space_gb)
            text = (f"Source: {MULSEN_URL}\nAccess: public Hugging Face dataset linked from the official repository.\n"
                    f"Licence: not stated by the authors → local research/teaching use only; never redistribute.\n\n"
                    f"Files: {p['files']}\nTransfer ≈ {p['compressed_bytes'] / GB:.2f} GB · extracted ≈ {p['uncompressed_bytes'] / GB:.2f} GB\n"
                    f"Destination: {dest}\n{msg}\n\nProceed?")
            if not good:
                self.main.show_error("Disk space", msg)
                return
            title = "Large download (> 1 GB) — approval required" if manager.requires_approval(p["compressed_bytes"]) else "Download plan"
            if QMessageBox.question(self, title, text) != QMessageBox.Yes:
                return
            self.cancel.setEnabled(True)
            self.task = start(rz.extract_members, rf=rf, infos=infos, dest=dest, strip_prefix="MulSen_AD",
                              on_done=lambda files: (manager.write_install_record(dest, {"source": MULSEN_URL, "categories": cats,
                                                                                          "files": len(files)}), self._done({"files": len(files)})),
                              on_error=self._err, on_progress=self._progress)
        self.msg.setText("Reading archive index (≈1 MB)…")
        start(plan_job, on_done=planned, on_error=self._err)

    def _synthetic(self):
        from core.datasets.synthetic import ensure_generated
        start(lambda progress, cancel: str(ensure_generated(settings.get().data_root_path)), on_done=self._done, on_error=self._err)


class DiagnosticsTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        b1 = QPushButton(amp(tr("diag.hw", "Hardware & software report")))
        b2 = QPushButton(tr("diag.ready", "Offline readiness check (workshop)"))
        b3 = QPushButton(amp(tr("diag.catalog", "Research catalog & link audit")))
        for b in (b1, b2, b3):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.out = QTextBrowser()
        self.out.setOpenExternalLinks(True)
        lay.addWidget(self.out, 1)
        b1.clicked.connect(self._hw)
        b2.clicked.connect(self._ready)
        b3.clicked.connect(self._catalog)

    def _hw(self):
        from core.system.diagnostics import full_report
        self.out.setPlainText("collecting…")
        start(lambda progress, cancel: full_report(settings.get().data_root_path),
              on_done=lambda r: self.out.setPlainText(json.dumps(r, indent=2)))

    def _ready(self):
        from core.system.readiness import run_checks

        def show(checks):
            html = "<table cellpadding=4>" + "".join(
                f"<tr><td>{'✅' if c.ok else ('❌' if c.required else '⚠️')}</td><td><b>{c.name}</b></td><td>{c.detail}</td></tr>"
                for c in checks) + "</table>"
            n_bad = sum(1 for c in checks if c.required and not c.ok)
            self.out.setHtml(f"<h3>{'Ready for offline workshop' if not n_bad else f'{n_bad} required item(s) missing'}</h3>" + html)
        self.out.setPlainText("checking…")
        start(lambda progress, cancel: run_checks(), on_done=show)

    def _catalog(self):
        c = self.main.catalog
        audit = c.link_audit
        lines = [f"<h3>Catalog</h3>{len(c.datasets)} datasets · {len(c.algorithms)} methods · {len(c.publications)} publications",
                 "<h4>Validation problems</h4>" + ("<br>".join(c.problems) if c.problems else "none")]
        if audit:
            from collections import Counter
            cnt = Counter(v.get("status") for v in audit.values())
            lines.append("<h4>Link audit</h4>" + ", ".join(f"{k}: {v}" for k, v in cnt.items()))
            bad = [(u, v) for u, v in audit.items() if v.get("status") in ("error", "not_found")]
            lines += [f"<a href='{u}'>{u}</a> — {v.get('status')} {v.get('http_status')}" for u, v in bad]
        else:
            lines.append("<h4>Link audit</h4>not run yet — python -m scripts.verify_research_links")
        self.out.setHtml("<br>".join(lines))


class ExperimentsTab(QWidget):
    def __init__(self, main):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(muted(tr("log.intro", "Every run is stored with its provenance category. Categories are never "
                                            "mixed: author_reported · reproduced · local_educational · synthetic_demo.")))
        self.filter = QComboBox()
        self.filter.addItem(tr("log.all", "all"), None)
        for c in ("author_reported", "reproduced", "local_educational", "synthetic_demo"):
            self.filter.addItem(c, c)
        lay.addWidget(self.filter)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([tr("log.date", "Date"), tr("log.category", "Category"), tr("log.method", "Method"),
                                              tr("log.dataset", "Dataset"), tr("xp.m_img_auroc", "Image AUROC"),
                                              tr("xp.m_pix_auroc", "Pixel AUROC"), tr("log.protocol", "Protocol"),
                                              tr("log.run_id", "Run id")])
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)
        b = QPushButton(tr("mm.refresh", "Refresh"))
        b.clicked.connect(self.refresh)
        lay.addWidget(b)
        self.filter.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self):
        import time
        from core.experiments.store import ExperimentStore
        runs = ExperimentStore().runs(self.filter.currentData())
        self.table.setRowCount(len(runs))
        for i, r in enumerate(runs):
            m = r["metrics"]
            f = lambda v: "—" if v is None else f"{v:.4f}"
            vals = [time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created"])), r["category"], r["method_id"],
                    f"{r['dataset_id']}/{r['dataset_category']}", f(m.get("image_auroc")), f(m.get("pixel_auroc")),
                    r["protocol"], r["run_id"]]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setToolTip(str(v))
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()


class SettingsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(PageHeader(tr("nav.settings", "Settings"), tr("set.subtitle", "Preferences, datasets, models and diagnostics")))
        self.tabs = QTabWidget()
        self.tabs.addTab(GeneralTab(main), tr("set.tab_general", "General"))
        self.dm = DatasetsTab(main)
        self.tabs.addTab(self.dm, tr("set.tab_datasets", "Dataset Manager"))
        self.diag = DiagnosticsTab(main)
        self.tabs.addTab(self.diag, amp(tr("set.tab_diag", "Diagnostics & readiness")))
        self.tabs.addTab(ExperimentsTab(main), tr("set.tab_exp", "Experiment log"))
        lay.addWidget(self.tabs, 1)

    def apply_preset(self, tab: str | None = None, **_):
        if tab == "datasets":
            self.tabs.setCurrentWidget(self.dm)
        elif tab == "diagnostics":
            self.tabs.setCurrentWidget(self.diag)
