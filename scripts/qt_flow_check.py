"""Real Qt user-flow check of the workshop path, with the offline demo profile (IDD_WORKSHOP=1).

  python -m scripts.qt_flow_check [--out cache/qt_flow]

Drives the full application window like a participant: Guided Workshop → activity A → Educational
Experiments (resolution, illumination, metrics) → Methods Laboratory (load a Workshop Verified model,
run it) → Compare Methods (verified models) → offline readiness check. Each step is checked and a
screenshot is written to ``--out`` (git-ignored: it contains third-party dataset imagery).
Exit status 1 if any check fails. No training is started (a training request would fail the run).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from core import paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.CACHE / "qt_flow"))
    ap.add_argument("--lang", default="en", choices=["en", "es"])
    a = ap.parse_args(argv)
    os.environ["IDD_WORKSHOP"] = "1"
    os.environ.setdefault("TORCH_HOME", str(paths.CHECKPOINTS / "torch"))
    os.environ.setdefault("HF_HOME", str(paths.CHECKPOINTS / "huggingface"))
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import QApplication
    from app import i18n
    from app.ui import theme
    from core import settings
    app = QApplication(sys.argv)
    s = settings.load()
    s.language = a.lang                    # content (Markdown/YAML) follows settings.language; not saved
    i18n.init(a.lang)
    from app.i18n import tr
    theme.apply(app, s.theme)
    from app.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1680, 1000)
    w.show()
    out = os.path.abspath(os.path.join(a.out, a.lang))
    os.makedirs(out, exist_ok=True)
    log: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        log.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    def shot(name):
        app.processEvents()
        w.grab().save(os.path.join(out, f"{name}.png"))
        check(f"window keeps its size ({name})", w.width() <= 1700 and w.height() <= 1020, f"{w.width()}×{w.height()}")

    def wait(cond, timeout=180.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            app.processEvents()
            try:
                if cond():
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)
        return False

    def run():
        check("workshop profile active", settings.get().workshop_mode)
        # ---- Guided Workshop → activity A → its action button
        w.navigate("workshop")
        ws = w.page("workshop")
        row = [x["id"] for x in ws.acts].index("A")
        ws.list.setCurrentRow(row)
        txt = ws.text.toPlainText()
        check("activity A shows verified results", tr("ws.results", "Real results (Workshop Verified, test_public)")[:10] in txt
              and "1024×256" in txt)
        shot("01_workshop_activity_A")
        btn = next(ws.actions.itemAt(i).widget() for i in range(ws.actions.count())
                   if ws.actions.itemAt(i).widget() is not None)
        btn.click()
        xp = w.page("experiments")
        check("action opens Educational Experiments → A", w.stack.currentWidget() is xp and xp.tabs.currentIndex() == 0)
        # ---- A. resolution
        r = xp.res
        i = next(k for k in range(r.sample.count()) if r.sample.itemText(k).startswith(tr("xp.anomalous", "anomalous"))
                 and "regular" in r.sample.itemText(k))
        r.sample.setCurrentIndex(i)
        r.run.click()
        ok = wait(lambda: r.run.isEnabled() and r.live.rowCount() > 0 and r.live.item(1, 1) is not None)
        check("A: both PatchCore models ran on a real image", ok, r.msg.text())
        if ok:
            check("A: decisions shown for both models", all(r.live.item(2, j) is not None for j in (1, 2)),
                  f"{r.live.item(2, 1).text()} / {r.live.item(2, 2).text()}")
        shot("02_resolution")
        # ---- B. illumination
        xp.tabs.setCurrentIndex(1)
        il = xp.ill
        il.model.setCurrentIndex(max(0, il.model.findText("PatchCore · sheet_metal", Qt.MatchStartsWith)))
        k = next((j for j in range(il.scenes.count()) if tr("xp.flips", "decision flips") in il.scenes.item(j).text()), 0)
        il.scenes.setCurrentRow(k)
        il.maps_btn.click()
        ok = wait(lambda: il.maps_btn.isEnabled() and tr("xp.same_scale", "Same colour scale")[:10] in il.msg.text())
        check("B: maps computed for every lighting condition of a scene", ok, il.scene_title.text()[:90])
        check("B: scene-aware report with cluster bootstrap", tr("xp.scene_analysis", "Scene-aware analysis") in il.report.toPlainText())
        shot("03_illumination")
        # ---- C. metrics
        w.navigate("experiments", tab="metrics", preset_id="d05-fruit_jelly-padim-256")
        m = xp.met
        f1 = m.tbl.item(7, 1).text()
        check("C: PaDiM fruit_jelly valid-protocol F1 = 0", f1 == "0.000", f"F1={f1}, {m.summary.text()[:80]}")
        shot("04_metrics_valid")
        m.slider.setValue(max(0, m.slider0 - 350))
        check("C: retrospective banner when exploring", tr("xp.retro", "RETROSPECTIVE")[:12] in m.banner.text())
        shot("05_metrics_exploration")
        m.reset.click()
        check("C: reset returns to the valid protocol", tr("xp.valid_note", "Valid protocol")[:12] in m.banner.text())
        check("C: table and plot labels translated", m.tbl.item(0, 0).text() == tr("xp.t_thr", "Threshold")
              and m.fig.axes[0].get_xlabel() == tr("xp.ax_score", "image anomaly score"), m.tbl.item(0, 0).text())
        # ---- Methods Laboratory: Model Registry
        w.navigate("lab", dataset_id="D05", method_id="patchcore")
        lab = w.page("lab")
        lab.browser.cat.setCurrentIndex(lab.browser.cat.findData("sheet_metal"))
        wait(lambda: lab.browser.index is not None and lab.browser.index.category == "sheet_metal")
        check("Lab: registry lists Workshop Verified first", lab.ckpt.itemText(0).startswith("[Workshop Verified]"),
              lab.ckpt.itemText(0))
        asked = []
        lab._ask_train = lambda text, can_load: asked.append(text) or "load"
        lab._fit()                    # 'Fit images' = 60 → no exact cache → must offer the checkpoint
        check("Lab: Fit with default 60 images asks instead of training", bool(asked))
        ok = wait(lambda: lab.session is not None and lab.session.method.fitted and lab.load_btn.isEnabled())
        check("Lab: verified model loaded (no training)", ok and len(lab.session.fit_ids) == 137,
              lab.progress_msg.text()[:100])
        k = next(j for j, x in enumerate(lab.browser.samples) if x.label == "bad")
        lab.browser.list.setCurrentRow(k)
        lab._run()
        ok = wait(lambda: lab.result is not None)
        check("Lab: inference on a real test image", ok, f"score={lab.result.score:.2f}" if ok else "")
        shot("06_lab_registry")
        # ---- Compare Methods with verified models
        w.navigate("compare", dataset_id="D05")
        cmp = w.page("compare")
        cmp.browser.cat.setCurrentIndex(cmp.browser.cat.findData("fruit_jelly"))
        wait(lambda: cmp.browser.index is not None and cmp.browser.index.category == "fruit_jelly")
        for j in range(cmp.mlist.count()):
            it = cmp.mlist.item(j)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in ("patchcore", "padim", "efficientad", "anomalydino")
                             else Qt.Unchecked)
        trained = []
        cmp._confirm_training = lambda mids: trained.append(mids) or False
        cmp.browser.list.setCurrentRow(next(j for j, x in enumerate(cmp.browser.samples) if x.label == "bad"))
        cmp._run()
        ok = wait(lambda: len(cmp.rows) == 4, 300)
        check("Compare: 4 verified models, no training requested", ok and not trained,
              f"fit images {[r['n_fit_images'] for r in cmp.rows]}")
        shot("07_compare_verified")
        # ---- every page must fit a 1280-px (logical) laptop screen
        for key in ("dashboard", "atlas", "explorer", "library", "lab3d", "workshop", "settings"):
            w.navigate(key)
            app.processEvents()
        mw = w.minimumSizeHint().width()
        check("window minimum width fits 1280-px screens (all pages loaded)", mw <= 1280, f"{mw} px")
        # ---- readiness
        from core.system.readiness import run_checks
        checks = run_checks(deep=True)
        bad = [c for c in checks if c.required and not c.ok]
        vc = next(c for c in checks if c.name.startswith("Workshop Verified"))
        check("Readiness: all required checks OK", not bad, "; ".join(c.name for c in bad))
        check("Readiness: verified presets (SHA-256)", vc.ok, vc.detail)
        w.navigate("settings", tab="diagnostics")
        shot("08_settings")
        n_fail = sum(not ok for _, ok, _ in log)
        print(f"\n{len(log) - n_fail}/{len(log)} checks passed · screenshots in {out}", flush=True)
        app.exit(1 if n_fail else 0)

    QTimer.singleShot(800, run)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
