"""Real Qt check of the Methods Laboratory reference comparison (Reference image difference).

  python -m scripts.qt_reference_check [--lang en|es] [--dataset D05] [--category sheet_metal] [--out cache/qt_reference]

Drives the real window at 1280×800 (client area 1280×721 after taskbar and title bar) and checks:
the reference shown is exactly the selected identifier; the difference map comes from the two
displayed images and the algorithm's own preprocessing; changing the reference discards results,
thresholds and evaluations computed with the previous one; a run that finishes after the
reference changed is discarded; test_private images get a prediction only (no ground truth).
Screenshots go to ``--out`` (git-ignored: may contain third-party dataset imagery).
No dataset is downloaded and no model is trained. Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from core import paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.CACHE / "qt_reference"))
    ap.add_argument("--lang", default="en", choices=["en", "es"])
    ap.add_argument("--dataset", default="D05")
    ap.add_argument("--category", default="sheet_metal")
    a = ap.parse_args(argv)
    import cv2
    import numpy as np
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from app import i18n
    from app.ui import theme
    from core import settings
    app = QApplication(sys.argv)
    s = settings.load()
    s.language = a.lang                 # not saved
    i18n.init(a.lang)
    from app.i18n import tr
    theme.apply(app, s.theme)
    from app.ui.main_window import MainWindow
    from core.datasets import registry as dreg
    from core.datasets.io import read_rgb8
    w = MainWindow()
    w.resize(1280, 721)
    w.show()
    out = os.path.abspath(os.path.join(a.out, a.lang))
    os.makedirs(out, exist_ok=True)
    log: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        log.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    def shot(name, widget=None):
        app.processEvents()
        (widget or w).grab().save(os.path.join(out, f"{name}.png"))

    def wait(cond, timeout=120.0):
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

    def gray(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def run():
        ds, cat = a.dataset, a.category
        if cat not in (dreg.installed_categories(ds) or []):
            ds, cat = "SYN", None
        w.navigate("lab", dataset_id=ds, method_id="reference_difference")
        lab = w.page("lab")
        if cat:
            lab.browser.cat.setCurrentIndex(lab.browser.cat.findData(cat))
        wait(lambda: lab.browser.index is not None and (cat is None or lab.browser.index.category == cat))
        idx = lab.browser.index
        check("dataset selected", idx is not None, f"{idx.dataset_id}/{idx.category}" if idx else "")
        rv = lab.refview
        check("comparison tab shown for Reference image difference",
              lab.center_tabs.currentWidget() is rv and lab.center_tabs.isTabVisible(1))
        check("reference list filled before any run", lab.ref_combo.count() == min(200, len(idx.train_normals())),
              f"{lab.ref_combo.count()} train/good images")
        lab.ref_combo.setCurrentIndex(2)
        app.processEvents()
        sid = lab.ref_combo.currentData()
        ref = next(x for x in idx.samples if x.sample_id == sid)
        check("displayed reference = selected identifier",
              rv.ref_sample.sample_id == sid == lab.session.reference.sample_id and sid in rv.ref_info.text(),
              sid)
        check("displayed reference pixels = file of that identifier",
              np.array_equal(rv.ref_image, read_rgb8(ref.primary_path)) and np.array_equal(rv.p_ref._img, rv.ref_image))
        check("reference panel shows dataset, category and split",
              all(x in rv.ref_info.text() for x in (idx.category, ref.split)))
        bad = next(j for j, x in enumerate(lab.browser.samples) if x.label == "bad")
        lab.browser.list.setCurrentRow(bad)
        app.processEvents()
        shot("01_reference_selected")
        lab._run()
        ok = wait(lambda: lab.result is not None)
        r = lab.result
        check("comparison ran", ok, f"score={r.score:.2f}" if ok else lab.progress_msg.text())
        st = r.extras["stages"]
        rg = gray(rv.ref_image)
        if rg.shape != st["test_gray"].shape:
            rg = cv2.resize(rg, st["test_gray"].shape[::-1], interpolation=cv2.INTER_AREA)
        check("algorithm input = displayed reference", np.array_equal(st["reference_gray"], rg))
        check("algorithm input = displayed test image", np.array_equal(st["test_gray"], gray(rv.test_image))
              and np.array_equal(rv.test_image, lab.image))
        check("difference map = |aligned+blurred test − blurred reference|",
              np.array_equal(r.anomaly_map, cv2.absdiff(st["test_pre"], st["reference_pre"]).astype(np.float32)))
        k = st["blur_kernel"]
        check("pre-blur reproduced from the displayed aligned test",
              k <= 1 or np.array_equal(st["test_pre"], cv2.GaussianBlur(st["test_aligned"], (k, k), 0)))
        check("aligned panel shows the warped test", np.array_equal(rv.p_aligned._img[..., 0], st["test_aligned"]))
        check("threshold before calibration = method parameter (exact, not slider-quantised)",
              lab._thr_value() == r.threshold == 30.0, f"thr={lab._thr_value():.6g}")
        check("mask panel = the algorithm's own mask (map > 30)",
              rv.shown_op == ">" and np.array_equal(rv.shown_mask, r.anomaly_map > 30.0),
              f"pixels exactly at 30 (> vs ≥): {int((r.anomaly_map == 30).sum())}")
        check("result tagged with its reference", lab.result_ref == sid and rv.result_ref_id == sid)
        shot("02_compared")
        rv.enlarge_all()
        app.processEvents()
        dlg = rv.dialog
        check("enlarge shows all 5 stages with synchronised zoom", len(dlg.views) == 5)
        v0 = dlg.views[0]
        v0.scale(2.0, 2.0)                 # what ImageView.wheelEvent does
        v0._fitted = False
        v0.viewChanged.emit()
        app.processEvents()
        check("zoom propagates to every enlarged view",
              all(abs(v.transform().m11() - dlg.views[0].transform().m11()) < 1e-6 for v in dlg.views))
        shot("03_enlarged", dlg)
        dlg.close()
        # ---- calibrate + evaluate with this reference, then change it
        lab._calibrate()
        wait(lambda: lab.session.calibration.image_threshold is not None and lab.cal_btn.isEnabled())
        check("calibrated with this reference", lab._cal_ref == sid, lab.session.calibration.source[:60])
        score1 = r.score
        lab._step_ref(+1)
        app.processEvents()
        sid2 = lab.ref_combo.currentData()
        check("◀ ▶ changes the selected reference", sid2 != sid and rv.ref_sample.sample_id == sid2, sid2)
        check("old result discarded on reference change", lab.result is None and rv.result is None and rv.stale)
        check("old calibration discarded on reference change", lab.session.calibration.image_threshold is None)
        check("no stale map displayed", rv.p_diff._img is None and rv.p_mask._img is None and lab.v_map.view._img is None)
        shot("04_reference_changed")
        lab._run()
        wait(lambda: lab.result is not None)
        check("new comparison uses the new reference", lab.result_ref == sid2 and len(rv.log) == 2,
              f"scores {score1:.2f} → {lab.result.score:.2f}")
        for _ in range(2):
            lab._step_ref(+1)
            lab._run()
            wait(lambda: lab.result is not None)
        check("table lists 4 references", len(rv.log) == 4, ", ".join(f"{x['score']:.1f}" for x in rv.log))
        shot("05_several_references")
        # ---- a run that finishes after the reference changed must be discarded
        lab._run()
        lab._step_ref(+1)
        wait(lambda: lab.run_btn.isEnabled())
        check("run finished after a reference change is discarded", lab.result is None and rv.result is None,
              lab.progress_msg.text()[:70])
        # ---- test_private: prediction only
        priv = [j for j, x in enumerate(lab.browser.samples) if x.split.startswith("test_private")]
        if not priv:
            lab.browser.split.setCurrentIndex(max(0, lab.browser.split.findData("test_private")))
            wait(lambda: any(x.split.startswith("test_private") for x in lab.browser.samples), 10)
            priv = [j for j, x in enumerate(lab.browser.samples) if x.split.startswith("test_private")]
        if priv:
            lab._calibrate()
            wait(lambda: lab.session.calibration.image_threshold is not None and lab.cal_btn.isEnabled())
            lab.browser.list.setCurrentRow(priv[0])
            app.processEvents()
            lab._run()
            wait(lambda: lab.result is not None)
            txt = lab.decision.text()
            check("test_private: decision is a prediction, not ground truth",
                  tr("lab.prediction", "Prediction") in txt and "(anomalous)" not in txt and "(normal)" not in txt, txt)
            check("test_private: warning in the comparison card", "test_private" in rv.info.toPlainText())
            shot("06_test_private")
        else:
            check("test_private split available (skipped: not in this dataset)", True, "skipped")
        mh = w.minimumSizeHint()
        check("window fits 1280×800 (client 1280×721)", mh.width() <= 1280 and mh.height() <= 721,
              f"min {mh.width()}×{mh.height()}")
        check("window kept 1280×721", w.width() == 1280 and w.height() == 721, f"{w.width()}×{w.height()}")
        n_fail = sum(not ok for _, ok, _ in log)
        print(f"\n{len(log) - n_fail}/{len(log)} checks passed · screenshots in {out}", flush=True)
        app.exit(1 if n_fail else 0)

    def safe_run():
        try:
            run()
        except Exception:  # noqa: BLE001 — a crash must fail the check, not leave the window open
            import traceback
            traceback.print_exc()
            check("check script completed", False)
            app.exit(1)

    QTimer.singleShot(800, safe_run)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
