"""Methods Library visual explanations: content completeness (EN/ES), worked examples, Qt widgets."""

from __future__ import annotations

import re

import numpy as np
import pytest
import yaml

from core import paths

PRIORITY = ["reference_difference", "otsu", "morphology", "canny", "gabor", "adaptive_threshold",
            "padim", "patchcore", "efficientad", "winclip", "anomalydino", "pc_deviation", "multimodal_fusion",
            "cnn_classification", "yolo", "unet"]
SUPERVISED = ["cnn_classification", "yolo", "unet"]
KINDS = {"data", "train", "label", "pre", "feature", "model", "head", "score", "decision", "step"}
PH = re.compile(r"\{[^}]*\}")


def _load(lang):
    with (paths.EDUCATION / lang / "method_visuals.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def examples():
    from core.visualization import method_examples as me
    return {m: me.build(m) for m in PRIORITY}


# ------------------------------------------------------------------------------ content
@pytest.mark.parametrize("lang", ["en", "es"])
def test_every_priority_method_has_complete_content(lang):
    data = _load(lang)
    from core.research.catalog import get_catalog
    cat = get_catalog()
    for m in PRIORITY:
        assert m in cat.algorithms, m
        v = data[m]
        assert all(v["summary"].get(k) for k in ("problem", "idea", "when", "limits", "io", "training")), m
        nodes = [n for lane in v["diagram"]["lanes"] for n in lane["nodes"]]
        ids = [n["id"] for n in nodes]
        assert len(nodes) >= 4 and len(ids) == len(set(ids)), m
        assert all(n["kind"] in KINDS and n["title"] and n["text"] for n in nodes), m
        for link in v["diagram"].get("links", []):
            assert link["from"] in ids and link["to"] in ids, (m, link)
        assert v["strengths"] and v["limitations"] and v["example"]["interpretation"], m


def test_spanish_mirrors_english_structure():
    en, es = _load("en"), _load("es")
    assert set(en) == set(es)
    for m in en:
        lanes = lambda d: [[(n["id"], n["kind"]) for n in lane["nodes"]] for lane in d["diagram"]["lanes"]]
        assert lanes(en[m]) == lanes(es[m]), m
        links = lambda d: [(lk["from"], lk["to"]) for lk in d["diagram"].get("links", [])]
        assert links(en[m]) == links(es[m]), m
        grid = lambda d: (d["diagram"].get("layout"), [n.get("pos") for lane in d["diagram"]["lanes"] for n in lane["nodes"]],
                          d["diagram"].get("edges"), [nt["pos"] for nt in d["diagram"].get("notes", [])])
        assert grid(en[m]) == grid(es[m]), m
        assert ("legend" in en[m]["example"]) == ("legend" in es[m]["example"]), m
        ce, cs = en[m]["example"]["captions"], es[m]["example"]["captions"]
        assert set(ce) == set(cs), m
        for k in ce:
            assert set(PH.findall(ce[k])) == set(PH.findall(cs[k])), (m, k)
        assert set(PH.findall(en[m]["example"]["interpretation"])) == set(PH.findall(es[m]["example"]["interpretation"])), m
        assert ("note" in en[m]["example"]) == ("note" in es[m]["example"]), m
        assert es[m]["summary"]["problem"] != en[m]["summary"]["problem"], f"{m}: not translated"


# ------------------------------------------------------------------------------ examples
def test_examples_have_captions_and_valid_images(examples):
    from app.widgets.method_visuals import fill
    for lang in ("en", "es"):
        data = _load(lang)
        for m, ex in examples.items():
            caps = data[m]["example"]["captions"]
            for s in ex.stages + ex.outputs:
                assert s.key in caps, (lang, m, s.key)
                assert s.image.dtype == np.uint8 and s.image.ndim == 3 and s.image.shape[2] == 3, (m, s.key)
                txt = fill(caps[s.key], ex.facts)
                assert "{" not in txt, (lang, m, s.key, txt)      # every placeholder resolved
            assert "{" not in fill(data[m]["example"]["interpretation"], ex.facts), (lang, m)


def test_classical_examples_use_the_real_implementation(examples):
    """The replayed intermediate stage equals the real IDD Lab output (no algorithm drift)."""
    for m in ("reference_difference", "otsu", "morphology", "canny", "adaptive_threshold"):
        ex = examples[m]
        assert ex.provenance == "implementation"
        np.testing.assert_allclose(ex.facts["_replay"], ex.facts["_real_map"], atol=1e-4, err_msg=m)
    assert examples["gabor"].provenance == "implementation"
    for m in ("padim", "patchcore", "efficientad", "winclip", "anomalydino", "pc_deviation", "multimodal_fusion",
              *SUPERVISED):
        assert examples[m].provenance == "toy", m


def test_examples_find_their_defect(examples):
    """Each worked example must actually illustrate a detection (sanity of the didactic story)."""
    f = examples["reference_difference"].facts
    assert f["unaligned_px"] > 2 * f["aligned_px"]               # alignment matters
    assert examples["adaptive_threshold"].facts["global_pct"] > 30  # global threshold fails under the gradient
    pc = examples["pc_deviation"].facts
    assert pc["rmse"] < pc["threshold"] < pc["score"] < pc["before"]
    rows = {r[0]: r[1:] for r in examples["multimodal_fusion"].facts["rows"]}
    thr = examples["multimodal_fusion"].facts["threshold"]
    assert rows["good"][3] < thr and rows["rgb"][3] > thr and rows["geo"][3] > thr   # max fusion catches both
    assert rows["geo"][2] < thr                                                    # mean fusion dilutes 3D-only


def test_examples_are_deterministic():
    from core.visualization import method_examples as me
    a = me.BUILDERS["patchcore"](me._t_default)
    b = me.BUILDERS["patchcore"](me._t_default)
    assert all(np.array_equal(x.image, y.image) for x, y in zip(a.stages, b.stages))


# ------------------------------------------------------------------------------ Qt
@pytest.fixture
def library(qtbot):
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("en")
    from app.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1280, 721)                       # 1280×800 screen minus title bar and taskbar
    w.show()
    w.navigate("library")
    yield w, w.page("library")
    w.close()


def test_every_priority_method_shows_diagram_and_example(library):
    w, lib = library
    en = _load("en")
    for m in PRIORITY:
        w.navigate("library", method_id=m)
        assert lib.method_id == m
        nodes = lib.how.diagram.nodes()
        assert [n["id"] for n in nodes] == [n["id"] for lane in en[m]["diagram"]["lanes"] for n in lane["nodes"]]
        assert lib.how.diagram.heightForWidth(700) > 100
        assert not lib.how.diagram.grab().isNull()
        assert lib.summary.cards["problem"].text() == en[m]["summary"]["problem"]
        lib.center.setCurrentIndex(lib.TAB_EXAMPLE)
        assert len(lib.example.gallery.tiles) == len(en[m]["example"]["captions"]) - \
            sum(k.startswith("out_") for k in en[m]["example"]["captions"])
        assert lib.example.interp.text()
        lib.center.setCurrentIndex(lib.TAB_OVERVIEW)


def test_diagram_click_selects_stage(library, qtbot):
    from PySide6.QtCore import QPoint, Qt
    w, lib = library
    w.navigate("library", method_id="patchcore", tab="how")
    d = lib.how.diagram
    _, rects, _, _ = d._layout(d.width())
    r = rects["coreset"]
    qtbot.mouseClick(d, Qt.LeftButton, pos=QPoint(int(r.center().x()), int(r.center().y())))
    assert d.selected == "coreset"
    assert "k-center" in lib.how.details.toPlainText()


def test_library_fits_1280x800(library):
    w, lib = library
    for m in ("patchcore", "multimodal_fusion", "reference_difference", *SUPERVISED):
        for t in range(lib.center.count()):
            w.navigate("library", method_id=m)
            lib.center.setCurrentIndex(t)
            mh = w.minimumSizeHint()
            assert mh.width() <= 1280 and mh.height() <= 721, (m, t, mh)
            assert w.width() == 1280, (m, t, w.width())


def test_method_without_curated_visuals_falls_back(library):
    w, lib = library
    w.navigate("library", method_id="conv_autoencoder")
    assert lib.how.diagram.nodes() and all(n["kind"] == "step" for n in lib.how.diagram.nodes())
    assert not lib.how.caption.isHidden()
    lib.center.setCurrentIndex(lib.TAB_EXAMPLE)
    assert not lib.example.empty.isHidden() and not lib.example.gallery.tiles


def test_spanish_library(qtbot):
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("es")
    try:
        from app.ui.main_window import MainWindow
        w = MainWindow()
        qtbot.addWidget(w)
        w.navigate("library", method_id="canny")
        lib = w.page("library")
        assert [lib.center.tabText(i).replace(" (—)", "") for i in range(lib.center.count())] == \
            ["Resumen", "Cómo funciona", "Paso a paso", "Ejemplo", "Ventajas y límites",
             "Ficha técnica", "Referencias"]
        es = _load("es")["canny"]
        assert lib.summary.cards["idea"].text() == es["summary"]["idea"]
        assert lib.how.diagram.nodes()[0]["title"] == es["diagram"]["lanes"][0]["nodes"][0]["title"]
        lib.center.setCurrentIndex(lib.TAB_EXAMPLE)
        assert lib.example.badge.text().startswith("Implementación real")
        caps = [t.findChildren(type(lib.example.interp))[-1].text() for t in lib.example.gallery.tiles]
        assert caps[0] == es["example"]["captions"]["input"]
        w.close()
    finally:
        i18n.init("en")
