"""Group B (supervised learning) in the Methods Library: toy models genuinely trained on labels, outputs of
the right kind (class / boxes / mask), architecture diagrams, the cross-method comparison (EN/ES), the
Qt panels at 1280×800 and independence from datasets (Student Edition)."""

from __future__ import annotations

import re

import numpy as np
import pytest
import yaml

from core import paths

SUPERVISED = ["cnn_classification", "yolo", "unet"]
PH = re.compile(r"\{[^}]*\}")


def _load(name, lang):
    with (paths.EDUCATION / lang / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def se():
    from core.visualization import supervised_examples as m
    return m


# ------------------------------------------------------------------ the toy models really learn from labels
def test_classifier_learns_image_labels(se):
    m, X, y, pv, yv = se.classifier()
    assert set(np.unique(y)) == {0, 1} and len(X) == se.N_TRAIN
    assert ((pv >= 0.5) == (yv == 1)).mean() >= 0.8                 # validation accuracy (held-out images)


def test_classification_has_no_location(se):
    """Global pooling: moving the defect elsewhere (180° rotation) does not change the answer."""
    m = se.classifier()[0]
    s = se.sample(4245, ("scratch",))
    p = lambda im: float(m.predict_proba(se.cls_vector(im)[None])[0, 1])
    assert p(s.image) >= 0.5
    assert abs(p(s.image) - p(np.ascontiguousarray(np.rot90(s.image, 2)))) < 0.05
    assert p(se.sample(4343).image) < 0.5


def test_detector_learns_boxes(se):
    v = se.detector_validation()
    assert v["n_boxes"] >= 40 and v["precision"] >= 0.8 and v["recall"] >= 0.4
    s = se.sample(6215, ("scratch", "stain"))
    kept = se.nms([c for c in se.detect(s.image)[2] if c.conf >= se.DET_CONF])
    assert kept
    for d in kept:
        x0, y0, x1, y1 = d.box
        assert d.cls in se.CLASSES and 0.0 <= d.conf <= 1.0
        assert 0 <= x0 < x1 < se.SIZE and 0 <= y0 < y1 < se.SIZE
    for i, a in enumerate(kept):                       # NMS: no two boxes of one class overlap strongly
        for b in kept[i + 1:]:
            assert not (a.cls == b.cls and se._iou(a.box, b.box) > 0.3)
    assert any(d.cls == "scratch" for d in kept)


def test_segmentation_skip_connections_help(se):
    v = se.segmenter_validation()
    assert v["n"] >= 10
    assert v["dice_skip"] >= 0.45 and v["dice_skip"] > v["dice_noskip"] + 0.1


def test_box_targets_follow_the_yolo_rule(se):
    """The cell containing a labelled box centre is responsible for it (objectness 1, its class, offsets)."""
    s = se.sample(6215, ("scratch", "stain"))
    obj, cls, box = se._box_targets(s)
    assert obj.sum() <= len(s.boxes) and obj.sum() >= 1
    for i, j in zip(*np.nonzero(obj)):
        assert 0 <= cls[i, j] < len(se.CLASSES)
        assert -0.5 <= box[i, j, 0] <= 0.5 and -0.5 <= box[i, j, 1] <= 0.5 and box[i, j, 2] > 0 and box[i, j, 3] > 0


def test_training_and_example_images_are_disjoint(se):
    tr_, va = se.train_set()
    seeds = {7 * 1000 + i for i in range(len(tr_))} | {8 * 1000 + i for i in range(len(va))}
    assert not seeds & {4245, 4343, 5114, 6215, 7373}


def test_examples_are_deterministic_and_dataset_free(se, monkeypatch):
    """Student Edition: no dataset file is ever read; the same pictures every time."""
    import core.datasets.io as dio

    def no_read(*a, **k):
        raise AssertionError("a dataset file was read")
    monkeypatch.setattr(dio, "read_rgb8", no_read)
    t = lambda k, d: d
    for m in SUPERVISED:
        a, b = se.BUILDERS[m](t), se.BUILDERS[m](t)
        assert [s.key for s in a.stages] == [s.key for s in b.stages]
        assert all(np.array_equal(x.image, y.image) for x, y in zip(a.stages + a.outputs, b.stages + b.outputs)), m
    c = se.comparison(t)
    assert all(c[k].dtype == np.uint8 and c[k].ndim == 3 for k in c if k != "facts")


def test_image_labels_are_ascii(se):
    assert se.ascii_text("raya · picadura · mancha") == "raya  picadura  mancha"
    img = se.draw_boxes(se.sample(1).image, [("spot", (10, 10, 30, 30))], t=lambda k, d: "picadúra")
    assert img.shape == (2 * se.SIZE, 2 * se.SIZE, 3)


# ------------------------------------------------------------------ content: diagrams and comparison
@pytest.mark.parametrize("lang", ["en", "es"])
def test_supervised_content_and_grid_diagrams(lang):
    data = _load("method_visuals.yaml", lang)
    for m in SUPERVISED:
        v = data[m]
        assert v["summary"]["supervision"] and v["summary"]["labels"], m
        assert v["example"]["note"] and v["example"]["legend"], m
        nodes = [n for lane in v["diagram"]["lanes"] for n in lane["nodes"]]
        assert any(n["kind"] == "label" for n in nodes), f"{m}: no supervision node"
        if v["diagram"].get("layout") == "grid":
            ids = {n["id"] for n in nodes}
            cells = [tuple(n["pos"]) for n in nodes]
            assert len(cells) == len(set(cells)), f"{m}: two nodes in one cell"
            assert not set(cells) & {tuple(nt["pos"]) for nt in v["diagram"].get("notes", [])}, m
            assert all(a in ids and b in ids for a, b in v["diagram"]["edges"]), m
    unet = [n["id"] for lane in data["unet"]["diagram"]["lanes"] for n in lane["nodes"]]
    assert {"enc1", "enc2", "bottleneck", "dec2", "dec1", "out"} <= set(unet)
    skips = {(lk["from"], lk["to"]) for lk in data["unet"]["diagram"]["links"]}
    assert {("enc1", "dec1"), ("enc2", "dec2")} <= skips
    yolo = {n["id"] for lane in data["yolo"]["diagram"]["lanes"] for n in lane["nodes"]}
    assert {"backbone", "head", "box", "cls", "conf", "nms", "out"} <= yolo
    cnn = [n["id"] for lane in data["cnn_classification"]["diagram"]["lanes"] for n in lane["nodes"]]
    assert cnn[-4:] == ["conv", "pool", "fc", "out"]


def test_comparison_content_en_es(se):
    en, es = _load("method_comparisons.yaml", "en")["supervised"], _load("method_comparisons.yaml", "es")["supervised"]
    assert en["methods"] == es["methods"] == SUPERVISED and en["short"] == es["short"]
    assert [len(r["values"]) for r in en["rows"]] == [len(r["values"]) for r in es["rows"]] == [3] * len(en["rows"])
    assert len(en["rows"]) >= 6 and all(set(r["values"]) == set(SUPERVISED) for r in en["rows"] + es["rows"])
    assert set(en["figure"]["captions"]) == set(es["figure"]["captions"])
    for k in en["figure"]["captions"]:
        assert set(PH.findall(en["figure"]["captions"][k])) == set(PH.findall(es["figure"]["captions"][k])), k
    imgs = se.comparison(lambda k, d: d)
    from app.widgets.method_visuals import fill
    for k, cap in en["figure"]["captions"].items():
        assert k in imgs and "{" not in fill(cap, imgs["facts"]), k
    assert es["title"] != en["title"]


# ------------------------------------------------------------------ Qt
@pytest.fixture
def library(qtbot):
    pytest.importorskip("pytestqt")
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("en")
    from app.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1280, 721)
    w.show()
    w.navigate("library")
    yield w, w.page("library")
    w.close()


def test_grid_diagram_draws_and_selects(library, qtbot):
    from PySide6.QtCore import QPoint, Qt
    w, lib = library
    w.navigate("library", method_id="unet", tab="how")
    d = lib.how.diagram
    assert d.is_grid() and d.minimumWidth() <= 700
    _, rects, _, h = d._layout(d.width())
    assert h > 250 and not d.grab().isNull()
    # the U: the bottleneck is the lowest box, input and output share the top row
    assert rects["bottleneck"].top() > rects["enc2"].top() > rects["enc1"].top() > rects["input"].top()
    assert abs(rects["input"].top() - rects["out"].top()) < 1 and rects["out"].left() > rects["bottleneck"].left()
    r = rects["bottleneck"]
    qtbot.mouseClick(d, Qt.LeftButton, pos=QPoint(int(r.center().x()), int(r.center().y())))
    assert d.selected == "bottleneck" and "compressed" in lib.how.details.toPlainText()
    w.navigate("library", method_id="yolo", tab="how")
    _, rects, _, _ = lib.how.diagram._layout(lib.how.diagram.width())
    assert rects["box"].top() < rects["cls"].top() < rects["conf"].top()       # the head fans out


def test_comparison_panel_only_for_supervised(library):
    w, lib = library
    for m in SUPERVISED:
        w.navigate("library", method_id=m)
        assert not lib.compare.isHidden() and lib.compare.current == m and lib.compare.columns == SUPERVISED
        assert "Classification, detection or segmentation?" in lib.compare.title.text()
        assert any(b.text().startswith("**Type of supervision:**") or "Type of supervision" in b.text()
                   for b in lib.summary.findChildren(type(lib.summary.cards["idea"])))
    w.navigate("library", method_id="patchcore")
    assert lib.compare.isHidden()


def test_supervised_examples_in_the_library(library):
    w, lib = library
    for m in SUPERVISED:
        w.navigate("library", method_id=m, tab="example")
        assert lib.example.badge.text().startswith("Toy illustration")
        assert lib.example.gallery.tiles and lib.example.interp.text() and "{" not in lib.example.interp.text()
        assert lib.example.legend.text() == _load("method_visuals.yaml", "en")[m]["example"]["legend"]
        mh = w.minimumSizeHint()
        assert mh.width() <= 1280 and mh.height() <= 721 and w.width() == 1280, (m, mh)


def test_spanish_supervised(qtbot):
    pytest.importorskip("pytestqt")
    from app import i18n
    from core import settings
    settings.load()
    i18n.init("es")
    try:
        from app.ui.main_window import MainWindow
        w = MainWindow()
        qtbot.addWidget(w)
        w.navigate("library", method_id="unet")
        lib = w.page("library")
        es = _load("method_visuals.yaml", "es")["unet"]
        assert lib.summary.cards["problem"].text() == es["summary"]["problem"]
        assert lib.compare.title.text() == "¿Clasificación, detección o segmentación?"
        assert lib.how.diagram.nodes()[0]["title"] == "Imágenes + máscaras de píxeles"
        from app.widgets.method_visuals import kind_label
        assert kind_label("label") == "datos etiquetados (supervisión)"
        w.navigate("library", method_id="yolo", tab="example")
        assert lib.example.badge.text().startswith("Ilustración simplificada")
        w.close()
    finally:
        i18n.init("en")


def test_venue_year_not_repeated():
    from app.pages.library import venue_year
    assert venue_year({"venue": "CVPR 2016", "year": 2016}) == "CVPR 2016"
    assert venue_year({"venue": "MICCAI", "year": 2015}) == "MICCAI 2015"
    assert venue_year({"venue": None, "year": None}) == ""


def test_comparison_pictures_are_never_cropped(library, qtbot):
    """Each picture label is at least as large as its pixmap (+ the highlight frame): nothing is clipped."""
    from PySide6.QtWidgets import QLabel
    w, lib = library
    w.navigate("library", method_id="cnn_classification")
    qtbot.wait(50)
    pics = [lb for lb in lib.compare.findChildren(QLabel) if lb.pixmap() is not None and not lb.pixmap().isNull()]
    assert len(pics) == 6
    for lb in pics:
        pm = lb.pixmap()
        assert lb.width() >= pm.width() and lb.height() >= pm.height(), (lb.size(), pm.size())
        caps = [c for c in lb.parent().findChildren(QLabel) if c is not lb and c.text()]
        assert caps and all(c.geometry().top() >= lb.geometry().bottom() for c in caps), "caption overlaps picture"
        assert all(c.height() >= c.heightForWidth(c.width()) for c in caps), "caption squeezed (text cut)"
        assert all(c.geometry().bottom() <= c.parent().height() for c in caps), "caption outside its cell"
