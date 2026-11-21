"""Dataset adapters, indexing, split integrity, modality pairing, safe import/extraction."""

import io
import tarfile
import zipfile

import numpy as np
import pytest

from core.datasets import manager
from core.datasets.base import BAD, GOOD, UNKNOWN
from core.datasets.io import read_mask, read_point_cloud, read_rgb8, write_image
from core.datasets.mulsen import MulSenADAdapter
from core.datasets.mvtec import MVTecAD2Adapter, MVTecLOCOAdapter
from core.datasets.real3d import Real3DADAdapter, read_point_labels


# ------------------------------------------------------------------ synthetic / MVTec AD layout
def test_synthetic_index_and_split_integrity(synthetic_index):
    idx = synthetic_index
    assert idx.summary()["train"] == {"good": 12}
    assert all(s.label == GOOD for s in idx.split("train"))
    bads = [s for s in idx.samples if s.label == BAD]
    assert len(bads) == 18 and all(s.masks.get("rgb") for s in bads)     # 6 defect kinds x 3
    assert not idx.warnings
    ids = [s.sample_id for s in idx.samples]
    assert len(ids) == len(set(ids)), "sample ids must be unique"


def test_image_and_mask_loading(synthetic_index):
    s = next(s for s in synthetic_index.samples if s.label == BAD)
    img = read_rgb8(s.primary_path)
    m = read_mask(s.masks["rgb"], img.shape[:2])
    assert img.dtype == np.uint8 and img.shape == (128, 128, 3)
    assert m.dtype == bool and m.any()


def _png(path, val=128, shape=(8, 8)):
    write_image(path, np.full((*shape, 3), val, np.uint8))


def test_mvtec_ad2_private_splits_are_unlabelled(tmp_path):
    c = tmp_path / "sheet_metal"
    _png(c / "train/good/000_regular.png")
    _png(c / "validation/good/000_regular.png")
    _png(c / "test_public/good/000_regular.png")
    _png(c / "test_public/bad/001_regular.png")
    write_image(c / "test_public/ground_truth/bad/001_regular_mask.png", np.full((8, 8), 255, np.uint8))
    _png(c / "test_private/000.png")
    _png(c / "test_private_mixed/000.png")
    idx = MVTecAD2Adapter().index(tmp_path, "sheet_metal")
    by = {s.split: s for s in idx.samples if s.split.startswith("test_private")}
    assert by["test_private"].label == UNKNOWN and by["test_private_mixed"].label == UNKNOWN
    bad = next(s for s in idx.samples if s.label == BAD)
    assert bad.masks["rgb"][0].name == "001_regular_mask.png"
    assert all(not s.split.startswith("test_private") for s in idx.labelled_test())


def test_loco_multi_masks(tmp_path):
    c = tmp_path / "pushpins"
    _png(c / "train/good/000.png")
    _png(c / "test/logical_anomalies/000.png")
    write_image(c / "ground_truth/logical_anomalies/000/000.png", np.full((8, 8), 255, np.uint8))
    write_image(c / "ground_truth/logical_anomalies/000/001.png", np.zeros((8, 8), np.uint8))
    idx = MVTecLOCOAdapter().index(tmp_path, "pushpins")
    bad = next(s for s in idx.samples if s.label == BAD)
    assert len(bad.masks["rgb"]) == 2


# ------------------------------------------------------------------ MulSen-AD layout
def _mulsen_fixture(root):
    c = root / "capsule"
    for mod, ext in (("RGB", ".png"), ("Infrared", ".png")):
        for i in range(2):
            _png(c / f"{mod}/train/{i}{ext}")
        _png(c / f"{mod}/test/good/0{ext}")
        _png(c / f"{mod}/test/hole/0{ext}")
        _png(c / f"{mod}/test/hole/1{ext}")
    write_image(c / "Infrared/GT/hole/1.png", np.full((8, 8), 255, np.uint8))
    for p in ("train/0", "train/1", "test/good/0", "test/hole/0", "test/hole/1"):
        f = c / f"Pointcloud/{p}.stl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    (c / "Pointcloud/GT/hole").mkdir(parents=True)
    (c / "Pointcloud/GT/hole/0.txt").write_text("# x, y, z\n0,0,0\n", encoding="utf-8")
    (c / "RGB/GT/hole").mkdir(parents=True, exist_ok=True)
    (c / "RGB/GT/hole/data.csv").write_text("object,RGB,infrared,pointcloud\n0,0,0,1\n1,0,1,0\n", encoding="utf-8")
    return root


def test_mulsen_modality_pairing(tmp_path):
    root = _mulsen_fixture(tmp_path)
    idx = MulSenADAdapter().index(root, "capsule")
    assert idx.summary() == {"train": {"good": 2}, "test": {"good": 1, "bad": 2}}
    for s in idx.samples:
        stems = {p.stem for p in s.modalities.values()}
        assert len(stems) == 1, f"modalities of {s.sample_id} not paired: {stems}"
        assert set(s.modalities) == {"rgb", "infrared", "pointcloud"}
    s1 = next(s for s in idx.samples if s.sample_id == "test/hole/1")
    assert s1.meta["modality_labels"] == {"rgb": 0, "infrared": 1, "pointcloud": 0}
    assert "infrared" in s1.masks and "rgb" not in s1.masks
    s0 = next(s for s in idx.samples if s.sample_id == "test/hole/0")
    assert "pointcloud" in s0.masks


def test_real_mulsen_if_installed():
    from core.datasets import registry
    cats = registry.installed_categories("D06")
    if not cats:
        pytest.skip("MulSen-AD not installed")
    idx = registry.load_index("D06", cats[0])
    assert not idx.warnings
    for s in idx.samples[:: max(1, len(idx.samples) // 15)]:
        assert {p.stem for p in s.modalities.values()} == {s.sample_id.split("/")[-1]}
        img = read_rgb8(s.modalities["rgb"])           # BMP-encoded '.png' must decode
        assert img.ndim == 3


# ------------------------------------------------------------------ 3D
def test_point_cloud_loading_and_labels(synthetic_root):
    from core.datasets.synthetic import SyntheticAdapter
    idx = SyntheticAdapter().index(synthetic_root, "surface_patch_3d")
    s = next(s for s in idx.samples if s.label == BAD)
    pts, _ = read_point_cloud(s.modalities["pointcloud"])
    assert pts.shape[1] == 3 and len(pts) > 1000
    lab = read_point_labels(s.masks["pointcloud"], pts)
    assert lab.shape == (len(pts),) and 0 < lab.sum() < len(lab)


def test_real3d_layout(tmp_path):
    c = tmp_path / "duck"
    for d in ("train", "test", "gt"):
        (c / d).mkdir(parents=True)
    for n in ("1_template.pcd", ):
        (c / "train" / n).write_text("")
    for n in ("1_bulge.pcd", "2_good.pcd", "3_temp.pcd"):
        (c / "test" / n).write_text("")
    (c / "gt/1_bulge.txt").write_text("0 0 0 1\n")
    idx = Real3DADAdapter().index(tmp_path, "duck")
    names = {s.sample_id for s in idx.samples}
    assert "test/3_temp" not in names
    bulge = next(s for s in idx.samples if s.sample_id == "test/1_bulge")
    assert bulge.label == BAD and bulge.defect_type == "bulge" and bulge.masks


# ------------------------------------------------------------------ safe import / extraction
def test_zip_selective_extraction(tmp_path):
    arch = tmp_path / "a.zip"
    with zipfile.ZipFile(arch, "w") as z:
        z.writestr("mvtec_ad_2/sheet_metal/train/good/000.png", b"x")
        z.writestr("mvtec_ad_2/vial/train/good/000.png", b"y")
    dest = tmp_path / "out"
    rec = manager.import_archive(arch, dest, ["sheet_metal"], ["sheet_metal", "vial"], compute_checksum=True)
    assert (dest / "sheet_metal/train/good/000.png").exists()
    assert not (dest / "vial").exists()
    assert rec["categories"] == ["sheet_metal"] and len(rec["sha256"]) == 64
    assert (dest / "idd_install_log.json").exists()


def test_zip_path_traversal_rejected(tmp_path):
    # Built in memory: a traversal archive written to disk is quarantined by endpoint protection.
    arch = io.BytesIO()
    with zipfile.ZipFile(arch, "w") as z:
        z.writestr("../../outside.txt", b"x")
    arch.seek(0)
    with pytest.raises(manager.UnsafeArchiveError):
        manager.extract(arch, tmp_path / "out")
    assert not (tmp_path / "outside.txt").exists() and not (tmp_path.parent / "outside.txt").exists()


def test_tar_links_are_skipped(tmp_path):
    arch = io.BytesIO()                       # in memory, see test_zip_path_traversal_rejected
    with tarfile.open(fileobj=arch, mode="w") as t:
        data = b"hello"
        ti = tarfile.TarInfo("cat/train/good/0.png")
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
        ln = tarfile.TarInfo("cat/link")
        ln.type = tarfile.SYMTYPE
        ln.linkname = "../outside"
        t.addfile(ln)
    arch.seek(0)
    files = manager.extract(arch, tmp_path / "out")
    assert [f.name for f in files] == ["0.png"]


def test_disk_space_guard(tmp_path):
    ok, _ = manager.check_space(tmp_path, 10 ** 18, 15)
    assert not ok
    assert manager.requires_approval(2 * manager.GB) and not manager.requires_approval(10 * 2 ** 20)


def test_remote_zip_member_extraction(tmp_path):
    """extract_members on an in-memory 'remote' file (CRC verified, deflate + stored)."""
    from core.datasets import remote_zip as rz
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("MulSen_AD/capsule/RGB/train/0.png", b"a" * 5000)
        z.writestr(zipfile.ZipInfo("MulSen_AD/nut/RGB/train/0.png"), b"b" * 10)
    raw = buf.getvalue()

    class Fake:
        transferred = 0

        def fetch(self, s, e):
            self.transferred += e - s
            return raw[s:e]

    zf = zipfile.ZipFile(io.BytesIO(raw))
    infos = rz.select(zf, ["MulSen_AD/capsule"])
    assert rz.plan(infos)["files"] == 1
    out = rz.extract_members(Fake(), infos, tmp_path, strip_prefix="MulSen_AD")
    assert out[0].read_bytes() == b"a" * 5000
    assert out[0].relative_to(tmp_path).as_posix() == "capsule/RGB/train/0.png"
