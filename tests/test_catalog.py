"""Research catalog: schema validation, scientific-reference requirements, BibTeX sanity."""

import re

import pytest

from core.research.catalog import is_valid_url, load_catalog
from research_catalog.schemas import ALGORITHM_FIELDS, DATASET_FIELDS


@pytest.fixture(scope="module")
def cat():
    return load_catalog()


def test_catalog_has_no_validation_problems(cat):
    assert cat.problems == []


def test_initial_dataset_catalog_complete(cat):
    assert set(cat.datasets) >= {f"D0{i}" for i in range(1, 9)}


def test_dataset_records_have_all_mandatory_fields(cat):
    for rid, rec in cat.datasets.items():
        missing = [f for f in DATASET_FIELDS if f not in rec]
        assert not missing, f"{rid} missing {missing}"


def test_algorithm_records_have_all_mandatory_fields(cat):
    for rid, rec in cat.algorithms.items():
        missing = [f for f in ALGORITHM_FIELDS if f not in rec]
        assert not missing, f"{rid} missing {missing}"


def test_every_dataset_has_paper_and_access_route(cat):
    for rid, r in cat.datasets.items():
        assert r["paper_url"] or r["doi"], rid
        assert r["official_website"] or r["download_url"], rid
        assert r["publication_ids"], rid
        assert r["resource_guide"].strip(), rid


def test_every_research_method_has_reference(cat):
    for rid, r in cat.algorithms.items():
        assert r["publication_ids"], f"{rid} has no publication"
        assert r["paper_url"] or r["doi"] or r.get("docs_url"), rid


def test_unofficial_code_is_never_labelled_official(cat):
    # WinCLIP, PaDiM, EfficientAD: no official authors' code was found during verification.
    for rid in ("winclip", "padim", "efficientad"):
        assert cat.algorithms[rid]["official_repository_url"] is None
        assert cat.algorithms[rid]["implementation_type"] != "official"


def test_runnable_statuses_match_registry(cat):
    from core.algorithms.registry import RUNTIMES
    runnable = {"runnable_classical", "runnable_pretrained", "requires_fitting", "validated_local_dataset"}
    for rid, r in cat.algorithms.items():
        if r["implementation_status"] in runnable:
            assert rid in RUNTIMES, f"{rid} marked runnable but has no runtime"


def test_all_urls_are_syntactically_valid(cat):
    for coll in (cat.datasets, cat.algorithms, cat.publications):
        for rid, r in coll.items():
            for k, v in r.items():
                if k.endswith("_url") and v is not None:
                    assert is_valid_url(v), f"{rid}.{k}={v}"


def test_bibtex_entries_are_balanced(cat):
    keys = set()
    for pid, p in cat.publications.items():
        b = p["bibtex"].strip()
        assert re.match(r"@\w+\{[^,\s]+,", b), pid
        assert b.count("{") == b.count("}"), f"unbalanced braces in {pid}"
        key = re.match(r"@\w+\{([^,]+),", b).group(1)
        assert key not in keys, f"duplicate BibTeX key {key}"
        keys.add(key)


def test_doi_format(cat):
    for coll in (cat.datasets, cat.algorithms, cat.publications):
        for rid, r in coll.items():
            if r.get("doi"):
                assert re.match(r"^10\.\d{4,9}/\S+$", r["doi"]), f"{rid}: {r['doi']}"


def test_bilingual_descriptions(cat):
    for r in cat.datasets.values():
        assert r["description_en"] != r["description_es"]
    for r in cat.algorithms.values():
        assert r["intuitive_explanation_en"] != r["intuitive_explanation_es"]


def test_citations_bib_in_sync(cat):
    from scripts.build_references import render_bib
    from core import paths
    f = paths.PROJECT_ROOT / "CITATIONS.bib"
    assert f.exists(), "run python -m scripts.build_references"
    assert f.read_text(encoding="utf-8").strip() == render_bib(cat).strip()
