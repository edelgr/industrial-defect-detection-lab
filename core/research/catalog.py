"""Research catalog: loads and validates dataset, algorithm and publication records.

Records live as YAML files under ``research_catalog/`` so that they can be reviewed and edited
independently of the GUI.  The catalog is fully offline: nothing here touches the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import jsonschema
import yaml

from core import paths
from research_catalog.schemas import ALGORITHM_SCHEMA, DATASET_SCHEMA, PUBLICATION_SCHEMA

URL_FIELDS_DATASET = ["paper_url", "paper_pdf_url", "official_website", "official_repository_url",
                      "implementation_repository_url", "download_url", "license_url"]
URL_FIELDS_ALGORITHM = ["paper_url", "paper_pdf_url", "official_repository_url",
                        "implementation_repository_url", "weights_url", "docs_url"]
URL_FIELDS_PUBLICATION = ["paper_url", "pdf_url", "code_url"]


class CatalogError(ValueError):
    pass


def is_valid_url(url: str | None) -> bool:
    if not url:
        return False
    p = urlparse(url)
    return p.scheme in ("http", "https") and bool(p.netloc) and " " not in url


@dataclass
class Catalog:
    datasets: dict[str, dict] = field(default_factory=dict)
    algorithms: dict[str, dict] = field(default_factory=dict)
    publications: dict[str, dict] = field(default_factory=dict)
    link_audit: dict[str, dict] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- lookups
    def publication(self, pub_id: str) -> dict | None:
        return self.publications.get(pub_id)

    def publications_for(self, record: dict) -> list[dict]:
        return [self.publications[p] for p in record.get("publication_ids", []) if p in self.publications]

    def link_status(self, url: str | None) -> dict | None:
        return self.link_audit.get(url) if url else None

    def bibtex_for(self, record: dict) -> str:
        return "\n\n".join(p["bibtex"].strip() for p in self.publications_for(record))

    def doi_url(self, doi: str | None) -> str | None:
        return f"https://doi.org/{doi}" if doi else None

    def algorithms_by_group(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for a in self.algorithms.values():
            out.setdefault(a["group"], []).append(a)
        for v in out.values():
            v.sort(key=lambda r: (r.get("year") or 0, r["name"]))
        return dict(sorted(out.items()))


def _load_yaml_dir(directory: Path) -> list[tuple[Path, dict]]:
    out = []
    for f in sorted(directory.glob("*.yaml")):
        with f.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, list):
            out.extend((f, d) for d in data)
        elif isinstance(data, dict):
            out.append((f, data))
    return out


def _validate(records: list[tuple[Path, dict]], schema: dict, url_fields: list[str],
              kind: str, problems: list[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    validator = jsonschema.Draft202012Validator(schema)
    for f, rec in records:
        errs = sorted(validator.iter_errors(rec), key=lambda e: list(e.path))
        rid = rec.get("id", f"<{f.name}>")
        for e in errs:
            loc = "/".join(str(x) for x in e.path) or "(root)"
            problems.append(f"{kind} {rid} [{f.name}] {loc}: {e.message}")
        for uf in url_fields:
            v = rec.get(uf)
            if v is not None and not is_valid_url(v):
                problems.append(f"{kind} {rid}: invalid URL in '{uf}': {v!r}")
        if rid in result:
            problems.append(f"{kind} {rid}: duplicate id")
        result[rid] = rec
    return result


def load_catalog(root: Path = paths.RESEARCH_CATALOG, strict: bool = False) -> Catalog:
    problems: list[str] = []
    pubs = _validate(_load_yaml_dir(root / "publications"), PUBLICATION_SCHEMA,
                     URL_FIELDS_PUBLICATION, "publication", problems)
    dsets = _validate(_load_yaml_dir(root / "datasets"), DATASET_SCHEMA,
                      URL_FIELDS_DATASET, "dataset", problems)
    algos = _validate(_load_yaml_dir(root / "algorithms"), ALGORITHM_SCHEMA,
                      URL_FIELDS_ALGORITHM, "algorithm", problems)

    # Referential integrity: every referenced publication must exist.
    for kind, recs in (("dataset", dsets), ("algorithm", algos)):
        for rid, rec in recs.items():
            for c in rec.get("complementary_references") or []:
                if c.get("publication_id") and c["publication_id"] not in pubs:
                    problems.append(f"{kind} {rid}: unknown complementary publication id '{c['publication_id']}'")
            for pid in rec.get("publication_ids", []):
                if pid not in pubs:
                    problems.append(f"{kind} {rid}: unknown publication id '{pid}'")
    for pid, p in pubs.items():
        for d in p.get("datasets", []):
            if d not in dsets:
                problems.append(f"publication {pid}: unknown dataset '{d}'")
        for a in p.get("algorithms", []):
            if a not in algos:
                problems.append(f"publication {pid}: unknown algorithm '{a}'")

    audit: dict[str, dict] = {}
    audit_file = root / "link_audit.json"
    if audit_file.exists():
        try:
            audit = json.loads(audit_file.read_text(encoding="utf-8")).get("links", {})
        except json.JSONDecodeError:
            problems.append("link_audit.json is not valid JSON")

    cat = Catalog(datasets=dsets, algorithms=algos, publications=pubs, link_audit=audit,
                  problems=problems)
    if strict and problems:
        raise CatalogError("\n".join(problems))
    return cat


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    return load_catalog()
