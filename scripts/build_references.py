"""Generate CITATIONS.bib and docs/RESEARCH_REFERENCES.md from the research catalog.

  python -m scripts.build_references
"""

from __future__ import annotations

from core import paths
from core.research.catalog import Catalog, load_catalog


def render_bib(cat: Catalog) -> str:
    out = ["% IDD Lab — CITATIONS.bib (generated from research_catalog/publications; do not edit by hand)\n"]
    for pid in sorted(cat.publications):
        out.append(cat.publications[pid]["bibtex"].strip() + "\n")
    return "\n".join(out)


def _link(label, url):
    return f"[{label}]({url})" if url else ""


def render_md(cat: Catalog) -> str:
    L = ["# Research References", "",
         "Generated from `research_catalog/` by `python -m scripts.build_references`. "
         "Verification status and dates come from the catalog; the latest automated link audit is in "
         "`docs/SOURCE_AUDIT.md`.", "",
         "Legend — **official**: code released by the paper's authors · **third-party**: re-implementation "
         "by others · **Anomalib**: Intel/open-edge-platform library integration · **IDD Lab**: educational "
         "implementation written for this project.", ""]
    L += ["## Datasets", "", "| ID | Dataset | Paper | Website | Code | Data access | Licence | Verified |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(cat.datasets.values(), key=lambda r: r["id"]):
        L.append(f"| {r['id']} | {r['name']} ({r['acronym']}) | {_link('paper', r['paper_url'])} "
                 f"{_link('DOI', cat.doi_url(r['doi']))} | {_link('site', r['official_website'])} | "
                 f"{_link('repo', r['official_repository_url']) or '—'} | {_link('access', r['download_url'])} | "
                 f"{r['license'] or 'unknown'} | {r['source_verification_status']} {r['source_verification_date'] or ''} |")
    L += ["", "## Methods", "", "| Group | Method | Paper | Authors' code | Implementation used | Type | Code licence | Status |",
          "|---|---|---|---|---|---|---|---|"]
    for g, recs in cat.algorithms_by_group().items():
        for r in recs:
            L.append(f"| {g} | {r['name']} | {_link('paper', r['paper_url'] or cat.doi_url(r['doi']))} | "
                     f"{_link('repo', r['official_repository_url']) or 'none verified'} | "
                     f"{_link('impl', r['implementation_repository_url']) or '—'} {r.get('implementation_version') or ''} | "
                     f"{r['implementation_type']} | {r['source_license'] or '—'} | {r['implementation_status']} |")
    L += ["", "## Publications", ""]
    for p in sorted(cat.publications.values(), key=lambda p: (p["year"], p["id"])):
        a = ", ".join(p["authors"][:6]) + (" et al." if len(p["authors"]) > 6 else "")
        links = " · ".join(x for x in (_link("paper", p["paper_url"]), _link("PDF", p["pdf_url"]),
                                         _link("DOI", cat.doi_url(p["doi"])),
                                         _link("arXiv", f"https://arxiv.org/abs/{p['arxiv_id']}" if p["arxiv_id"] else None),
                                         _link("code" + ("" if p.get("code_is_official") is not False else " (third-party)"), p["code_url"])) if x)
        L.append(f"- **{p['title']}** — {a}. *{p['venue']}*, {p['year']}. {links}  \n"
                 f"  `{p['id']}` · verification: {p['verification']['status']} {p['verification'].get('date') or ''}")
    return "\n".join(L) + "\n"


def main():
    cat = load_catalog(strict=True)
    (paths.PROJECT_ROOT / "CITATIONS.bib").write_text(render_bib(cat), encoding="utf-8")
    paths.DOCS.mkdir(exist_ok=True)
    md = render_md(cat)
    (paths.DOCS / "RESEARCH_REFERENCES.md").write_text(md, encoding="utf-8")
    (paths.PROJECT_ROOT / "RESEARCH_REFERENCES.md").write_text(md, encoding="utf-8")
    print(f"wrote CITATIONS.bib ({len(cat.publications)} entries), RESEARCH_REFERENCES.md and docs/RESEARCH_REFERENCES.md")


if __name__ == "__main__":
    main()
