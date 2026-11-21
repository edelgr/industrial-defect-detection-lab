"""Audit every scientific link of the research catalog.

  python -m scripts.verify_research_links            # online check, writes link_audit.json + docs/SOURCE_AUDIT.md
  python -m scripts.verify_research_links --offline  # structural checks only

What it does
1. reads the catalog and checks required reference fields and URL syntax;
2. requests every URL (HEAD, then GET) and records status, redirects and final URL;
3. checks DOIs against Crossref and arXiv IDs against the arXiv API, comparing titles;
4. classifies results: ok · redirect · blocked (401/403/405/418/429/503 — many publishers block
   bots; a block is NOT evidence of an invalid URL) · not_found (404/410) · error;
5. writes research_catalog/link_audit.json (read by the GUI) and docs/SOURCE_AUDIT.md.
Manual verification can be recorded in research_catalog/manual_verification.yaml
({url: {status: manual, note: ..., date: ...}}); it overrides automated 'blocked' results.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import difflib
import json
import re
import xml.etree.ElementTree as ET

import yaml

from core import paths
from core.research.catalog import (URL_FIELDS_ALGORITHM, URL_FIELDS_DATASET, URL_FIELDS_PUBLICATION,
                                   is_valid_url, load_catalog)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IDD-Lab-link-audit/0.1 (educational; research link verification)"
BLOCKED = {401, 403, 405, 406, 418, 429, 503}


def collect(cat) -> dict[str, list[str]]:
    urls: dict[str, list[str]] = {}
    for kind, coll, fields in (("dataset", cat.datasets, URL_FIELDS_DATASET),
                               ("algorithm", cat.algorithms, URL_FIELDS_ALGORITHM),
                               ("publication", cat.publications, URL_FIELDS_PUBLICATION)):
        for rid, r in coll.items():
            for f in fields:
                if r.get(f):
                    urls.setdefault(r[f], []).append(f"{kind}:{rid}.{f}")
            if r.get("doi"):
                urls.setdefault(f"https://doi.org/{r['doi']}", []).append(f"{kind}:{rid}.doi")
            for i, c in enumerate(r.get("complementary_references") or []):
                urls.setdefault(c["url"], []).append(f"{kind}:{rid}.complementary_references[{i}]")
    return urls


def check_url(session, url: str) -> dict:
    out = {"checked": dt.date.today().isoformat()}
    try:
        r = session.head(url, allow_redirects=True, timeout=25)
        if r.status_code in (405, 403, 400) or r.status_code >= 500:
            r = session.get(url, allow_redirects=True, timeout=30, stream=True)
            r.close()
        code = r.status_code
        final = r.url if len(r.url) < 200 else r.url.split("?")[0] + "?…"   # drop signed query strings
        out.update(http_status=code, final_url=final)
        if code < 300:
            out["status"] = "redirect" if r.history and r.url.rstrip("/") != url.rstrip("/") else "ok"
        elif code in BLOCKED:
            out["status"] = "blocked"
        elif code in (404, 410):
            out["status"] = "not_found"
        else:
            out["status"] = "error"
    except Exception as e:  # network errors are reported, never silently "fixed"
        out.update(status="error", error=f"{type(e).__name__}: {e}"[:200])
    return out


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def check_crossref(session, doi: str, title: str) -> dict:
    try:
        r = session.get(f"https://api.crossref.org/works/{doi}", timeout=30)
        if r.status_code != 200:
            return {"doi_status": f"crossref HTTP {r.status_code}"}
        t = (r.json()["message"].get("title") or [""])[0]
        sim = difflib.SequenceMatcher(None, _norm(t), _norm(title)).ratio()
        return {"doi_status": "match" if sim > 0.8 else "title_mismatch", "crossref_title": t, "similarity": round(sim, 3)}
    except Exception as e:
        return {"doi_status": f"error {type(e).__name__}"}


def check_arxiv(session, aid: str, title: str) -> dict:
    try:
        r = session.get(f"http://export.arxiv.org/api/query?id_list={aid}", timeout=30)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = ET.fromstring(r.text).find("a:entry", ns)
        t = " ".join((e.find("a:title", ns).text or "").split()) if e is not None else ""
        sim = difflib.SequenceMatcher(None, _norm(t), _norm(title)).ratio()
        return {"arxiv_status": "match" if sim > 0.8 else "title_mismatch", "arxiv_title": t, "similarity": round(sim, 3)}
    except Exception as e:
        return {"arxiv_status": f"error {type(e).__name__}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    cat = load_catalog()
    report = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "problems": list(cat.problems),
              "links": {}, "publications": {}}
    urls = collect(cat)
    for u in urls:
        if not is_valid_url(u):
            report["problems"].append(f"invalid URL syntax: {u}")
    for rid, r in cat.datasets.items():
        for f in ("paper_url", "official_website", "download_url"):
            if not r.get(f):
                report["problems"].append(f"dataset {rid}: missing {f} (explicitly null)")
    for rid, r in cat.algorithms.items():
        if not r.get("official_repository_url"):
            report["problems"].append(f"algorithm {rid}: no official authors' repository (stated as such)")
    if not a.offline:
        import requests
        s = requests.Session()
        s.headers["User-Agent"] = UA
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for u, res in zip(urls, ex.map(lambda u: check_url(s, u), urls)):
                res["used_by"] = urls[u]
                report["links"][u] = res
        for pid, p in cat.publications.items():
            pr = {}
            if p.get("doi"):
                pr.update(check_crossref(s, p["doi"], p["title"]))
            if p.get("arxiv_id"):
                pr.update(check_arxiv(s, p["arxiv_id"], p["title"]))
            report["publications"][pid] = pr
    manual_f = paths.RESEARCH_CATALOG / "manual_verification.yaml"
    if manual_f.exists():
        manual = yaml.safe_load(manual_f.read_text(encoding="utf-8")) or {}
        for u, m in manual.items():
            report["links"].setdefault(u, {}).update({"status": "manual", **m})
    if not a.offline:
        (paths.LINK_AUDIT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(report)
    from collections import Counter
    print("link status:", dict(Counter(v.get("status") for v in report["links"].values())))
    bad = {k: v for k, v in report["publications"].items() if any("mismatch" in str(x) or "HTTP" in str(x) for x in v.values())}
    print("publication metadata issues:", bad or "none")


def write_md(rep: dict) -> None:
    L = ["# Source Audit", "", f"Generated {rep['generated']} by `scripts/verify_research_links.py`.", "",
         "A `blocked` status (HTTP 401/403/405/418/429/503) means the site refuses automated clients; it is "
         "**not** evidence that the URL is invalid. Such links are verified manually.", ""]
    from collections import Counter
    c = Counter(v.get("status") for v in rep["links"].values())
    L += ["## Summary", "", ", ".join(f"**{k}**: {v}" for k, v in sorted(c.items(), key=lambda x: str(x[0]))) or "offline run", ""]
    L += ["## Links", "", "| Status | HTTP | URL | Final URL | Used by |", "|---|---|---|---|---|"]
    for u, v in sorted(rep["links"].items(), key=lambda kv: (str(kv[1].get("status")), kv[0])):
        fu = v.get("final_url", "")
        L.append(f"| {v.get('status')} | {v.get('http_status', '')} | {u} | {fu if fu != u else ''} | {', '.join(v.get('used_by', []))} |")
    L += ["", "## Publication metadata (Crossref / arXiv title match)", "", "| Publication | DOI | arXiv |", "|---|---|---|"]
    for pid, v in sorted(rep["publications"].items()):
        L.append(f"| {pid} | {v.get('doi_status', '—')} {v.get('similarity', '')} | {v.get('arxiv_status', '—')} |")
    L += ["", "## Structural notes", ""] + [f"- {p}" for p in rep["problems"]]
    paths.DOCS.mkdir(exist_ok=True)
    (paths.DOCS / "SOURCE_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
