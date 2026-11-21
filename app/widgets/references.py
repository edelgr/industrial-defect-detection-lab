"""'Research Sources & Downloads' / 'Scientific References & Source Code' panel.

Every action is either a working link (opened in the system browser) or a clearly disabled
button that states why the resource is unavailable — never a fake link.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from app.i18n import tr
from app.widgets.common import Badge, copy_text, muted, open_url
from core.research.catalog import Catalog

_STATUS_KIND = {"ok": "ok", "redirect": "ok", "blocked": "warn", "manual": "warn", "error": "bad",
                "not_found": "bad"}


def _status_badge(cat: Catalog, url: str | None) -> QLabel | None:
    st = cat.link_status(url)
    if not st:
        return None
    s = st.get("status", "?")
    label = {"ok": tr("link.ok", "reachable"), "redirect": tr("link.redirect", "redirects"),
             "blocked": tr("link.blocked", "blocks bots – verify manually"),
             "error": tr("link.error", "unreachable at last check"), "not_found": "404"}.get(s, s)
    b = Badge(f"{label} · {st.get('checked', '')}", _STATUS_KIND.get(s, ""))
    b.setToolTip(f"HTTP {st.get('http_status')} — {st.get('final_url', url)}")
    return b


def _wrapped(text: str):
    lab = muted(text)
    lab.setMinimumHeight(lab.heightForWidth(360))
    return lab


class LinkAction(QWidget):
    def __init__(self, cat: Catalog, label: str, url: str | None, missing_reason: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.setMinimumHeight(30)
        btn = QPushButton(("↗  " + label) if url else f"⊘  {label} — {missing_reason}")
        btn.setObjectName("LinkButton")
        btn.setCursor(Qt.PointingHandCursor if url else Qt.ArrowCursor)
        if url:
            btn.setToolTip(url)
            btn.clicked.connect(lambda: open_url(url))
        else:
            btn.setEnabled(False)
            btn.setToolTip(missing_reason)
        lay.addWidget(btn)
        if url:
            b = _status_badge(cat, url)
            if b:
                lay.addWidget(b)
        lay.addStretch(1)


class ReferencesPanel(QFrame):
    """Persistent, visible references block for a dataset or algorithm record."""

    def __init__(self, cat: Catalog, record: dict, kind: str, parent=None):
        super().__init__(parent)
        self.setObjectName("RefPanel")
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)   # never compress below content
        self.cat, self.record = cat, record
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        title = tr("refs.dataset_title", "Research Sources & Downloads") if kind == "dataset" \
            else tr("refs.method_title", "Scientific References & Source Code")
        t = QLabel(title)
        t.setObjectName("RefTitle")
        lay.addWidget(t)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(0)
        r = record
        doi_url = cat.doi_url(r.get("doi"))
        na = tr("refs.not_available", "Not available")
        if kind == "dataset":
            actions = [
                (tr("refs.read_paper", "Read Original Paper"), r.get("paper_url") or doi_url, na),
                (tr("refs.open_pdf", "Open Paper PDF"), r.get("paper_pdf_url"), tr("refs.no_pdf", "No open-access PDF identified")),
                (tr("refs.website", "Open Official Dataset Website"), r.get("official_website"), tr("refs.no_website", "No official website")),
                (tr("refs.repo", "Open Research Repository"), r.get("official_repository_url") or r.get("implementation_repository_url"), tr("refs.no_repo", "No official repository identified")),
                (tr("refs.download", "Download / Request Dataset"), r.get("download_url"), tr("refs.no_download", "No public download route verified")),
                (tr("refs.license", "Open Licence"), r.get("license_url"), tr("refs.no_license", "Licence page not identified")),
            ]
            if doi_url and r.get("paper_url") != doi_url:
                actions.insert(1, (tr("refs.open_doi", "Open DOI") + f" ({r['doi']})", doi_url, na))
        else:
            impl_label = tr("refs.impl_used", "Implementation Used")
            it = r.get("implementation_type")
            impl_note = {"official": tr("impl.official", "official authors' code"),
                         "third_party_reproduction": tr("impl.third", "third-party reproduction"),
                         "anomalib": "Anomalib", "library": tr("impl.library", "library implementation"),
                         "idd_lab_educational": tr("impl.idd", "IDD Lab educational implementation"),
                         "none": tr("impl.none", "no runnable implementation")}.get(it, it)
            actions = [
                (tr("refs.read_paper_short", "Read Paper"), r.get("paper_url") or doi_url, na),
                (tr("refs.open_pdf", "Open Paper PDF"), r.get("paper_pdf_url"), tr("refs.no_pdf", "No open-access PDF identified")),
                (tr("refs.source_repo", "Source Repository (authors)"), r.get("official_repository_url"),
                 tr("refs.no_official_code", "No official authors' code verified")),
                (f"{impl_label} — {impl_note}", r.get("implementation_repository_url"), tr("refs.no_impl", "No runnable implementation in IDD Lab")),
                (tr("refs.weights", "Model Weights"), r.get("weights_url"), tr("refs.no_weights", "No pretrained weights used / not applicable")),
                (tr("refs.docs", "Technical Documentation"), r.get("docs_url"), tr("refs.no_docs", "No separate documentation")),
            ]
            if doi_url and r.get("paper_url") != doi_url:
                actions.insert(1, (tr("refs.open_doi", "Open DOI") + f" ({r['doi']})", doi_url, na))
        # available links first, then the explicitly unavailable ones (never fake links)
        actions.sort(key=lambda a: a[1] is None)
        for i, (label, url, why) in enumerate(actions):
            grid.addWidget(LinkAction(cat, label, url, why), i, 0)
        lay.addLayout(grid)

        row = QHBoxLayout()
        bib = cat.bibtex_for(record)
        b = QPushButton(tr("refs.copy_bibtex", "Copy BibTeX Citation"))
        b.setEnabled(bool(bib))
        b.clicked.connect(lambda: (copy_text(bib), b.setText(tr("refs.copied", "Copied ✓"))))
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        guide = r.get("resource_guide") or r.get("implementation_notes")
        if guide:
            g = muted("ⓘ " + guide)
            g.setTextInteractionFlags(Qt.TextSelectableByMouse)
            g.setMinimumHeight(g.heightForWidth(360))
            lay.addWidget(g)
        if kind == "algorithm" and r.get("source_license"):
            lay.addWidget(_wrapped(f"{tr('refs.code_license', 'Code licence')}: {r['source_license']}"
                                + (f" · {tr('refs.weights_license', 'Weights licence')}: {r['weights_license']}"
                                   if r.get("weights_license") else "")))
        if kind == "dataset" and r.get("license"):
            redist = r.get("redistribution_allowed")
            lay.addWidget(_wrapped(f"{tr('refs.data_license', 'Dataset licence')}: {r['license']} · "
                                f"{tr('refs.redistribution', 'Redistribution')}: {redist if redist is not None else 'unknown'}"))

        pubs = cat.publications_for(record)
        if pubs:
            lay.addWidget(QLabel(f"<b>{tr('refs.publications', 'Publications')}</b>"))
        for p in pubs:
            lay.addWidget(PublicationRow(cat, p))
        comp = r.get("complementary_references") or []
        if comp:
            lay.addWidget(QLabel(f"<b>{tr('refs.complementary', 'Complementary references')}</b>"))
        for c in comp:
            pub = cat.publications.get(c.get("publication_id") or "")
            if pub:
                lay.addWidget(PublicationRow(cat, pub))
            else:
                lay.addWidget(LinkAction(cat, c["title"] or c["label"], c["url"], ""))
            status = {"manual_required": tr("refs.unverified_meta", "bibliographic data NOT verified — confirm before citing"),
                      "verified": tr("refs.verified_meta", "bibliographic data verified (Crossref)")}.get(
                c["verification_status"], c["verification_status"])
            lay.addWidget(Badge(status, "warn" if c["verification_status"] != "verified" else "ok"))
            lay.addWidget(_wrapped("ⓘ " + c["note"]))
        ver = r.get("source_verification_status")
        if ver:
            lay.addWidget(_wrapped(f"{tr('refs.verification', 'Source verification')}: {ver} "
                                f"({r.get('source_verification_date') or 'n/a'})"))


class PublicationRow(QWidget):
    def __init__(self, cat: Catalog, p: dict, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 4)
        authors = ", ".join(p["authors"][:4]) + (" et al." if len(p["authors"]) > 4 else "")
        lab = QLabel(f"<i>{p['title']}</i><br><span>{authors} — {p['venue']} {p['year']}</span>")
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lab.setMinimumHeight(lab.heightForWidth(360))
        lay.addWidget(lab)
        self.setMinimumHeight(lab.minimumHeight() + 40)
        row = QHBoxLayout()
        row.setSpacing(2)
        doi = cat.doi_url(p.get("doi"))
        arxiv = f"https://arxiv.org/abs/{p['arxiv_id']}" if p.get("arxiv_id") else None
        for label, url in ((tr("refs.open_paper", "Open Paper"), p.get("paper_url")),
                           (tr("refs.pdf", "PDF"), p.get("pdf_url")), ("DOI", doi), ("arXiv", arxiv),
                           (tr("refs.code", "Source Code"), p.get("code_url"))):
            if url:
                btn = QPushButton("↗ " + label)
                btn.setObjectName("LinkButton")
                btn.setToolTip(url)
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(lambda _=False, u=url: open_url(u))
                row.addWidget(btn)
        cb = QPushButton(tr("refs.copy_bibtex_short", "Copy BibTeX"))
        cb.setObjectName("LinkButton")
        cb.clicked.connect(lambda: (copy_text(p["bibtex"].strip()), cb.setText(tr("refs.copied", "Copied ✓"))))
        row.addWidget(cb)
        if p.get("code_url") and p.get("code_is_official") is False:
            row.addWidget(Badge(tr("refs.unofficial", "code: third-party"), "warn"))
        row.addStretch(1)
        lay.addLayout(row)
