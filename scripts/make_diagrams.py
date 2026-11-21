"""Generate conceptual pipeline diagrams (SVG) for 'How This Algorithm Works'.

  python -m scripts.make_diagrams

One SVG per step (<method>_<k>.svg) showing the whole pipeline with step k highlighted. Every
diagram carries the label "CONCEPTUAL DIAGRAM — not real model activations".
"""

from __future__ import annotations

import html

import yaml

from core import paths

W, H = 960, 250
ACCENT, ACCENT2, BOX, BORDER, TEXT, MUTED = "#2bb3a3", "#f0a742", "#24303c", "#3a4a5a", "#e8eef4", "#9aa8b6"

ICONS = {  # tiny glyphs drawn inside boxes (purely illustrative)
    "grid": '<g stroke="{c}" stroke-width="1.2" fill="none">' + "".join(
        f'<rect x="{x}" y="{y}" width="9" height="9"/>' for x in (0, 10, 20) for y in (0, 10, 20)) + "</g>",
    "dots": '<g fill="{c}">' + "".join(f'<circle cx="{x}" cy="{y}" r="2.2"/>' for x, y in
                                       ((3, 5), (12, 9), (22, 4), (7, 18), (18, 21), (27, 14), (14, 27))) + "</g>",
    "bars": '<g fill="{c}">' + "".join(f'<rect x="{i * 6}" y="{28 - h}" width="4" height="{h}"/>' for i, h in
                                       enumerate((6, 14, 24, 16, 8))) + "</g>",
    "heat": '<g><rect width="30" height="30" fill="#2a1f3d"/><circle cx="19" cy="12" r="8" fill="#f0a742" opacity=".9"/>'
            '<circle cx="19" cy="12" r="4" fill="#ffe08a"/></g>',
}


def _icon_for(label: str) -> str:
    l = label.lower()
    if any(k in l for k in ("patch", "window", "grid", "parche", "ventana")):
        return "grid"
    if any(k in l for k in ("histogram", "score", "puntu", "top", "threshold", "umbral")):
        return "bars"
    if any(k in l for k in ("heat", "calor", "map", "colour", "color", "local")):
        return "heat"
    return "dots"


def svg(steps: list[str], current: int, title: str) -> str:
    n = len(steps)
    per_row = n if n <= 5 else (n + 1) // 2
    rows = 1 if n <= 5 else 2
    bw, bh, gap = 150, 62, 34
    total_w = per_row * bw + (per_row - 1) * gap
    x0 = (W - total_w) / 2
    ys = [70] if rows == 1 else [52, 150]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="Segoe UI, Arial, sans-serif">',
             f'<text x="16" y="24" fill="{TEXT}" font-size="15" font-weight="600">{html.escape(title)}</text>',
             f'<text x="{W - 16}" y="24" fill="{ACCENT2}" font-size="11" text-anchor="end">CONCEPTUAL DIAGRAM — '
             f'not real model activations</text>',
             '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>']
    pos = []
    for i, s in enumerate(steps):
        r = 0 if i < per_row else 1
        c = i if r == 0 else i - per_row
        x = x0 + c * (bw + gap)
        y = ys[r]
        pos.append((x, y))
        active = i == current
        done = i < current
        fill = ACCENT if active else BOX
        stroke = ACCENT if (active or done) else BORDER
        parts.append(f'<rect x="{x}" y="{y}" rx="9" width="{bw}" height="{bh}" fill="{fill}" stroke="{stroke}" '
                     f'stroke-width="{2.5 if active else 1.4}"/>')
        ic = ICONS[_icon_for(s)].format(c=("#0d2a27" if active else ACCENT))
        parts.append(f'<g transform="translate({x + 10},{y + 16})">{ic}</g>')
        parts.append(f'<text x="{x + 48}" y="{y + 27}" fill="{"#06231f" if active else MUTED}" font-size="11">{i + 1}</text>')
        parts.append(f'<text x="{x + 48}" y="{y + 44}" fill="{"#06231f" if active else TEXT}" font-size="13" '
                     f'font-weight="{700 if active else 500}">{html.escape(s)}</text>')
    for i in range(n - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if abs(y1 - y2) < 1:
            parts.append(f'<line x1="{x1 + bw}" y1="{y1 + bh / 2}" x2="{x2 - 3}" y2="{y2 + bh / 2}" stroke="{MUTED}" '
                         f'stroke-width="1.6" marker-end="url(#a)"/>')
        else:  # wrap to next row
            parts.append(f'<path d="M{x1 + bw / 2},{y1 + bh} C{x1 + bw / 2},{y1 + bh + 22} {x2 + bw / 2},{y2 - 22} '
                         f'{x2 + bw / 2},{y2 - 3}" fill="none" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#a)"/>')
    parts.append("</svg>")
    return "\n".join(parts)


TITLES = {"patchcore": "PatchCore", "padim": "PaDiM", "efficientad": "EfficientAD", "winclip": "WinCLIP",
          "anomalydino": "AnomalyDINO", "pc_deviation": "3D registration + deviation (educational baseline)",
          "otsu": "Otsu thresholding"}


def main():
    data = yaml.safe_load((paths.EDUCATION / "en" / "steps.yaml").read_text(encoding="utf-8"))
    out = paths.EDUCATION / "diagrams"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for mid, d in data.items():
        labels = [s["short"] for s in d["steps"]]
        for k in range(len(labels)):
            (out / f"{mid}_{k + 1}.svg").write_text(svg(labels, k, TITLES.get(mid, mid)), encoding="utf-8")
            n += 1
    print(f"wrote {n} diagrams to {out}")


if __name__ == "__main__":
    main()
