"""Compare r/n/ME metrics embedded in v1 preprint vs v2 final figure PDFs."""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "perso" / "v1_1_preprint"
V2 = ROOT / "perso" / "v2_modifs" / "final"
ARTICLE = ROOT / "article"

FIGURE_PAIRS = [
    ("fig3.pdf", "fig3.pdf", "fig:map_clim"),
    ("fig5.pdf", "fig5.pdf", "fig:map_daily_trend"),
    ("fig6.pdf", "fig6.pdf", "fig:map_hourly_trend"),
    ("fig7.pdf", "figc1.pdf", "fig:map_monthly_trend (appendix maps)"),
    ("fig8.pdf", "fig7.pdf", "fig:correlation_trend (panel c n only in PDF text)"),
    ("annexe1.pdf", "figa1.pdf", "ann:map_trend_daily"),
    ("annexe2.pdf", "figa2.pdf", "ann:map_trend_hourly"),
]

MATH_LETTER = {
    "\u1d434": "A", "\u1d435": "B", "\u1d436": "C", "\u1d437": "D", "\u1d438": "E",
    "\u1d439": "F", "\u1d43a": "G", "\u1d43b": "H", "\u1d43c": "I", "\u1d43d": "J",
    "\u1d43e": "K", "\u1d43f": "L", "\u1d440": "M", "\u1d441": "N", "\u1d442": "O",
    "\u1d443": "P", "\u1d444": "Q", "\u1d445": "R", "\u1d446": "S", "\u1d447": "T",
    "\u1d448": "U", "\u1d449": "V", "\u1d44a": "W", "\u1d44b": "X", "\u1d44c": "Y",
    "\u1d44d": "Z",
    "\u1d45a": "m", "\u1d45b": "n", "\u1d45f": "r", "\u1d45e": "q", "\u1d45d": "p",
}


@dataclass
class MetricBlock:
    r: float | None = None
    me: float | None = None
    n: int | None = None


def normalize_pdf_text(text: str) -> str:
    out = []
    for ch in text:
        if ch in MATH_LETTER:
            out.append(MATH_LETTER[ch])
        elif ch in ("\u2212", "\u2013", "\u2014"):
            out.append("-")
        elif ch == "\u00a0":
            out.append(" ")
        else:
            out.append(ch)
    return unicodedata.normalize("NFKC", "".join(out))


def pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    parts = [page.get_text("text") for page in doc]
    doc.close()
    return normalize_pdf_text("\n".join(parts))


def extract_blocks(text: str) -> list[MetricBlock]:
    inline = re.findall(
        r"\(\s*r\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*n\s*=\s*(\d+)\s*,\s*ME\s*=\s*([+-]?\d+(?:\.\d+)?)",
        text,
        flags=re.I,
    )
    if inline:
        return [MetricBlock(float(r), float(me), int(n)) for r, n, me in inline]

    rs = [(float(r), int(n)) for r, n in re.findall(r"r\s*=\s*([+-]?\d+(?:\.\d+)?)\s*\(\s*n\s*=\s*(\d+)\s*\)", text, re.I)]
    mes = [float(x) for x in re.findall(r"ME\s*=\s*([+-]?\d+(?:\.\d+)?)", text, re.I)]
    if rs:
        blocks = []
        for i, (r, n) in enumerate(rs):
            me = mes[i] if i < len(mes) else None
            blocks.append(MetricBlock(r, me, n))
        return blocks
    return []


def extract_n_series(text: str) -> list[int]:
    nums = [int(x) for x in re.findall(r"^\s*(\d{2,4})\s*$", text, re.M)]
    return nums


def blocks_equal(a: MetricBlock, b: MetricBlock, tol: float = 0.02) -> bool:
    if a.r is None or b.r is None or abs(a.r - b.r) > tol:
        return False
    if a.n is not None and b.n is not None and a.n != b.n:
        return False
    if a.me is not None and b.me is not None and abs(a.me - b.me) > tol:
        return False
    return True


def compare_blocks(label: str, b1: list[MetricBlock], b2: list[MetricBlock]) -> int:
    print(f"\n{label}")
    print(f"  blocks: v1={len(b1)}  v2={len(b2)}")
    if len(b1) != len(b2):
        print("  WARN block count differs")
    warn = 0
    for i, (left, right) in enumerate(zip(b1, b2), start=1):
        if blocks_equal(left, right):
            continue
        warn += 1
        print(f"  block {i}: v1 r={left.r} ME={left.me} n={left.n} | v2 r={right.r} ME={right.me} n={right.n}")
    if warn == 0:
        print("  OK all blocks match")
    return warn


def compare_pair(v1_name: str, v2_name: str, label: str) -> int:
    p1, p2 = V1 / v1_name, V2 / v2_name
    if not p1.exists() or not p2.exists():
        print(f"\n{label}: SKIP missing file")
        return 1
    if "fig8" in v1_name and "fig7" in v2_name:
        t1, t2 = pdf_text(p1), pdf_text(p2)
        n1, n2 = extract_n_series(t1), extract_n_series(t2)
        n2_corr = n2[-len(n1):] if len(n2) >= len(n1) else n2
        print(f"\n{label}")
        print(f"  n labels: v1={len(n1)}  v2(panel c subset)={len(n2_corr)}")
        if n1 == n2_corr:
            print("  OK n-series identical (daily+hourly, panel c)")
            return 0
        print("  WARN n-series differ")
        print(f"    v1: {n1}")
        print(f"    v2: {n2_corr}")
        return 1
    b1 = extract_blocks(pdf_text(p1))
    b2 = extract_blocks(pdf_text(p2))
    return compare_blocks(label, b1, b2)


def main() -> None:
    print("Comparison v1 preprint figures vs v2 final figures")
    print(f"  v1 dir: {V1}")
    print(f"  v2 dir: {V2}")
    total_warn = 0
    for v1, v2, label in FIGURE_PAIRS:
        total_warn += compare_pair(v1, v2, label)
    print(f"\n{'='*72}")
    print(f"TOTAL warnings: {total_warn}")
    raise SystemExit(1 if total_warn else 0)


if __name__ == "__main__":
    main()
