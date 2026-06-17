"""Check marker sizes in z_T_p signif maps."""
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def extract_path_d(text: str) -> list[str]:
    out = []
    for m in re.finditer(r'<path d="', text):
        start = m.end()
        end = text.find('"', start)
        while end != -1 and text[end - 1] == "\\":
            end = text.find('"', end + 1)
        if end == -1:
            continue
        out.append(text[start:end])
    return out


def path_bbox(d: str) -> tuple[float, float, float, float] | None:
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", d)]
    if len(nums) < 2:
        return None
    xs, ys = nums[0::2], nums[1::2]
    return min(xs), max(xs), min(ys), max(ys)


def analyze(text: str, label: str) -> None:
    paths = extract_path_d(text)
    bboxes = [path_bbox(d) for d in paths]
    bboxes = [b for b in bboxes if b]
    widths = [round(b[1] - b[0], 2) for b in bboxes]
    heights = [round(b[3] - b[2], 2) for b in bboxes]
    c = Counter(widths)
    print(f"{label}: {len(paths)} paths, {len(c)} unique widths")
    print(f"  width range: {min(widths) if widths else 'n/a'} - {max(widths) if widths else 'n/a'}")
    print(f"  top widths: {c.most_common(8)}")


def circle_widths(text: str) -> Counter:
    circles = [d for d in extract_path_d(text) if "C" in d and d.strip().startswith("M")]
    widths = []
    for d in circles:
        b = path_bbox(d)
        if b:
            widths.append(round(b[1] - b[0], 2))
    return Counter(widths)


def main() -> None:
    base = ROOT / "outputs/maps/gev_z_T_p/quotidien/compare_5/sat_99.0"
    asm_path = Path(__file__).parent / "figures/trend_pluie_ond.svg"

    print("=== obs circles by season ===")
    for season in ["hydro", "ond", "jfm", "amj", "jas"]:
        obs = (base / season / "obs_signif_norast.svg").read_text(encoding="utf-8")
        c = circle_widths(obs)
        print(f"{season}: {sum(c.values())} circles, widths {dict(c)}")

    mod = (base / "ond" / "mod_signif_norast.svg").read_text(encoding="utf-8")
    obs = (base / "ond" / "obs_signif_norast.svg").read_text(encoding="utf-8")
    asm = asm_path.read_text(encoding="utf-8")

    for name, text in [("mod", mod), ("obs", obs), ("asm", asm)]:
        c = circle_widths(text)
        w = list(c.elements())
        print(
            f"{name}: {sum(c.values())} circles, {len(c)} unique widths, "
            f"range {min(w) if w else 'n/a'}-{max(w) if w else 'n/a'}"
        )
        if len(c) <= 5:
            print(f"  {dict(c)}")
        else:
            print(f"  top: {c.most_common(5)}")


if __name__ == "__main__":
    main()
