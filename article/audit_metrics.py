"""Detect compare-folder mismatches and verify figure macros against explicit CSV paths."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = Path(__file__).resolve().parent

DUPLICATE_CHECKS = [
    ("fig5", Path("outputs/maps/gev_z_T_p/quotidien/compare_5/sat_99.0/metrics_signif.csv"), "quotidien", "z_T_p", ["hydro", "ond", "jfm", "amj", "jas"]),
    ("fig6", Path("outputs/maps/gev_z_T_p/horaire/compare_5/sat_90.0/metrics_signif.csv"), "horaire", "z_T_p", ["hydro", "ond", "jfm", "amj", "jas"]),
    ("fig7", Path("outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/metrics_signif.csv"), "horaire", "z_T_p", ["jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec"]),
    ("fig3 numday", Path("outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/metrics.csv"), "quotidien", "numday", ["hydro"]),
    ("fig3 mean", Path("outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/metrics.csv"), "quotidien", "mean", ["hydro"]),
    ("fig3 zTpa q", Path("outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/metrics.csv"), "quotidien", "zTpa", ["hydro"]),
    ("fig3 zTpa h", Path("outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/metrics.csv"), "horaire", "zTpa", ["hydro"]),
    ("annexe quot", Path("outputs_nr10/maps/gev_zTpa/quotidien/compare_5/sat_99.0/metrics.csv"), "quotidien", "zTpa", ["ond", "jfm", "amj", "jas"]),
    ("annexe hor", Path("outputs_nr10/maps/gev_zTpa/horaire/compare_5/sat_99.0/metrics.csv"), "horaire", "zTpa", ["ond", "jfm", "amj", "jas"]),
]

MACRO_CHECKS = [
    ("fig7 jan ME", Path("outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/metrics_signif.csv"), "jan", "me", ARTICLE / "macros_fig7.tex", r"\\meJAN\}\{([+-]?\d+\.\d+)\}"),
    ("fig5 hydro ME", Path("outputs/maps/gev_z_T_p/quotidien/compare_5/sat_99.0/metrics_signif.csv"), "hydro", "me", ARTICLE / "macros_fig5.tex", r"\\meHYDRO\}\{([+-]?\d+\.\d+)\}"),
    ("fig6 hydro ME", Path("outputs/maps/gev_z_T_p/horaire/compare_5/sat_90.0/metrics_signif.csv"), "hydro", "me", ARTICLE / "macros_fig6.tex", r"\\meHYDRO\}\{([+-]?\d+\.\d+)\}"),
    ("fig3 dNRJour", Path("outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/metrics.csv"), "hydro", "delta", ARTICLE / "macros_fig3_nr.tex", r"\\dNRJour\}\{([+-]?\d+\.\d+)\}"),
]


def rglob_rows(base: Path, echelle: str, col: str, season: str) -> list[tuple[str, float]]:
    rows = []
    for path in base.rglob("metrics*.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for _, row in df.iterrows():
            if (
                str(row.get("echelle", "")) == echelle
                and str(row.get("col_calculate", "")).replace("_signif", "") == col
                and str(row.get("season", "")).lower() == season
            ):
                rows.append((str(path), float(row["me"])))
    return rows


def csv_value(path: Path, season: str, column: str) -> float:
    df = pd.read_csv(ROOT / path)
    row = df.loc[df["season"].str.lower() == season.lower()]
    if row.empty:
        raise ValueError(f"No row season={season} in {path}")
    return float(row[column].iloc[0])


def macro_value(macro_path: Path, pattern: str) -> float:
    text = macro_path.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Pattern not found in {macro_path.name}: {pattern}")
    return float(match.group(1))


def main() -> int:
    print("=== Duplicate compare-folder audit (rglob would pick wrong file?) ===")
    dup_fail = 0
    for name, expected_path, echelle, col, seasons in DUPLICATE_CHECKS:
        base = ROOT / expected_path.parts[0]
        expected = pd.read_csv(ROOT / expected_path)
        for season in seasons:
            rows = rglob_rows(base, echelle, col, season)
            if len(rows) <= 1:
                continue
            good = float(expected.loc[expected["season"].str.lower() == season, "me"].iloc[0])
            first_path, first_me = rows[0]
            ok = abs(first_me - good) < 1e-4
            status = "OK" if ok else "FAIL"
            if not ok:
                dup_fail += 1
            print(f"{status:4} {name:12} {season:5} dup={len(rows)} rglob_first={first_me:+.2f} expected={good:+.2f}")

    print("\n=== Macro vs explicit CSV (must all be OK) ===")
    macro_fail = 0
    for label, csv_path, season, column, macro_path, pattern in MACRO_CHECKS:
        if not macro_path.exists():
            print(f"SKIP {label}: missing {macro_path.name}")
            continue
        try:
            expected = csv_value(csv_path, season, column)
            actual = macro_value(macro_path, pattern)
            ok = abs(expected - actual) < 0.015  # macros rounded to 2 decimals
            status = "OK" if ok else "FAIL"
            if not ok:
                macro_fail += 1
            print(f"{status:4} {label:16} macro={actual:+.2f} csv={expected:+.2f}")
        except Exception as exc:
            macro_fail += 1
            print(f"FAIL {label:16} {exc}")

    total_fail = macro_fail
    if dup_fail:
        print(f"\nNote: {dup_fail} duplicate compare_* paths exist on disk; scripts must use explicit CSV paths (not rglob).")
    print(f"\nMacro failures: {total_fail}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
