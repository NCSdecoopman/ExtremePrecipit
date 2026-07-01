import sys
import os
from pathlib import Path
from typing import Tuple

from svgutils.transform import fromfile, SVGFigure
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def _to_px(value: str | None) -> float:
    if not value:
        return 0.0
    value = value.strip()
    num = ''
    unit = ''
    for ch in value:
        if ch.isdigit() or ch in '.-':
            num += ch
        else:
            unit += ch
    if not num:
        return 0.0
    numf = float(num)
    unit = unit.strip().lower()
    if unit in ('', 'px'):
        return numf
    if unit == 'mm':
        return numf * 3.779527559055  # 96 dpi
    if unit == 'cm':
        return numf * 37.79527559055
    if unit == 'in':
        return numf * 96
    if unit == 'pt':
        return numf * 1.3333333333333
    return numf

def _dims(fig) -> Tuple[float, float]:
    root = fig.root
    viewbox = root.get('viewBox')
    if viewbox:
        parts = [p for p in viewbox.replace(',', ' ').split() if p]
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])
    w_attr = root.get('width')
    h_attr = root.get('height')
    if w_attr and h_attr:
        return _to_px(w_attr), _to_px(h_attr)
    w, h = fig.get_size()
    return _to_px(w), _to_px(h)

from metrics_utils import load_metrics_csv

def assemble(arome: Path, stations: Path, legend: Path, output: Path) -> str:
    fig_arome = fromfile(str(arome))
    fig_stations = fromfile(str(stations))
    w_a, h_a = _dims(fig_arome)
    w_s, h_s = _dims(fig_stations)
    scale_s = h_a / h_s if h_s != 0 else 1.0
    root_a = fig_arome.getroot()
    root_s = fig_stations.getroot()
    root_s.scale(scale_s)
    w_s_scaled, h_s_scaled = w_s * scale_s, h_s * scale_s
    width  = w_a + w_s_scaled
    height = max(h_a, h_s_scaled)
    canvas = SVGFigure(f"{width}px", f"{height}px")
    canvas.root.set("viewBox", f"0 0 {width} {height}")
    root_a.moveto(0, 0)
    root_s.moveto(w_a, 0)
    canvas.append([root_a, root_s])
    os.makedirs(os.path.dirname(str(output)), exist_ok=True)
    canvas.save(str(output))
    pdf_path = str(output)[:-4] + ".pdf"
    
    # Strip DTDs to prevent network hangs in svglib
    content = Path(output).read_text(encoding="utf-8")
    import re
    cleaned = re.sub(r'<!DOCTYPE[^>]*>', '', content)
    temp_svg = Path(output).parent / (Path(output).name + "_temp_cleaned.svg")
    temp_svg.write_text(cleaned, encoding="utf-8")
    try:
        drawing = svg2rlg(str(temp_svg))
        renderPDF.drawToFile(drawing, pdf_path)
    finally:
        if temp_svg.exists():
            temp_svg.unlink()
    return str(output)

print("Starting fig7 assembly...")
trend_pluie_jan = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/jan/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/jan/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_jan.svg",
)
trend_pluie_fev = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/fev/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/fev/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_fev.svg",
)
trend_pluie_mar = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/mar/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/mar/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_mar.svg",
)
trend_pluie_avr = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/avr/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/avr/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_avr.svg",
)
trend_pluie_mai = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/mai/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/mai/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_mai.svg",
)
trend_pluie_jui = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/jui/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/jui/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_jui.svg",
)
trend_pluie_juill = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/juill/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/juill/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_juill.svg",
)
trend_pluie_aou = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/aou/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/aou/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_aou.svg",
)
trend_pluie_sep = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/sep/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/sep/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_sep.svg",
)
trend_pluie_oct = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/oct/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/oct/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_oct.svg",
)
trend_pluie_nov = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/nov/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/nov/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_nov.svg",
)
trend_pluie_dec = assemble(
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/dec/mod_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/dec/obs_signif_rast.svg",
"../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/legend_signif.svg",
"figures/trend_horaire_pluie_dec.svg",
)

metrics_h = load_metrics_csv("../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/metrics_signif.csv")

def _month_metrics(season: str):
    row = metrics_h.loc[metrics_h["season"] == season].iloc[0]
    return float(row["r"]), int(row["n"]), float(row["me"])

r_jan, n_jan, me_jan = _month_metrics("jan")
r_fev, n_fev, me_fev = _month_metrics("fev")
r_mar, n_mar, me_mar = _month_metrics("mar")
r_avr, n_avr, me_avr = _month_metrics("avr")
r_mai, n_mai, me_mai = _month_metrics("mai")
r_jun, n_jun, me_jun = _month_metrics("jui")
r_jul, n_jul, me_jul = _month_metrics("juill")
r_aug, n_aug, me_aug = _month_metrics("aou")
r_sep, n_sep, me_sep = _month_metrics("sep")
r_oct, n_oct, me_oct = _month_metrics("oct")
r_nov, n_nov, me_nov = _month_metrics("nov")
r_dec, n_dec, me_dec = _month_metrics("dec")

macros = rf"""
\newcommand{{\rJAN}}{{{r_jan:.2f}}}
\newcommand{{\nJAN}}{{{n_jan}}}
\newcommand{{\meJAN}}{{{me_jan:+.2f}}}

\newcommand{{\rFEB}}{{{r_fev:.2f}}}
\newcommand{{\nFEB}}{{{n_fev}}}
\newcommand{{\meFEB}}{{{me_fev:+.2f}}}

\newcommand{{\rMAR}}{{{r_mar:.2f}}}
\newcommand{{\nMAR}}{{{n_mar}}}
\newcommand{{\meMAR}}{{{me_mar:+.2f}}}

\newcommand{{\rAPR}}{{{r_avr:.2f}}}
\newcommand{{\nAPR}}{{{n_avr}}}
\newcommand{{\meAPR}}{{{me_avr:+.2f}}}

\newcommand{{\rMAY}}{{{r_mai:.2f}}}
\newcommand{{\nMAY}}{{{n_mai}}}
\newcommand{{\meMAY}}{{{me_mai:+.2f}}}

\newcommand{{\rJUN}}{{{r_jun:.2f}}}
\newcommand{{\nJUN}}{{{n_jun}}}
\newcommand{{\meJUN}}{{{me_jun:+.2f}}}

\newcommand{{\rJUL}}{{{r_jul:.2f}}}
\newcommand{{\nJUL}}{{{n_jul}}}
\newcommand{{\meJUL}}{{{me_jul:+.2f}}}

\newcommand{{\rAUG}}{{{r_aug:.2f}}}
\newcommand{{\nAUG}}{{{n_aug}}}
\newcommand{{\meAUG}}{{{me_aug:+.2f}}}

\newcommand{{\rSEP}}{{{r_sep:.2f}}}
\newcommand{{\nSEP}}{{{n_sep}}}
\newcommand{{\meSEP}}{{{me_sep:+.2f}}}

\newcommand{{\rOCT}}{{{r_oct:.2f}}}
\newcommand{{\nOCT}}{{{n_oct}}}
\newcommand{{\meOCT}}{{{me_oct:+.2f}}}

\newcommand{{\rNOV}}{{{r_nov:.2f}}}
\newcommand{{\nNOV}}{{{n_nov}}}
\newcommand{{\meNOV}}{{{me_nov:+.2f}}}

\newcommand{{\rDEC}}{{{r_dec:.2f}}}
\newcommand{{\nDEC}}{{{n_dec}}}
\newcommand{{\meDEC}}{{{me_dec:+.2f}}}
"""

with open("macros_fig7.tex", "w", encoding="utf-8") as f:
    f.write(macros)
print("Finished fig7 assembly and macros generation.")
