import sys
import os
from pathlib import Path
from typing import Tuple

from svgutils.transform import fromfile, SVGFigure
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import numpy as np
import pandas as pd

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

def assemble_vertical(arome: Path, stations: Path, legend: Path, output: Path) -> None:
    fig_arome = fromfile(str(arome))
    fig_stations = fromfile(str(stations))
    fig_legend = fromfile(str(legend))
    w_arome, h_arome = _dims(fig_arome)
    w_stations, h_stations = _dims(fig_stations)
    w_leg, h_leg = _dims(fig_legend)
    w_maps = max(w_arome, w_stations)
    h_maps = h_arome + h_stations
    height = h_maps
    scale_leg = (1.5 * h_arome) / h_leg
    w_leg_scaled = w_leg * scale_leg
    h_leg_scaled = h_leg * scale_leg
    width = w_maps + w_leg_scaled
    canvas = SVGFigure(f"{width}px", f"{height}px")
    canvas.root.set('viewBox', f"0 0 {width} {height}")
    root_arome = fig_arome.getroot()
    root_stations = fig_stations.getroot()
    root_legend = fig_legend.getroot()
    root_legend.scale(scale_leg, scale_leg)
    root_stations.moveto(0, h_arome)
    y_leg = (h_maps - h_leg_scaled) / 2
    root_legend.moveto(w_maps, y_leg)
    canvas.append([root_arome, root_stations, root_legend])
    os.makedirs(os.path.dirname(str(output)), exist_ok=True)
    canvas.save(str(output))
    pdf_path = str(output)[:-4] + ".pdf"
    
    # Render using svglib
    drawing = svg2rlg(str(output))
    renderPDF.drawToFile(drawing, pdf_path)
    return str(output)

def combined_metrics_df(name_file: str, base_dir: Path = Path("../outputs")):
    csv_paths = list(base_dir.rglob(name_file))
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            df["source"] = str(path.relative_to(base_dir))
            frames.append(df)
        except Exception as exc:
            sys.stderr.write(f"Fichier ignore {path}: {exc}\n")
    if not frames:
        raise SystemExit("Aucun metrics.csv trouve !")
    combined = pd.concat(frames, ignore_index=True)
    mask_q = combined["source"].str.contains("quotidien_reduce", na=False)
    combined.loc[mask_q & (combined["echelle"] == "quotidien"), "echelle"] = "quotidien_reduce"
    mask_h = combined["source"].str.contains("horaire_reduce", na=False)
    combined.loc[mask_h & (combined["echelle"] == "horaire"), "echelle"] = "horaire_reduce"
    return combined

combined = combined_metrics_df("metrics.csv")
combined_nr = combined_metrics_df("metrics.csv", Path("../outputs_nr10"))

# Largeur cible des légendes diff/rdiff (ticks sur 4 car. + ylabel), en unités viewBox.
_LEGEND_SLOT_W = 210.0

def assemble_un(carte: Path, legend: Path, output: Path) -> str:
    fig_map = fromfile(str(carte))
    fig_leg = fromfile(str(legend))
    w_map, h_map = _dims(fig_map)
    w_leg, h_leg = _dims(fig_leg)
    if h_map == 0 or h_leg == 0:
        raise ValueError("Hauteur nulle detectee.")
    scale_leg = h_map / h_leg
    w_leg_scaled = w_leg * scale_leg
    h_leg_scaled = h_leg * scale_leg
    slot_w_scaled = _LEGEND_SLOT_W * scale_leg
    width = w_map + max(slot_w_scaled, w_leg_scaled)
    height = h_map
    canvas = SVGFigure(f"{width}px", f"{height}px")
    canvas.root.set("viewBox", f"0 0 {width} {height}")
    root_map = fig_map.getroot()
    root_leg = fig_leg.getroot()
    root_leg.scale(scale_leg, scale_leg)
    root_map.moveto(0, 0)
    y_leg = max(0, (h_map - h_leg_scaled) / 2.0)
    x_leg = w_map + max(0.0, slot_w_scaled - w_leg_scaled)
    root_leg.moveto(x_leg, y_leg)
    canvas.append([root_map, root_leg])
    os.makedirs(os.path.dirname(str(output)), exist_ok=True)
    canvas.save(str(output))
    
    # Render using svglib
    drawing = svg2rlg(str(output))
    renderPDF.drawToFile(drawing, str(output)[:-4] + ".pdf")
    return str(output)

print("Starting assembly...")

jour_pluie = assemble_vertical(
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/hydro/mod_norast.svg"),
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/hydro/obs_norast.svg"),
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/legend.svg"),
  Path("figures/jour_pluie.svg")
)
mean_pluie_jour = assemble_vertical(
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/hydro/mod_norast.svg"),
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/hydro/obs_norast.svg"),
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/legend.svg"),
  Path("figures/mean_pluie_jour.svg")
)
nr_pluie_jour = assemble_vertical(
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/hydro/mod_norast.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/hydro/obs_norast.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/legend.svg"),
  Path("figures/nr_pluie_jour.svg")
)
nr_pluie_horaire = assemble_vertical(
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/hydro/mod_norast.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/hydro/obs_norast.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/legend.svg"),
  Path("figures/nr_pluie_horaire.svg")
)
jour_pluie_diff = assemble_un(
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/hydro/obs_norast_diff.svg"),
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/legend_diff.svg"),
  Path("figures/jour_pluie_diff.svg")
)
mean_pluie_jour_diff = assemble_un(
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/hydro/obs_norast_diff.svg"),
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/legend_diff.svg"),
  Path("figures/mean_pluie_jour_diff.svg")
)
nr_pluie_jour_diff = assemble_un(
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/hydro/obs_norast_diff.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/legend_diff.svg"),
  Path("figures/nr_pluie_jour_diff.svg")
)
nr_pluie_horaire_diff = assemble_un(
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/hydro/obs_norast_diff.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/legend_diff.svg"),
  Path("figures/nr_pluie_horaire_diff.svg")
)

jour_pluie_rdiff = assemble_un(
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/hydro/obs_norast_rdiff.svg"),
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/legend_rdiff.svg"),
  Path("figures/jour_pluie_rdiff.svg")
)
mean_pluie_jour_rdiff = assemble_un(
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/hydro/obs_norast_rdiff.svg"),
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/legend_rdiff.svg"),
  Path("figures/mean_pluie_jour_rdiff.svg")
)
nr_pluie_jour_rdiff = assemble_un(
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/hydro/obs_norast_rdiff.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/legend_rdiff.svg"),
  Path("figures/nr_pluie_jour_rdiff.svg")
)
nr_pluie_horaire_rdiff = assemble_un(
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/hydro/obs_norast_rdiff.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/legend_rdiff.svg"),
  Path("figures/nr_pluie_horaire_rdiff.svg")
)

df_jour_pluie = combined.loc[(combined["echelle"] == "quotidien") & (combined["season"] == "hydro") & (combined["col_calculate"] == "numday"), ["n", "r", "me", "delta"]]
df_mean_pluie_jour = combined.loc[(combined["echelle"] == "quotidien") & (combined["season"] == "hydro") & (combined["col_calculate"] == "mean"), ["n", "r", "me", "delta"]]
r_jp  = float(df_jour_pluie["r"].iloc[0]);  n_jp  = int(df_jour_pluie["n"].iloc[0])
r_mpj = float(df_mean_pluie_jour["r"].iloc[0]); n_mpj = int(df_mean_pluie_jour["n"].iloc[0])
me_jp  = float(df_jour_pluie["me"].iloc[0]);  d_jp  = float(df_jour_pluie["delta"].iloc[0])
me_mpj = float(df_mean_pluie_jour["me"].iloc[0]); d_mpj = float(df_mean_pluie_jour["delta"].iloc[0])

df_jour_nr = combined_nr.loc[(combined_nr["echelle"] == "quotidien") & (combined_nr["season"] == "hydro") & (combined_nr["col_calculate"] == "zTpa"), ["n", "r", "me", "delta"]]
df_horaire_nr = combined_nr.loc[(combined_nr["echelle"] == "horaire") & (combined_nr["season"] == "hydro") & (combined_nr["col_calculate"] == "zTpa"), ["n", "r", "me", "delta"]]
r_jn  = float(df_jour_nr["r"].iloc[0]);  n_jn  = int(df_jour_nr["n"].iloc[0])
r_hn = float(df_horaire_nr["r"].iloc[0]); n_hn= int(df_horaire_nr["n"].iloc[0])
me_jn  = float(df_jour_nr["me"].iloc[0]); me_hn = float(df_horaire_nr["me"].iloc[0])
d_jn     = float(df_jour_nr["delta"].iloc[0]); d_hn  = float(df_horaire_nr["delta"].iloc[0])

macros = rf"""
\newcommand{{\rJourPluie}}{{{r_jp:.2f}}}
\newcommand{{\nJourPluie}}{{{n_jp:d}}}
\newcommand{{\rMeanPluieJour}}{{{r_mpj:.2f}}}
\newcommand{{\nMeanPluieJour}}{{{n_mpj:d}}}
\newcommand{{\meJourPluie}}{{{me_jp:+.2f}}}
\newcommand{{\dJourPluie}}{{{d_jp:+.2f}}}
\newcommand{{\meMeanPluieJour}}{{{me_mpj:+.2f}}}
\newcommand{{\dMeanPluieJour}}{{{d_mpj:+.2f}}}
\newcommand{{\rNRJour}}{{{r_jn:.2f}}}
\newcommand{{\nNRJour}}{{{n_jn:d}}}
\newcommand{{\rNRHoraire}}{{{r_hn:.2f}}}
\newcommand{{\nNRHoraire}}{{{n_hn:d}}}
\newcommand{{\meNRJour}}{{{me_jn:+.2f}}}
\newcommand{{\meNRHoraire}}{{{me_hn:+.2f}}}
\newcommand{{\dNRJour}}{{{d_jn:+.2f}}}
\newcommand{{\dNRHoraire}}{{{d_hn:+.2f}}}
"""
with open("macros_fig3_nr.tex", "w", encoding="utf-8") as f:
    f.write(macros)
print("Finished assembling SVG/PDF component maps and macros.")
