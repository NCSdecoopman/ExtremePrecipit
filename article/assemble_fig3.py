import re
import sys
import os
from pathlib import Path
from typing import Tuple

from svgutils.transform import fromfile, SVGFigure
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import numpy as np
import pandas as pd

def render_pdf_with_svglib(svg_path: Path, pdf_path: Path) -> None:
    content = svg_path.read_text(encoding="utf-8")
    import re
    cleaned = re.sub(r'<!DOCTYPE[^>]*>', '', content)
    temp_svg = svg_path.parent / (svg_path.name + "_temp_cleaned.svg")
    temp_svg.write_text(cleaned, encoding="utf-8")
    try:
        drawing = svg2rlg(str(temp_svg))
        renderPDF.drawToFile(drawing, str(pdf_path))
    finally:
        if temp_svg.exists():
            temp_svg.unlink()

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
    
    render_pdf_with_svglib(output, Path(pdf_path))
    return str(output)

# Largeur cible des légendes diff/rdiff (ticks sur 4 car. + ylabel), en unités viewBox.
_LEGEND_SLOT_W = 600.0


def _standard_legend_path(legend: Path) -> Path:
    name = legend.name
    if name.endswith("_rdiff.svg"):
        return legend.with_name(name.replace("_rdiff.svg", ".svg"))
    if name.endswith("_diff.svg"):
        return legend.with_name(name.replace("_diff.svg", ".svg"))
    return legend


def _colorbar_x_in_legend(legend: Path) -> float:
    text = legend.read_text(encoding="utf-8")
    xs = re.findall(
        r'x="([\d.]+)" y="[\d.]+" style="stroke: #000000; stroke-width: 0.8"',
        text,
    )
    if not xs:
        raise ValueError(f"Ticks de colorbar introuvables dans {legend}")
    return float(xs[0])


def _ref_colorbar_frac(w_maps: float, h_arome: float, legend_std: Path) -> float:
    """Fraction horizontale de la colorbar, identique a assemble_vertical."""
    w_leg, h_leg = _dims(fromfile(str(legend_std)))
    scale_leg = (1.5 * h_arome) / h_leg
    cb_x = _colorbar_x_in_legend(legend_std)
    cb_abs = w_maps + cb_x * scale_leg
    total_w = w_maps + w_leg * scale_leg
    return cb_abs / total_w


def assemble_un(carte: Path, legend: Path, output: Path) -> str:
    fig_map = fromfile(str(carte))
    fig_leg = fromfile(str(legend))
    w_map, h_map = _dims(fig_map)
    w_leg, h_leg = _dims(fig_leg)
    if h_map == 0 or h_leg == 0:
        raise ValueError("Hauteur nulle detectee.")
    scale_leg = h_map / h_leg
    h_leg_scaled = h_leg * scale_leg
    w_leg_scaled = w_leg * scale_leg
    
    # Calculate width_std (standard figure width with standard legend)
    legend_std = _standard_legend_path(legend)
    w_leg_std, h_leg_std = _dims(fromfile(str(legend_std)))
    scale_leg_std = (1.5 * h_map) / h_leg_std
    width_std = w_map + w_leg_std * scale_leg_std
    
    # Calculate minimum width required to fit the legend without overlap or truncation (with 5px safety margin on the right)
    min_width = w_map + w_leg_scaled + 5.0
    
    # Use the max of both to keep the canvas width (and thus map size scaling in Quarto) as close to standard as possible
    width = max(width_std, min_width)
    height = h_map
    
    ref_frac = _ref_colorbar_frac(w_map, h_map, legend_std)
    cb_x = _colorbar_x_in_legend(legend)
    
    # Theoretical aligned position of the colorbar
    x_leg = ref_frac * width - cb_x * scale_leg
    
    # Adjust x_leg if it causes the legend to extend beyond the right edge of the canvas (leaving a 5px margin)
    if x_leg + w_leg_scaled > width - 5.0:
        x_leg = width - 5.0 - w_leg_scaled
        
    # Ensure x_leg does not overlap the map on the left
    if x_leg < w_map:
        x_leg = w_map
        
    canvas = SVGFigure(f"{width}px", f"{height}px")
    canvas.root.set("viewBox", f"0 0 {width} {height}")
    root_map = fig_map.getroot()
    root_leg = fig_leg.getroot()
    root_leg.scale(scale_leg, scale_leg)
    root_map.moveto(0, 0)
    y_leg = max(0, (h_map - h_leg_scaled) / 2.0)
    root_leg.moveto(x_leg, y_leg)
    canvas.append([root_map, root_leg])
    os.makedirs(os.path.dirname(str(output)), exist_ok=True)
    canvas.save(str(output))
    
    render_pdf_with_svglib(output, Path(str(output)[:-4] + ".pdf"))
    return str(output)


print("Starting assembly...")

jour_pluie = assemble_vertical(
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/hydro/mod_rast.svg"),
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/hydro/obs_rast.svg"),
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/legend.svg"),
  Path("figures/jour_pluie.svg")
)
mean_pluie_jour = assemble_vertical(
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/hydro/mod_rast.svg"),
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/hydro/obs_rast.svg"),
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/legend.svg"),
  Path("figures/mean_pluie_jour.svg")
)
nr_pluie_jour = assemble_vertical(
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/hydro/mod_rast.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/hydro/obs_rast.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/legend.svg"),
  Path("figures/nr_pluie_jour.svg")
)
nr_pluie_horaire = assemble_vertical(
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/hydro/mod_rast.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/hydro/obs_rast.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/legend.svg"),
  Path("figures/nr_pluie_horaire.svg")
)
jour_pluie_diff = assemble_un(
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/hydro/obs_rast_diff.svg"),
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/legend_diff.svg"),
  Path("figures/jour_pluie_diff.svg")
)
mean_pluie_jour_diff = assemble_un(
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/hydro/obs_rast_diff.svg"),
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/legend_diff.svg"),
  Path("figures/mean_pluie_jour_diff.svg")
)
nr_pluie_jour_diff = assemble_un(
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/hydro/obs_rast_diff.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/legend_diff.svg"),
  Path("figures/nr_pluie_jour_diff.svg")
)
nr_pluie_horaire_diff = assemble_un(
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/hydro/obs_rast_diff.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/legend_diff.svg"),
  Path("figures/nr_pluie_horaire_diff.svg")
)

jour_pluie_rdiff = assemble_un(
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/hydro/obs_rast_rdiff.svg"),
  Path("../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/legend_rdiff.svg"),
  Path("figures/jour_pluie_rdiff.svg")
)
mean_pluie_jour_rdiff = assemble_un(
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/hydro/obs_rast_rdiff.svg"),
  Path("../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/legend_rdiff.svg"),
  Path("figures/mean_pluie_jour_rdiff.svg")
)
nr_pluie_jour_rdiff = assemble_un(
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/hydro/obs_rast_rdiff.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/legend_rdiff.svg"),
  Path("figures/nr_pluie_jour_rdiff.svg")
)
nr_pluie_horaire_rdiff = assemble_un(
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/hydro/obs_rast_rdiff.svg"),
  Path("../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/legend_rdiff.svg"),
  Path("figures/nr_pluie_horaire_rdiff.svg")
)

from metrics_utils import read_map_metrics

r_jp, n_jp, me_jp, d_jp = read_map_metrics(
    "../outputs/maps/stats_numday/quotidien/compare_1/sat_99.9/metrics.csv", "hydro"
)
r_mpj, n_mpj, me_mpj, d_mpj = read_map_metrics(
    "../outputs/maps/stats_mean/quotidien/compare_1/sat_99.0/metrics.csv", "hydro"
)
r_jn, n_jn, me_jn, d_jn = read_map_metrics(
    "../outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/metrics.csv", "hydro"
)
r_hn, n_hn, me_hn, d_hn = read_map_metrics(
    "../outputs_nr10/maps/gev_zTpa/horaire/compare_1/sat_99.0/metrics.csv", "hydro"
)

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
