#!/usr/bin/env python3
"""assemble_cartes.py – Assemble une carte AROME + stations + légende.

Correctifs :

* calcul fiable des dimensions (viewBox prioritaire, conversion mm/cm/pt/in → px) pour éviter
  le rognage des SVG et garantir un canvas assez grand ;
* refactorisation légère (_to_px, _dims) ;
* messages d'erreur plus explicites.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

from svgutils.transform import fromfile, SVGFigure


def _to_px(value: str | None) -> float:
    """
    Convertit une longueur SVG (px, mm, cm, pt, in) en pixels (float).

    Si `value` est None ou vide, retourne 0.
    """
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
        return numf * 3.779527559055  # 96 dpi
    if unit == 'cm':
        return numf * 37.79527559055
    if unit == 'in':
        return numf * 96
    if unit == 'pt':
        return numf * 1.3333333333333  # 1 pt = 1/72 in
    # Fallback: assume pixels
    return numf


def _dims(fig) -> Tuple[float, float]:
    """
    Renvoie (width, height) de `fig` en pixels.

    1) viewBox (les 2 derniers termes)
    2) attributs width/height de la racine
    3) fig.get_size()
    """
    root = fig.root  # Correction ici : on accède à la balise <svg>
    viewbox = root.get('viewBox')
    if viewbox:
        parts = [p for p in viewbox.replace(',', ' ').split() if p]
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])

    # attributs width/height sur la balise <svg>
    w_attr = root.get('width')
    h_attr = root.get('height')
    if w_attr and h_attr:
        return _to_px(w_attr), _to_px(h_attr)

    # fallback
    w, h = fig.get_size()
    return _to_px(w), _to_px(h)


def assemble(arome: Path, stations: Path, legend: Path, output: Path) -> None:
    # Charger les 3 SVG
    fig_arome = fromfile(str(arome))
    fig_stations = fromfile(str(stations))
    fig_legend = fromfile(str(legend))

    # Dimensions
    w_arome, h_arome = _dims(fig_arome)
    w_stations, h_stations = _dims(fig_stations)
    w_leg, h_leg = _dims(fig_legend)

    # Canvas global
    w_maps = max(w_arome, w_stations)
    h_maps = h_arome + h_stations
    height = h_maps  # somme des deux cartes

    # Facteur d'échelle pour que la légende fasse 1.5 fois la hauteur de la carte AROME
    scale_leg = (1.5 * h_arome) / h_leg
    w_leg_scaled = w_leg * scale_leg
    h_leg_scaled = h_leg * scale_leg  # = 1.5 * h_arome

    width = w_maps + w_leg_scaled  # légende à droite

    canvas = SVGFigure(f"{width}px", f"{height}px")
    canvas.root.set('viewBox', f"0 0 {width} {height}")

    # Racines
    root_arome = fig_arome.getroot()
    root_stations = fig_stations.getroot()
    root_legend = fig_legend.getroot()

    # Redimensionnement de la légende
    root_legend.scale(scale_leg, scale_leg)

    # Positionnement
    root_stations.moveto(0, h_arome)
    # Centrage vertical de la légende sur la hauteur totale
    y_leg = (h_maps - h_leg_scaled) / 2
    root_legend.moveto(w_maps, y_leg)

    canvas.append([root_arome, root_stations, root_legend])
    canvas.save(str(output))
    print(f"✅  Carte assemblée\xa0: {output.resolve()}")
    


def main(argv: list[str]) -> None:
    if len(argv) != 5:
        print(__doc__)
        sys.exit(1)
    assemble(Path(argv[1]), Path(argv[2]), Path(argv[3]), Path(argv[4]))


if __name__ == "__main__":
    main(sys.argv)
