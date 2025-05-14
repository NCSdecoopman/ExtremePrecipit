from pathlib import Path

import numpy as np
import polars as pl

import plotly.io as pio

from app.utils.map_utils import plot_map
from app.utils.data_utils import match_and_compare
from app.utils.stats_utils import generate_metrics
from app.utils.scatter_plot_utils import generate_scatter_plot_interactive

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_map import pipeline_map

def pipeline_data_quarto(
        config: dict,
        stat_choice: str,
        echelle: str,
        min_year: int,
        max_year: int,
        season_choice: str,
        unit_choice: str = None,
        missing_choice: int = 0.15,
        quantile_choice: int = 0.999
    ):
    
    # Définir l'unité par défaut si non spécifiée
    if unit_choice is None:
        unit_choice = "mm/j" if echelle == "quotidien" else "mm/h"

    # Déterminer la clé de l’échelle à partir de l’unité
    scale_map = {
        "mm/j": "mm_j",
        "mm/h": "mm_h"
    }
    scale_choice_key = scale_map.get(unit_choice, "")

    # Adapter l'année minimale si l'échelle est saisonnière
    if echelle in {"hydro", "djf"}:
        min_year += 1

    # Préparation des paramètres pour pipeline_data
    params_load = (
        stat_choice,
        scale_choice_key,
        min_year,
        max_year,
        season_choice,
        missing_choice,
        quantile_choice
    )

    result = pipeline_data(params_load, config, use_cache=False)

    return result


def pipeline_map_quarto(
    name: str,
    result: dict,
    echelle: str,
    stat_choice_label: str,
    unit_choice: str = None,
    height: int = 500
):

    if unit_choice == None:
        unit_choice = "mm/j" if echelle == "quotidien" else "mm/h"

    Path("assets").mkdir(exist_ok=True)

    deck_path = f"assets/deck_map_{name}.html"

    params_map = (
        stat_choice_label,
        result,
        unit_choice, 
        height
    )
    layer, scatter_layer, tooltip, view_state, legend = pipeline_map(
        params_map,
        param_view={"latitude":46.9, "longitude":1.7, "zoom":4.65}
    )
    
    # Carte Pydeck
    deck = plot_map([layer, scatter_layer], view_state, tooltip)

    # Enregistrement de la carte
    deck.to_html(deck_path, notebook_display=False)

    # Affichage côte à côte
    html_map_legend = f"""
    <div style="display: flex; flex-direction: row; align-items: flex-start; margin-top: 10px;">
        <iframe loading="lazy" src="{deck_path}" height="{height}" frameborder="0" style="flex: 3; width: 0; min-width: 0; max-width: 100%;"></iframe>
        <div style="flex: 1; max-width: 220px; margin-left: 5px;">{legend}</div>
    </div>
    """

    return html_map_legend


def pipeline_scatter_quarto(
    name: str,
    result: dict,
    echelle: str,
    stat_choice_label: str,
    unit_choice: str = None,
    missing_choice: int = 0.15,
    height: int = 500
):
    
    scatter_path = f"assets/scatter_plot_{name}.html"

    df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
    obs_vs_mod = match_and_compare(result["observed"], result["modelised"], result["column"], df_obs_vs_mod)

    me, _, _, r2 = generate_metrics(obs_vs_mod)

    mean_mod = obs_vs_mod.select(pl.col("AROME").mean()).to_numpy()
    mean_obs = obs_vs_mod.select(pl.col("Station").mean()).to_numpy()
    n = obs_vs_mod.shape[0]
    delta = me / np.mean([mean_mod, mean_obs]) # écart relatif
 
    fig_scatter = generate_scatter_plot_interactive(
        df=obs_vs_mod, 
        stat_choice=stat_choice_label,
        unit_label=unit_choice, 
        height=height-60
    )
 
    fig_scatter.update_layout(
        template="simple_white",
        margin=dict(l=100, r=0, t=50, b=50),
        xaxis=dict(title=dict(text=f"AROME ({unit_choice})"), showticklabels=True),
        yaxis=dict(title=dict(text=f"Stations ({unit_choice})"), showticklabels=True)
    )

    pio.write_html(
        fig_scatter,
        file=scatter_path,
        include_plotlyjs='cdn',
        full_html=False
    )

    html_scatter = f"""
    <div style="height: {height-10}px; border: 1px solid #ccc; border-radius: 6px; display: flex; align-items: center; justify-content: center; overflow: hidden; margin-top: 10px;">
        <iframe loading="lazy" src="{scatter_path}" width="100%" height="100%" frameborder="0" style="flex: 3; width: 0; min-width: 0; max-width: 100%;"></iframe>
    </div>

    <div class="metric-caption">
        <strong>r²</strong> = {r2:.3f} &nbsp;|&nbsp; <strong>ME</strong> = {me:.3f} &nbsp;|&nbsp; <strong>n</strong> = {obs_vs_mod.shape[0]:.0f} (Tx NaN ≤ {missing_choice})
    </div>
    """


    return html_scatter, r2, me, n, delta


def pipeline_map_legend_scatter(
    name: str,
    result: dict,
    echelle: str,
    stat_choice_label: str,
    unit_choice: str = None,
    missing_choice: int = 0.15,
    height: int = 500):

    if unit_choice == None:
        unit_choice = "mm/j" if echelle == "quotidien" else "mm/h"

    # Affichage côte à côte
    html_map_legend = pipeline_map_quarto(
        name,
        result,
        echelle,
        stat_choice_label,
        unit_choice,
        height
    )

    html_scatter, r2, me, n, delta = pipeline_scatter_quarto(
        name,
        result,
        echelle,
        stat_choice_label,
        unit_choice,
        missing_choice,
        height
    )

    return html_map_legend, html_scatter, r2, me, n, delta

def pipeline_title(
    title: str,
    year_display_min: int,
    year_display_max: int,
    echelle: str
):
    return f"""
    <div style="font-size: 18px; color: #333;">
        <p style="font-size: 22px; font-weight: bold; color: #2c3e50;">{title}</p>
        <p style="font-size: 18px; color: #3498db;">
            de <span style="font-weight: bold; color: #e74c3c;">{year_display_min}</span> à <span style="font-weight: bold; color: #e74c3c;">{year_display_max}</span> 
            (Saison : <span style="font-weight: bold; color: #f39c12;">année hydrologique</span> | Echelle : <span style="font-weight: bold; color: #f39c12;">{echelle}</span>)
        </p> 
    </div>

    """

def pipeline_show_html(map_legend, scatter):
    return f"""
    <div class="columns" style="display: flex; gap: 0px; margin: 0;">
        <div class="column" style="width: 50%;">{map_legend}</div>
        <div class="column" style="width: 50%;">{scatter}</div>
    </div>
    """