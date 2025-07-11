from pathlib import Path

import numpy as np
import polars as pl

import plotly.io as pio

from app.utils.map_utils import plot_map
from app.utils.data_utils import match_and_compare
from app.utils.stats_utils import generate_metrics
from app.utils.scatter_plot_utils import generate_scatter_plot_interactive

from app.pipelines.import_data import pipeline_data, pipeline_data_gev
from app.pipelines.import_map import pipeline_map

from app.utils.config_utils import get_readable_season

def pipeline_total_title_map_legend_scatter_delta(
    title: str,
    year_min: int,
    year_max: int,
    echelle: str,
    html_map_legend: str,
    html_scatter: str,
    season_choice: str="hydro",
    me: int=None,
    percent: int=None,
    r2: int=None,
    unit: str=None,
    nb_after_comma: int=1
):
    if unit is None:
        unit = "mm/j" if echelle == "quotidien" else "mm/h"


    title_show = pipeline_title(title, year_min, year_max, echelle, season_choice)
    display(HTML(title_show))
    display(HTML(pipeline_show_html(html_map_legend, html_scatter)))
    
    if r2 is not None:
        display(HTML(f"r² = {r2:+.3f}"))
    elif me is not None:
        if percent is not None:
            display(HTML(f"Δ (AROME - Stations) : {me:+.{nb_after_comma}f} {unit} ({percent*100:+.1f}%)"))
        else:
            display(HTML(f"Δ (AROME - Stations) : {me:+.{nb_after_comma}f} {unit}"))
            

def pipeline_data_quarto(
        config: dict,
        stat_choice: str,
        echelle: str,
        min_year: int,
        max_year: int,
        season_choice: str,
        unit_choice: str = None,
        missing_choice: int = 0.10,
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

    if echelle == "quotidien":
        scale_choice = "Journalière"
    else:
        scale_choice = echelle.lower()

    # Préparation des paramètres pour pipeline_data
    params_load = (
        stat_choice,
        scale_choice_key,
        min_year,
        max_year,
        season_choice,
        missing_choice,
        quantile_choice,
        scale_choice
    )

    result = pipeline_data(params_load, config, use_cache=False)

    return result

def pipeline_data_gev_quarto(
        config: dict,
        model_name: str,
        echelle: str,
        min_year: int,
        max_year: int,
        season_choice: str,
        T_choice: int = 10,
        par_X_annees: int = 10,
        quantile_choice: float = 0.999,
        param_choice: str = "Δqᵀ"
    ):

    # Préparation des chemins vers les dossiers Parquet GEV
    mod_dir = Path(config["gev"]["modelised"]) / echelle / season_choice
    obs_dir = Path(config["gev"]["observed"]) / echelle / season_choice

    params = {
        "mod_dir": mod_dir,
        "obs_dir": obs_dir,
        "model_name": model_name,
        "echelle": echelle,
        "T_choice": T_choice,
        "par_X_annees": par_X_annees,
        "quantile_choice": quantile_choice,
        "min_year_choice": min_year,
        "max_year_choice": max_year,
        "param_choice": param_choice,
        "config": config  # pour l’accès à rupture, etc.
    }

    result = pipeline_data_gev(params)
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
    missing_choice: int = 0.10,
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
    missing_choice: int = 0.10,
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
    echelle: str,
    season_choice: str="hydro",
):
    
    if season_choice == None:
        return f"""
        <div style="font-size: 18px; color: #333;">
            <p><span style="font-size: 22px; font-weight: bold; color: #2c3e50;">{title}</span>
            <span style="font-size:18px;color:#3498db;white-space:nowrap;display:inline-block;margin:0;">
            de <span style="font-weight:bold;color:#e74c3c">{year_display_min}</span> à <span style="font-weight:bold;color:#e74c3c">{year_display_max}</span> pour l'échelle : <span style="font-weight:bold;color:#f39c12">{echelle}</span>
            </span>
        </div>
        """ 
    return f"""
    <div style="font-size: 18px; color: #333;">
        <p style="font-size: 22px; font-weight: bold; color: #2c3e50;">{title}</p>
        <p style="font-size: 18px; color: #3498db;">
            de <span style="font-weight: bold; color: #e74c3c;">{year_display_min}</span> à <span style="font-weight: bold; color: #e74c3c;">{year_display_max}</span> 
            (Saison : <span style="font-weight: bold; color: #f39c12;">{get_readable_season(season_choice)}</span> | Echelle : <span style="font-weight: bold; color: #f39c12;">{echelle}</span>)
        </p> 
    </div>
    """

def pipeline_show_html(map_legend, scatter: None):
    if scatter is None:
        return f"""
        <div class="columns" style="display: flex; gap: 0; margin: 0; width: 100%; max-width: 100%; padding: 0;">
            <div class="column" style="flex: 1; width: 100%; padding: 0;">{map_legend}</div>
        </div>
        """
    else:
        return f"""
        <div class="columns" style="display: flex; gap: 0px; margin: 0;">
            <div class="column" style="width: 50%;">{map_legend}</div>
            <div class="column" style="width: 50%;">{scatter}</div>
        </div>
        """


def pipeline_quarto_gev_half_france(
    model_name: str,
    season_choice: str,
    echelle: str,
    par_X_annees: int,
    result: dict, 
    half: str = "nord",           # "nord" ou "sud"
    height: int = 500,
    assets_dir: str = "assets"
):
    # 1. Limites latitudes
    if half == "nord":
        min_lat, max_lat = 46.2, 51.2
        min_lon, max_lon = -np.inf, np.inf
    elif half == "sud":
        min_lat, max_lat = 41.0, 46.2
        min_lon, max_lon = -2, 7.7
    else:
        raise ValueError("half doit être 'nord' ou 'sud'")

    Path(assets_dir).mkdir(exist_ok=True)
    name = f"gev_{model_name}_{season_choice}_{half}"

    # Filtrage spatial
    def filter_latlon(df, min_lat, max_lat, min_lon, max_lon):
        return df.filter(
            (pl.col("lat") >= min_lat) & (pl.col("lat") < max_lat) &
            (pl.col("lon") >= min_lon) & (pl.col("lon") < max_lon)
        )
    result["modelised_show"] = filter_latlon(result["modelised_show"], min_lat, max_lat, min_lon, max_lon)
    result["observed_show"] = filter_latlon(result["observed_show"], min_lat, max_lat, min_lon, max_lon)

    # 5. Carte (Pydeck + légende)
    params_map = (
        result["stat_choice_key"] if "stat_choice_key" in result else "Δqᵀ",
        result,
        f"mm/j/{par_X_annees} ans",
        height
    )
    layer, scatter_layer, tooltip, view_state, legend = pipeline_map(
        params_map,
        param_view={
            "latitude": 44.3 if half == "sud" else 48.0, 
            "longitude": 3.1, 
            "zoom": 6.3
        }
    )

    deck = plot_map([layer, scatter_layer], view_state, tooltip)
    deck_path = f"{assets_dir}/deck_map_{name}.html"
    deck.to_html(deck_path, notebook_display=False)

    html_map = f"""
    <div style="display: flex; flex-direction: row; align-items: flex-start; margin-top: 10px;">
        <iframe loading="lazy" src="{deck_path}" height="{height}" frameborder="0" style="flex: 3; width: 0; min-width: 0; max-width: 100%;"></iframe>
        <div style="flex: 1; max-width: 220px; margin-left: 5px;">{legend}</div>
    </div>
    """

    # 6. Scatterplot + métriques
    scatter_html, r2, me, n, delta = pipeline_scatter_quarto_gev(
        name,
        result,
        echelle,
        stat_choice_label="Δqᵀ",
        unit_choice=f"mm/j/{par_X_annees} ans",
        height=height
    )

    return html_map, scatter_html, r2, me, n, delta

# -- Sous-fonction scatter
def pipeline_scatter_quarto_gev(
    name: str,
    result: dict,
    echelle: str,
    stat_choice_label: str,
    unit_choice: str = None,
    height: int = 500
):
    scatter_path = f"assets/scatter_plot_{name}.html"
    echelle_match = "quotidien" if echelle == "quotidien" else "horaire"
    df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle_match}.csv")
    obs_vs_mod = match_and_compare(result["observed"], result["modelised"], result["column"], df_obs_vs_mod)

    me, _, _, r2 = generate_metrics(obs_vs_mod)
    n = obs_vs_mod.shape[0]
    mean_mod = obs_vs_mod.select(pl.col("AROME").mean()).item()
    mean_obs = obs_vs_mod.select(pl.col("Station").mean()).item()
    delta = me / np.mean([mean_mod, mean_obs]) if (mean_mod is not None and mean_obs is not None) else None

    fig_scatter = generate_scatter_plot_interactive(
        df=obs_vs_mod, 
        stat_choice=stat_choice_label,
        unit_label=unit_choice, 
        height=height-60
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
        <strong>r²</strong> = {r2:.3f} &nbsp;|&nbsp; <strong>ME</strong> = {me:.3f} &nbsp;|&nbsp; <strong>n</strong> = {n:.0f}
    </div>
    """
    return html_scatter, r2, me, n, delta
