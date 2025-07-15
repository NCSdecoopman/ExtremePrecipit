import argparse
import shutil
from pathlib import Path
from typing import Sequence, Mapping, Tuple, Optional

from src.utils.logger import get_logger
from src.utils.config_tools import load_config
from src.utils.data_utils import years_to_load, load_data, cleaning_data_observed
from app.utils.stats_utils import compute_statistic_per_point
from app.utils.data_utils import add_metadata
from app.utils.data_utils import match_and_compare

import numpy as np
import pandas as pd
import polars as pl

def import_data_stat(
    season,
    echelle,
    mesure,
    col_calculate,
    reduce_activate,
    config_mod,
    config_obs,
    scale
):
    input_dir_mod = Path(config_mod["statistics"]["path"]["outputdir"]) / "horaire"
    input_dir_obs = Path(config_obs["statistics"]["path"]["outputdir"]) / echelle
    for input_dir in [input_dir_mod, input_dir_obs]:
        # Paramètre de chargement des données
        if reduce_activate:
            _, min_year, max_year, len_serie = years_to_load("reduce", season, input_dir)
        else:
            _, min_year, max_year, len_serie = years_to_load(echelle, season, input_dir)
        cols = ["NUM_POSTE", mesure, "nan_ratio"]

        df = load_data(input_dir, season, echelle, cols, min_year, max_year)

        # Selection des stations suivant le NaN max
        df = cleaning_data_observed(df, echelle, len_serie)
        
        # Gestion des NaN
        df = df.drop_nulls(subset=[mesure])

        # Calcul de la statistics
        df = compute_statistic_per_point(df, col_calculate)

        # Ajout de l'altitude et des lat lon
        if "modelised" in str(input_dir):
            modelised = add_metadata(df, scale, type='modelised')
        elif "observed" in str(input_dir):
            observed = add_metadata(df, scale, type='observed')
     
    if col_calculate == "numday":
        col = "jours_pluie_moyen"
    elif col_calculate == "mean":
        col = f"mean_all_{scale}"
    elif col_calculate == "mean-max":
        col = f"max_mean_{scale}"
    
    return {
        "modelised": modelised,
        "observed": observed,
        "column": col,
        "season": season
    }


def import_data_gev(
    season,
    echelle,
    reduce_activate,
    col_calculate,
    config_mod,
    config_obs,
    scale
):
    gev_dir_mod = config_mod["gev"]["path"]["outputdir"]
    gev_dir_obs = config_obs["gev"]["path"]["outputdir"]
    suffix_save = "_reduce" if reduce_activate else ""

    for gev_dir in [gev_dir_mod, gev_dir_obs]:
        path_dir = Path(gev_dir) / f"{echelle}{suffix_save}" / season / "niveau_retour.parquet"
        
        if "modelised" in str(gev_dir):
            modelised = pl.read_parquet(path_dir)
            modelised = add_metadata(modelised, scale, type='modelised')
        elif "observed" in str(gev_dir):
            observed = pl.read_parquet(path_dir)
            observed = add_metadata(observed, scale, type='observed')          
     
    return {
        "modelised": modelised,
        "observed": observed,
        "column": col_calculate,
        "season": season
    }




import geopandas as gpd
from shapely.geometry import box
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.colorbar import ColorbarBase

def calculate_data_maps(
    datasets: Sequence[Mapping[str, pl.DataFrame]],
    *,

    data_type: str,
    col_calculate: str,
    echelle: str,
    reduce_activate: bool,
    titles: Sequence[str] | None = None,  # ex. ['SON', 'DJF', 'MAM', 'JJA']
    show_signif: bool = False,  # filtre significatif sur obs

    saturation_col: int = 100,
    # Grid
    side_km: float = 2.5,
) -> None:
    """Trace cartes + une légende commune (SVG)
    La plage de la légende est calculée sur **l'ensemble** des valeurs (modèle
    et observations), que l'on choisisse ou non de les afficher.
    """

    # S’assurer que le répertoire existe (création si nécessaire)
    suffix_reduce = "_reduce" if reduce_activate else ""
    name_dir = f"outputs/maps/{data_type}_{col_calculate}/{echelle}{suffix_reduce}/compare_{len(titles)}/sat_{saturation_col}"
    dir_path = Path(name_dir)

    # Si le dossier existe, on le supprime entièrement
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
        
    dir_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Conversion des DataFrames Polars en GeoDataFrames
    # ------------------------------------------------------------------
    model_gdfs: list[Optional[gpd.GeoDataFrame]] = []
    obs_gdfs: list[Optional[gpd.GeoDataFrame]] = []

    for d in datasets:
        val_col = d.get("column")

        # — Modèle ----------------------------------------------------------------
        df_m = d.get("modelised")
        if df_m is None:
            model_gdfs.append(None)
        else:
            if not isinstance(df_m, pl.DataFrame):
                raise TypeError("'modelised' doit être un pl.DataFrame.")
            gdf_m = gpd.GeoDataFrame(
                df_m.to_pandas(),
                geometry=gpd.points_from_xy(df_m["lon"].to_list(), df_m["lat"].to_list()),
                crs="EPSG:4326",
            ).to_crs("EPSG:2154")
            model_gdfs.append(gdf_m)

        # — Observations -----------------------------------------------------------
        df_o = d.get("observed")
        if df_o is None:
            obs_gdfs.append(None)
        else:
            if not isinstance(df_o, pl.DataFrame):
                raise TypeError("'observed_show' doit être un pl.DataFrame.")
            if "significant" in df_o.columns and show_signif:
                df_o = df_o.filter(pl.col("significant"))
            obs_gdfs.append(
                gpd.GeoDataFrame(
                    df_o.to_pandas(),
                    geometry=gpd.points_from_xy(df_o["lon"].to_list(), df_o["lat"].to_list()),
                    crs="EPSG:4326",
                ).to_crs("EPSG:2154")
                if df_o.height > 0
                else None
            )



    # Titres par défaut -----------------------------------------------------------
    if titles is None:
        titles = [str(i) for i in range(1, len(model_gdfs) + 1)]

    # ------------------------------------------------------------------
    # 2. Ajout de la grille carrée autour des points modélisés
    # ------------------------------------------------------------------
    half = side_km * 500  # mètres
    for gdf in (g for g in model_gdfs if g is not None):
        gdf.loc[:, "geometry"] = gdf.geometry.apply(
            lambda p: box(p.x - half, p.y - half, p.x + half, p.y + half)
        )

    # ------------------------------------------------------------------
    # 5. Ajout de la colonne _val_raw (copie intacte des valeurs)
    # ------------------------------------------------------------------
    for gdf in (g for g in model_gdfs if g is not None):
        gdf.loc[:, "_val_raw"] = gdf[val_col]
    for gdf in (g for g in obs_gdfs if g is not None):
        gdf.loc[:, "_val_raw"] = gdf[val_col]

    # ------------------------------------------------------------------
    # 6. Saturation des valeurs extrêmes (1 % sup. par exemple)
    # ------------------------------------------------------------------
    all_values: list[float] = []
    for gdf in (g for g in model_gdfs if g is not None):
        all_values.extend(gdf[val_col].to_numpy())
    for gdf in (g for g in obs_gdfs if g is not None):
        all_values.extend(gdf[val_col].to_numpy())

    if all_values and col_calculate not in ["significant", "model"]:
        seuil = np.percentile(np.abs(all_values), saturation_col)
        for gdf in (g for g in model_gdfs + obs_gdfs if g is not None):
            gdf.loc[:, val_col] = np.where(
                np.abs(gdf[val_col]) > seuil,
                np.sign(gdf[val_col]) * seuil,
                gdf[val_col],
            )

    return dir_path, val_col, titles, model_gdfs, obs_gdfs


def generate_maps(
    dir_path: str,
    model_gdfs,
    obs_gdfs,
    data_type: str,
    col_calculate: str,
    scale: str,
    
    titles: Sequence[str] | None = None,  # ex. ['SON', 'DJF', 'MAM', 'JJA']
    val_col: str=None,

    show_mod: bool = True,
    show_obs: bool = True,  # n'influence que l'affichage
    show_signif: bool = False,  # filtre significatif sur obs
    saturation_size: int = 100,

    # Points obs
    min_pt: int = 2,
    max_pt: int = 7,
    obs_edgecolor: str = "#000000",
    obs_facecolor: Optional[str] = None,
    # Relief
    relief_path: str = "data/external/niveaux/selection_courbes_niveau_france.shp",
    relief_linewidth: float = 0.3,
    relief_color: str = "#666666",
    figsize: Tuple[int, int] = (6, 6)
) -> None:
    
    if val_col not in ["significant", "model"]:
        if col_calculate == "numday":
            legend = "jours"
        elif col_calculate == "z_T_p":
            legend = "%"
        else:
            legend = scale.replace('_', '/')
    else:
        legend = ""

    # ------------------------------------------------------------------
    # 3. Masque France métropolitaine
    # ------------------------------------------------------------------
    deps = (
        gpd.read_file("https://france-geojson.gregoiredavid.fr/repo/departements.geojson")
        .to_crs("EPSG:2154")
    )
    deps_metro = deps[~deps["code"].isin(["2A", "2B"])].copy()
    deps_metro["geometry"] = deps_metro.geometry.simplify(500)
    mask = deps_metro.union_all()

    # *Les fonctions overlay/clip peuvent renvoyer des vues : on copie !*
    model_gdfs = [
        gpd.overlay(g, deps_metro[["geometry"]], how="intersection").copy() if g is not None else None
        for g in model_gdfs
    ]
    obs_gdfs = [g.clip(mask).copy() if g is not None else None for g in obs_gdfs]

    # ------------------------------------------------------------------
    # 4. Courbes de niveau
    # ------------------------------------------------------------------
    relief = (
        gpd.read_file(Path(relief_path).resolve())
        .to_crs("EPSG:2154")
        .clip(mask)
    )
    # ------------------------------------------------------------------
    # 7. Colormap & Normalisation communes
    # ------------------------------------------------------------------
    if data_type=="gev":

        if val_col in ["significant", "model"]:
            all_series = pd.concat([
                g[val_col].astype("category")
                for g in model_gdfs + obs_gdfs
                if g is not None
            ])
            all_cat   = all_series.cat.categories

            # 2) Colormap discrète
            #    On prend autant de couleurs que de catégories

            palette = [
                "#D55E00",  # rouille
                "#0072B2",  # bleu foncé
                "#8B4513",  # brun
                "#000000",  # noir
                "#009E73",  # vert
                "#F0E442"  # jaune
            ]

            cmap = ListedColormap(palette)
            # 3) Normalisation pour que chaque entier soit centré sur sa couleur
            norm = BoundaryNorm(np.arange(len(all_cat) + 1) - 0.5,
                                ncolors=len(all_cat))

        else:
            all_mins = [g[val_col].min() for g in model_gdfs if g is not None]
            all_mins += [g[val_col].min() for g in obs_gdfs if g is not None]
            all_maxs = [g[val_col].max() for g in model_gdfs if g is not None]
            all_maxs += [g[val_col].max() for g in obs_gdfs if g is not None]

            max_abs = max(abs(min(all_mins)), abs(max(all_maxs)))
            vmin, vmax = -max_abs, max_abs

            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            cmap = LinearSegmentedColormap.from_list(
                "bwr_custom", ["red", "#ffffff", "blue"], N=100
            )
        
    elif data_type=="stats":
        all_maxs = [g[val_col].max() for g in model_gdfs + obs_gdfs if g is not None]
        vmin = 0.0
        vmax = max(all_maxs)

        # --- définition échelle de couleurs ---
        custom_colorscale = [
            (0.0, "#ffffff"),  # blanc
            (0.1, "lightblue"),
            (0.3, "blue"),
            (0.45, "green"),
            (0.6, "yellow"),
            (0.7, "orange"),
            (0.8, "red"),      # rouge
            (1.0, "black"),    # noir
        ]

        # Extraction des couleurs (en ignorant les positions, la répartition sera uniforme)
        colors = [c for _, c in custom_colorscale]

        # Création d'un colormap à partir de ces couleurs, avec N=15 couleurs distinctes
        cmap = LinearSegmentedColormap.from_list("custom_discrete", colors, N=15)

        # Norme pour répartir les 15 tranches entre 0 et vmax
        levels = np.linspace(vmin, vmax, 16)  # 15 intervalles → 16 niveaux de bordure
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------------
    # 8. Export des cartes
    # ------------------------------------------------------------------
    modes = {"rast": 0, "norast": 1}

    if val_col == "model":
        TRANSFO = { 
            "s_gev":               "M₀(μ₀, σ₀)",
            "ns_gev_m1":           "M₁(μ, σ₀)",
            "ns_gev_m2":           "M₂(μ₀, σ)",
            "ns_gev_m3":           "M₃(μ, σ)",
            "ns_gev_m1_break_year":"M₁⋆(μ, σ₀)",
            "ns_gev_m2_break_year":"M₂⋆(μ₀, σ)",
            "ns_gev_m3_break_year":"M₃⋆(μ, σ)",
        }
    elif val_col == "significant":
        TRANSFO = { 
            True:               "Significatif",
            False:           "Non Significatif"
        }

    for title, gdf_m, gdf_o in zip(titles, model_gdfs, obs_gdfs):
        for mode, z in modes.items():
            fig, ax = plt.subplots(figsize=figsize)

            # 1. départements avec lignes fines
            deps_metro.boundary.plot(ax=ax, edgecolor="#AAAAAA", linewidth=0.2, zorder=0)

            # 2. contour national épaissi
            #    mask est déjà défini plus haut comme l'union de tous les départements
            gpd.GeoSeries(mask).boundary.plot(
                ax=ax,
                edgecolor="black" ,    # couleur du contour national
                linewidth=0.7,         # épaisseur plus élevée que 0.2
                zorder=1               # juste au-dessus des départements
            )


            # — Modèle ------------------------------------------------------
            if show_mod and gdf_m is not None:
                gdf_m.plot(ax=ax, column=val_col, cmap=cmap, norm=norm, linewidth=0, zorder=1)

            # — Observations -----------------------------------------------
            if show_obs and gdf_o is not None and not gdf_o.empty:

                # Pour "significant" et "model", on force une taille fixe moyenne
                if val_col in ["significant", "model"]:
                    # taille linéaire = moyenne de min_pt et max_pt
                    mean_pt = (min_pt + max_pt) / 2
                    # markersize attend la surface (pt^2)
                    gdf_o.loc[:, "_size_pt2"] = mean_pt**2
                    
                else:
                    abs_vals = gdf_o["_val_raw"].abs()
                    abs_max = abs_vals.max()
                    raw_size = ((abs_vals / abs_max) * (max_pt - min_pt) + min_pt) ** 2
                    seuil_size = np.percentile(abs_vals, saturation_size)
                    sizes = raw_size.copy()
                    sizes[abs_vals > seuil_size] = max_pt ** 2

                    gdf_o.loc[:, "_size_pt2"] = sizes

                kw = dict(
                    markersize="_size_pt2",
                    marker="o",
                    edgecolor=obs_edgecolor,
                    linewidth=0.1,
                    zorder=3,
                )

                if obs_facecolor is None:
                    gdf_o.plot(ax=ax, column=val_col, cmap=cmap, norm=norm, **kw)
                else:
                    gdf_o.plot(ax=ax, facecolor=obs_facecolor, **kw)

            # — Relief ------------------------------------------------------
            relief.plot(ax=ax, color=relief_color, linewidth=relief_linewidth, alpha=0.8, zorder=2)

            ax.set_axis_off()
            ax.set_rasterization_zorder(z)

            if mode == "rast":
                sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                if isinstance(norm, mpl.colors.BoundaryNorm) and val_col in ["significant", "model"]:
                    ticks = np.arange(len(all_cat))
                    cb = fig.colorbar(
                        sm, ax=ax,
                        boundaries = np.arange(len(all_cat)+1) - 0.5,
                        ticks = ticks,
                        spacing = "proportional",
                        shrink = 0.6,
                        aspect = 20
                    )
                    # labels dans l’ordre de all_cat
                    display_labels = [TRANSFO[code] for code in all_cat]
                    cb.ax.set_yticklabels(display_labels)
                else:
                    cb = fig.colorbar(
                        sm, ax=ax,
                        spacing = "proportional",
                        shrink = 0.6,
                        aspect = 20
                    )


            suffix_obs = "obs" if show_obs else ""
            suffix_mod = "mod" if show_mod else ""
            suffix_signif = "_signif" if show_signif else ""
            name_file = f"{suffix_mod}{suffix_obs}{suffix_signif}_{mode}"

            subdir = dir_path / title.lower() 
            subdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(subdir / f"{name_file}.svg", format="svg", bbox_inches="tight", pad_inches=0)
            fig.savefig(subdir / f"{name_file}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            logger.info(subdir / f"{name_file}.svg")

    # ------------------------------------------------------------------
    # 9. Légende seule ---------------------------------------------------
    # ------------------------------------------------------------------
    mpl.rcParams.update({"font.size": 18})
    fig2 = plt.figure(figsize=(1, figsize[1] * 2))
    ax2 = fig2.add_axes([0.25, 0.05, 0.5, 0.9])
    sm2 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])

    if isinstance(norm, mpl.colors.BoundaryNorm) and val_col in ["significant", "model"]:
        ticks = np.arange(len(all_cat))
        cb2 = ColorbarBase(
            ax2,
            cmap      = cmap,
            norm      = norm,
            boundaries= np.arange(len(all_cat)+1) - 0.5,
            ticks     = ticks,
            spacing   = "proportional",
            orientation="vertical"
        )

        # labels dans l’ordre de all_cat
        display_labels = [TRANSFO[code] for code in all_cat]
        cb2.ax.set_yticklabels(display_labels)
    else:
        # cas continu : on laisse matplotlib trouver lui‑même les ticks
        cb2 = fig2.colorbar(sm2, cax=ax2, spacing = "proportional")
        cb2.ax.set_ylabel(f"{legend}", rotation=90, fontsize=20)

    name_legend = "legend"
    fig2.savefig(dir_path / f"{name_legend}.svg", format="svg", bbox_inches="tight", pad_inches=0)
    fig2.savefig(dir_path / f"{name_legend}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig2)



def generate_scatter(
    datasets,
    dir_path: str,
    col_calculate: str,
    echelle: str,
    scale: str
):
    if col_calculate not in ["significant", "model"]:

        # Initialiser la liste avant la boucle (à mettre hors de la boucle, une seule fois)
        metrics = []  
        df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")

        for d in datasets:
            mod = d.get("modelised")
            obs = d.get("observed")
            col = d.get("column")
            season = d.get("season")

            obs_vs_mod = match_and_compare(obs, mod, col, df_obs_vs_mod)

            if col_calculate == "numday":
                legend = "jours"
            elif col_calculate == "z_T_p":
                legend = "%"
            else:
                legend = scale.replace('_', '/')
    
            # Chemin de sortie
            season_dir = dir_path / season
            season_dir.mkdir(parents=True, exist_ok=True)

            x = obs_vs_mod["AROME"].to_numpy()
            y = obs_vs_mod["Station"].to_numpy()

            # Calcul de la limite maximale commune
            max_lim = np.max([x.max(), y.max()])

            # Figure
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                x, y,
                label="Données",
                alpha=0.5,    # transparence
                s=20,         # taille des points
                facecolor="black", # couleur de remplissage
                edgecolor="white", # couleur du contour
                linewidth=0.5      # épaisseur du contour
            )
            ax.plot(
                [0, max_lim], [0, max_lim],
                linewidth=1, color="red",
                label="y = x"
            )

            ax.set_xlabel(f"AROME ({legend})")
            ax.set_ylabel(f"Stations ({legend})")
            ax.set_aspect("equal")           # carré pour lire la pente 1:1

            ax.set_xlim(0, max_lim)
            ax.set_ylim(0, max_lim)

            fig.savefig(season_dir / f"scatter.svg", format="svg", bbox_inches="tight", pad_inches=0)
            fig.savefig(season_dir / f"scatter.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            logger.info(season_dir / f"scatter.svg")

            me = np.mean(x - y)
            corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
            n = len(x)
            metrics.append({"echelle": echelle, "col_calculate": col_calculate, "season": season, "n": n, "r": corr, "me": me})

            pd.DataFrame(metrics).to_csv(dir_path/"metrics.csv", index=False)
            logger.info(f"{dir_path}/metrics.csv")

def main(args):
    global logger
    logger = get_logger(__name__)

    data_type = args.data_type
    col_calculate = args.col_calculate
    echelle = args.echelle
    season = args.season
    reduce_activate = args.reduce_activate
    sat = args.sat

    config_mod = load_config("config/modelised_settings.yaml")
    config_obs = load_config("config/observed_settings.yaml")

    datasets = []

    for e in echelle:
        
        scale = "mm_j" if e == "quotidien" else "mm_h"

        if col_calculate == "numday":
            mesure = "n_days_gt1mm"
        elif col_calculate == "mean":
            mesure = "mean_mm_h"
        else:
            mesure = f"max_{scale}"

        for s in season:
            if data_type == "stats":
                res = import_data_stat(
                    season=s,
                    echelle=e,
                    mesure=mesure,
                    col_calculate=col_calculate,
                    reduce_activate=reduce_activate,
                    config_mod=config_mod,
                    config_obs=config_obs,
                    scale=scale
                )
            
            elif data_type == "gev":
                res = import_data_gev(
                    season=s,
                    echelle=e,
                    col_calculate=col_calculate,
                    reduce_activate=reduce_activate,
                    config_mod=config_mod,
                    config_obs=config_obs,
                    scale=scale
                )

            logger.info(f"Echelle {e} - Saison {s} données générées")
            datasets.append(res)    # On range le résultat dans la bonne liste
   
        dir_path, val_col, titles, model_gdfs, obs_gdfs = calculate_data_maps(
            datasets,
            echelle=e,
            data_type=data_type,
            col_calculate=col_calculate,
            reduce_activate=reduce_activate,
            titles=[s.upper() for s in season],  # titres des 4 sous-cartes
            saturation_col=sat
        )

        for show in [True, False]:
            generate_maps(
                dir_path=dir_path,
                model_gdfs=model_gdfs,
                obs_gdfs=obs_gdfs,
                data_type=data_type,
                titles=titles,
                val_col=val_col,
                show_mod=show,
                show_obs=not show,
                col_calculate=col_calculate,
                scale=scale
            )

        generate_scatter(
            datasets=datasets,
            dir_path=dir_path,
            col_calculate=col_calculate,
            echelle=e,
            scale=scale
        )


def str2bool(v):
    if v == "True":
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de génération des représentation")
    parser.add_argument("--data_type", choices=["stats", "gev"])
    parser.add_argument("--col_calculate", choices=["numday", "mean", "mean-max", "z_T_p", "significant", "model"], default=None)
    parser.add_argument("--echelle", choices=["horaire", "quotidien"], nargs='+', default=["horaire"])
    parser.add_argument("--season", type=str, nargs='+', default=["son"])
    parser.add_argument("--reduce_activate", type=str2bool, default=False)
    parser.add_argument("--sat", type=float, default=100)

    args = parser.parse_args()  
    main(args)
