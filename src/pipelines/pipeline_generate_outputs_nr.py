import argparse
import os
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

def import_data_dispo(
    season,
    echelle,
    reduce_activate,
    config_obs,
    scale
):
    input_dir = Path(config_obs["statistics"]["path"]["outputdir"]) / echelle
    # Paramètre de chargement des données
    if reduce_activate:
        _, min_year, max_year, _ = years_to_load("reduce", season, input_dir)
    else:
        _, min_year, max_year, _ = years_to_load(echelle, season, input_dir)
    cols = ["NUM_POSTE", "nan_ratio"]

    observed_load = load_data(input_dir, season, echelle, cols, min_year, max_year)

    # Limite sur le ratio de NaNs
    nan_limit = 0.10

    # Filtrage et agrégation
    observed = (
        observed_load.filter(pl.col("nan_ratio") <= nan_limit)
        .group_by("NUM_POSTE")
        .agg(pl.col("year").n_unique().alias("n_years"))
    )
    
    # Charger les metadonnées avec Polars
    observed_meta = pl.read_csv(f"data/metadonnees/observed/postes_{echelle}.csv")
    # Harmoniser les types des colonnes lat/lon des deux côtés
    observed_meta = observed_meta.with_columns([
        pl.col("NUM_POSTE").cast(pl.Int32),
        pl.col("lat").cast(pl.Float32),
        pl.col("lon").cast(pl.Float32),
        pl.col("altitude").cast(pl.Int32)  # altitude en entier
    ])

    observed = observed.with_columns([  # forcer ici aussi
        pl.col("NUM_POSTE").cast(pl.Int32)
    ])

    # Join sur NUM_POSTE
    observed = observed.join(observed_meta, on=["NUM_POSTE"], how="left")
    logger.info(f"n = {observed.shape[0]}")
     
    return {
        "modelised": None,
        "observed": observed,
        "column": "n_years",
        "season": season
    }

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
        n = 365 if season == "hydro" else 3*30
        if echelle == "horaire":
            n = n*24
        modelised = modelised.with_columns(
            (pl.col(col) * n).alias(col)
        )
        observed = observed.with_columns(
            (pl.col(col) * n).alias(col)
        )
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
    gev_dir_mod = Path("data/gev_m0/modelised")
    gev_dir_obs = Path("data/gev_m0/observed")
    suffix_save = "_reduce" if reduce_activate else ""

    for gev_dir in [gev_dir_mod, gev_dir_obs]:
        path_dir = Path(gev_dir) / f"{echelle}{suffix_save}" / season / "niveau_retour.parquet"
        print(path_dir)
        
        if "modelised" in str(gev_dir):
            modelised = pl.read_parquet(path_dir)
            modelised = add_metadata(modelised, scale, type='modelised')
        elif "observed" in str(gev_dir):
            # Vérifie d’abord que gev_dir (ou path_dir) existe
            if os.path.exists(path_dir):
                observed = pl.read_parquet(path_dir)
                observed = add_metadata(observed, scale, type='observed')
            else:
                observed = None       
     
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
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm, LinearSegmentedColormap, ListedColormap, to_hex
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FormatStrFormatter

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
    diff: bool = False
) -> None:
    """Trace cartes + une légende commune (SVG)
    La plage de la légende est calculée sur **l'ensemble** des valeurs (modèle
    et observations), que l'on choisisse ou non de les afficher.
    """

    # S’assurer que le répertoire existe (création si nécessaire)
    suffix_reduce = "_reduce" if reduce_activate else ""
    suffix_diff = "_diff" if diff else ""
    name_dir = f"outputs_nr10/maps/{data_type}_{col_calculate}/{echelle}{suffix_reduce}/compare_{len(titles)}/sat_{saturation_col}"
    dir_path = Path(name_dir)

    # # Si le dossier existe, on le supprime entièrement
    # if dir_path.exists() and dir_path.is_dir():
    #     shutil.rmtree(dir_path)
        
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
    # 6. Saturation des valeurs extrêmes :
    #    – seuil individuel par tableau (percentile)
    #    – saturation au‑delà du max de ces seuils
    # ------------------------------------------------------------------
    if col_calculate not in ["significant", "model"]:
        seuils = []

        # 1) seuil (percentile) propre à chaque tableau -----------------
        for gdf in (g for g in model_gdfs + obs_gdfs if g is not None):
            if val_col not in gdf.columns or gdf[val_col].empty:
                continue
            seuil = np.percentile(
                np.abs(gdf[val_col].dropna()),    # on ignore les NaN éventuels
                saturation_col                    # ex. 99 pour le 99e percentile
            )
            seuils.append(seuil)

        # Rien à faire s’il n’y a pas de seuil calculé
        if not seuils:
            return None

        # 2) seuil global = max(seuils) ---------------------------------
        seuil_global = max(seuils)

        # 3) saturation dans chaque tableau -----------------------------
        for gdf in (g for g in model_gdfs + obs_gdfs if g is not None):
            if val_col not in gdf.columns or gdf[val_col].empty:
                continue
            gdf.loc[:, val_col] = np.where(
                np.abs(gdf[val_col]) > seuil_global,
                np.sign(gdf[val_col]) * seuil_global,  # on garde le signe
                gdf[val_col]
            )

    # ------------------------------------------------------------------
    # 7. Filtrage des données significatives si demandé
    # ------------------------------------------------------------------
    # if show_signif:

    #     for idx, gdf in enumerate(model_gdfs):
    #         if gdf is not None and "significant" in gdf.columns:
    #             model_gdfs[idx] = gdf.loc[gdf["significant"] == True].copy()

    #     for idx, gdf in enumerate(obs_gdfs):
    #         if gdf is not None and "significant" in gdf.columns:
    #             obs_gdfs[idx] = gdf.loc[gdf["significant"] == True].copy()     


    # ------------------------------------------------------------------
    # 7. Si demandé : on conserve tout mais on met à 0 les non-significatifs
    # ------------------------------------------------------------------
    if show_signif:
        for idx, gdf in enumerate(model_gdfs):
            if gdf is not None and "significant" in gdf.columns and val_col in gdf.columns:
                gdf = gdf.copy()
                gdf.loc[gdf["significant"] == False, val_col] = 0
                model_gdfs[idx] = gdf

        for idx, gdf in enumerate(obs_gdfs):
            if gdf is not None and "significant" in gdf.columns and val_col in gdf.columns:
                gdf = gdf.copy()
                gdf.loc[gdf["significant"] == False, val_col] = 0
                obs_gdfs[idx] = gdf

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
    min_pt: int = 4,
    max_pt: int = 10,
    middle_pt: int = 9,
    obs_edgecolor: str = "#BFBFBF",
    obs_facecolor: Optional[str] = None,
    # Relief
    relief_path: str = "data/external/niveaux/selection_courbes_niveau_france.shp",
    relief_linewidth: float = 0.3,
    relief_color: str = "#000000",
    figsize: Tuple[int, int] = (6, 6),
    diff: bool = False,
    mesure: str = None
) -> None:
    
    if data_type == "dispo":
        min_pt, max_pt = 0.5, 3

    legend = "mm"

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

    # # *Les fonctions overlay/clip peuvent renvoyer des vues : on copie !*
    # model_gdfs = [
    #     gpd.overlay(g, deps_metro[["geometry"]], how="intersection").copy() if g is not None else None
    #     for g in model_gdfs
    # ]
    # obs_gdfs = [g.clip(mask).copy() if g is not None else None for g in obs_gdfs]

    # On CLIP seulement sur le contour national, sans overlay
    model_gdfs = [
        g.clip(mask).copy() if g is not None else None
        for g in model_gdfs
    ]
    obs_gdfs = [
        g.clip(mask).copy() if g is not None else None
        for g in obs_gdfs
    ]

    # ------------------------------------------------------------------
    # 4. Courbes de niveau
    # ------------------------------------------------------------------
    relief = (
        gpd.read_file(Path(relief_path).resolve())
        .to_crs("EPSG:2154")
        .clip(mask)
    )
    relief["geometry"] = relief.geometry.simplify(500)
    # ------------------------------------------------------------------
    # 7. Colormap & Normalisation communes
    # ------------------------------------------------------------------
    if data_type=="gev" or diff:

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
            
        elif val_col == "zTpa" and not diff:
            # Palette extraite de la barre de couleur
            colors = [
                "#f7fdf0",
                "#ddeed3",
                "#cdebc6",
                "#a8ddb6",
                "#7cccc4",
                "#4fb3d2",
                "#2b8cbe",
                "#086aad",
                "#084284",
            ]
            n_colors = len(colors)
            cmap = LinearSegmentedColormap.from_list(
                "zTpa_custom", colors, N=n_colors
            )

            all_mins = [g[val_col].min() for g in model_gdfs if g is not None]
            all_mins += [g[val_col].min() for g in obs_gdfs if g is not None]
            all_maxs = [g[val_col].max() for g in model_gdfs if g is not None]
            all_maxs += [g[val_col].max() for g in obs_gdfs if g is not None]

            max_abs = max(abs(min(all_mins)), abs(max(all_maxs)))
            vmin, vmax = 0, max_abs

            # Normalisation discrète sur [vmin, vmax] avec n_colors classes
            levels = np.linspace(vmin, vmax, n_colors + 1)
            norm = BoundaryNorm(levels, ncolors=n_colors, clip=True)

        else:
            all_mins = [g[val_col].min() for g in model_gdfs if g is not None]
            all_mins += [g[val_col].min() for g in obs_gdfs if g is not None]
            all_maxs = [g[val_col].max() for g in model_gdfs if g is not None]
            all_maxs += [g[val_col].max() for g in obs_gdfs if g is not None]

            max_abs = max(abs(min(all_mins)), abs(max(all_maxs)))
            vmin, vmax = -max_abs, max_abs

            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

            n_colors = 15 # Choix du nombre de couleurs
            # Charger tous les triplets R G B (entiers 0‑255)
            rgb = np.loadtxt("src/colors/prec_div.txt") / 255.0               # 256 × 3 → 0‑1
            # Choisir n_colors indices également espacés
            idx = np.linspace(0, rgb.shape[0] - 1, n_colors, dtype=int)
            # Les convertir en codes hexadécimaux pour from_list()
            hex_colors = [to_hex(rgb[i]) for i in idx]
            cmap = LinearSegmentedColormap.from_list("prec_div", hex_colors, N=n_colors)

    elif data_type == "dispo":
        vmin, n_colors = 0, 15
        vmax = 64 if scale == 'mm_j' else 33
        levels = np.arange(vmin, vmax + 2)    
        cmap = plt.cm.viridis
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    elif data_type=="stats":
        all_mins = [g[val_col].min() for g in model_gdfs + obs_gdfs if g is not None]
        all_maxs = [g[val_col].max() for g in model_gdfs + obs_gdfs if g is not None]
        vmin = max(all_mins)
        vmax = max(all_maxs)

        # custom_colorscale = [
        #     (0.0, "#ffffff"),  # blanc
        #     (0.1, "lightblue"),
        #     (0.3, "blue"),
        #     (0.45, "green"),
        #     (0.6, "yellow"),
        #     (0.7, "orange"),
        #     (0.8, "red"),      # rouge
        #     (1.0, "black"),    # noir
        # ]
        # # Extraction des couleurs (en ignorant les positions, la répartition sera uniforme)
        # colors = [c for _, c in custom_colorscale]

        # # Création d'un colormap à partir de ces couleurs, avec N=15 couleurs distinctes
        # cmap = LinearSegmentedColormap.from_list("custom_discrete", colors, N=15)

        n_colors = 15 # Choix du nombre de couleurs
        # Charger tous les triplets R G B (entiers 0‑255)
        rgb = np.loadtxt("src/colors/prec_seq.txt") / 255.0               # 256 × 3 → 0‑1
        # Choisir n_colors indices également espacés
        idx = np.linspace(0, rgb.shape[0] - 1, n_colors, dtype=int)
        # Les convertir en codes hexadécimaux pour from_list()
        hex_colors = [to_hex(rgb[i]) for i in idx]
        cmap = LinearSegmentedColormap.from_list("prec_seq", hex_colors, N=n_colors)

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
            #deps_metro.boundary.plot(ax=ax, edgecolor="#AAAAAA", linewidth=0.2, zorder=0)

            # 2. contour national épaissi
            #    mask est déjà défini plus haut comme l'union de tous les départements
            #gpd.GeoSeries(mask).boundary.plot(
            #     ax=ax,
            #     edgecolor="black" ,    # couleur du contour national
            #     linewidth=0.7,         # épaisseur plus élevée que 0.2
            #     zorder=1               # juste au-dessus des départements
            # )

            if mask.geom_type == "MultiPolygon":
                polys = list(mask.geoms)
            elif mask.geom_type == "Polygon":
                polys = [mask]
            else:
                # au cas où, on met tout dans une liste
                polys = [mask]

            # extraire uniquement les contours extérieurs
            exteriors = [poly.exterior for poly in polys]

            coast = gpd.GeoSeries(exteriors, crs=deps_metro.crs)

            # optionnel : supprimer les petits bouts de ligne (artefacts)
            coast = coast[coast.length > 2_000]  # 2 km, à ajuster

            coast.plot(
                ax=ax,
                edgecolor="black",
                linewidth=0.6,
                zorder=1,
            )


            # — Modèle ------------------------------------------------------
            if show_mod and gdf_m is not None:
                gdf_m.plot(ax=ax, column=val_col, cmap=cmap, norm=norm, linewidth=0, zorder=1)

            # — Observations -----------------------------------------------
            if show_obs and gdf_o is not None and not gdf_o.empty:

                # Pour "significant" et "model", on force une taille fixe moyenne
                if val_col in ["significant", "model"]:
                    # taille linéaire = moyenne de min_pt et max_pt
                    mean_pt = middle_pt / 2
                    # markersize attend la surface (pt^2)
                    gdf_o.loc[:, "_size_pt2"] = mean_pt**2
                    
                else:
                    # abs_vals = gdf_o["_val_raw"].abs()
                    # abs_max = abs_vals.max()
                    # raw_size = ((abs_vals / abs_max) * (max_pt - min_pt) + min_pt) ** 2
                    # seuil_size = np.percentile(abs_vals, saturation_size)
                    # sizes = raw_size.copy()
                    # sizes[abs_vals > seuil_size] = max_pt ** 2

                    # gdf_o.loc[:, "_size_pt2"] = sizes

                    # 1) valeurs ABSOLUES **après** saturation (valeurs de la palette)
                    abs_vals = gdf_o[val_col].abs()

                    # 2) mise à l’échelle linéaire [min_pt ; max_pt] puis surface (pt²)
                    abs_max  = abs_vals.max() or 1                              # évite /0
                    sizes = ((abs_vals / abs_max) * (max_pt - min_pt) + min_pt) ** 2

                    # 3) éventuelle saturation supplémentaire sur la taille
                    if saturation_size < 100:                                   # 100 → aucune
                        seuil_size = np.percentile(abs_vals, saturation_size)
                        sizes[abs_vals > seuil_size] = max_pt ** 2

                    gdf_o.loc[:, "_size_pt2"] = sizes


                # split 0 vs non-0
                gdf_zero = gdf_o[gdf_o[val_col] == 0]
                gdf_nonzero = gdf_o[gdf_o[val_col] != 0]

                kw_zero = dict(
                    markersize="_size_pt2",
                    marker="o",
                    edgecolor=obs_edgecolor,
                    linewidth=0.1,
                )

                kw_nonzero = dict(
                    markersize="_size_pt2",
                    marker="o",
                    edgecolor="face",
                    linewidth=0.1,
                )

                # 1. points à 0 (en-dessous)
                if not gdf_zero.empty:
                    if obs_facecolor is None:
                        gdf_zero.plot(ax=ax, column=val_col, cmap=cmap, norm=norm,
                                    zorder=2, **kw_zero)
                    else:
                        gdf_zero.plot(ax=ax, facecolor=obs_facecolor,
                                    zorder=2, **kw_zero)

                # 2. points non nuls (par-dessus)
                if not gdf_nonzero.empty:
                    if obs_facecolor is None:
                        gdf_nonzero.plot(ax=ax, column=val_col, cmap=cmap, norm=norm,
                                        zorder=3, **kw_nonzero)
                    else:
                        gdf_nonzero.plot(ax=ax, facecolor=obs_facecolor,
                                        zorder=3, **kw_nonzero)


            # — Relief ------------------------------------------------------
            relief.plot(ax=ax, color=relief_color, linewidth=relief_linewidth, alpha=0.8, zorder=5)

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
                    if mesure is not None:
                        cb.ax.set_ylabel(f"{legend}", rotation=90, fontsize=10)
                    # Forcer un chiffre après la virgule
                    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


            suffix_obs = "obs" if show_obs else ""
            suffix_mod = "mod" if show_mod else ""
            suffix_signif = "_signif" if show_signif else ""
            suffix_diff = "_diff" if diff else ""
            name_file = f"{suffix_mod}{suffix_obs}{suffix_signif}_{mode}{suffix_diff}"

            subdir = dir_path / title.lower() 
            subdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(subdir / f"{name_file}.svg", format="svg", bbox_inches="tight", pad_inches=0)
            fig.savefig(subdir / f"{name_file}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            logger.info(subdir / f"{name_file}.svg")

    # ------------------------------------------------------------------
    # 9. Légende seule ---------------------------------------------------
    # ------------------------------------------------------------------
    if diff:
        mpl.rcParams.update({"font.size": 30})
    else:
        mpl.rcParams.update({"font.size": 22})
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
        if diff:
            cb2.ax.set_ylabel(f"{legend}", rotation=90, fontsize=30)
        else:
            cb2.ax.set_ylabel(f"{legend}", rotation=90, fontsize=22)
        # Forcer un chiffre après la virgule
        cb2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    suffix_signif = "_signif" if show_signif else ""
    suffix_diff = "_diff" if diff else ""
    name_legend = f"legend{suffix_signif}{suffix_diff}"
    
    fig2.savefig(dir_path / f"{name_legend}.svg", format="svg", bbox_inches="tight", pad_inches=0)
    fig2.savefig(dir_path / f"{name_legend}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig2)

    # --- 9bis. Légende horizontale ---------------------------------------
    mpl.rcParams.update({"font.size": 22})
    figh = plt.figure(figsize=(figsize[0]*1.4, 1.2))
    axh = figh.add_axes([0.08, 0.45, 0.84, 0.35])  # [left, bottom, width, height]

    smh = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    smh.set_array([])

    if isinstance(norm, mpl.colors.BoundaryNorm) and val_col in ["significant", "model"]:
        ticks = np.arange(len(all_cat))
        cbh = ColorbarBase(
            axh,
            cmap=cmap,
            norm=norm,
            boundaries=np.arange(len(all_cat)+1) - 0.5,
            ticks=ticks,
            spacing="proportional",
            orientation="horizontal"
        )
        display_labels = [TRANSFO[code] for code in all_cat]
        cbh.ax.set_xticklabels(display_labels, rotation=0, ha="center")
    else:
        cbh = figh.colorbar(
            smh,
            cax=axh,
            spacing="proportional",
            orientation="horizontal"
        )
        # if mesure is not None:
        #     cbh.ax.set_xlabel(f"{legend}", labelpad=6, fontsize=20)
        cbh.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # style propre
    cbh.outline.set_linewidth(0.8)
    cbh.ax.tick_params(axis="x", labelsize=12, length=4, width=0.8, direction="out")

    suffix_signif = "_signif" if show_signif else ""
    suffix_diff = "_diff" if diff else ""
    name_legend_h = f"legend_horiz{suffix_signif}{suffix_diff}"

    figh.savefig(dir_path / f"{name_legend_h}.svg", format="svg", bbox_inches="tight", pad_inches=0)
    figh.savefig(dir_path / f"{name_legend_h}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(figh)


from matplotlib.ticker import MaxNLocator, MultipleLocator

def generate_hist(res: dict, echelle: str, season: str, reduce_activate: bool) -> None:
    obs_df = res.get("observed")
    if obs_df is None or obs_df.height == 0:
        return

    years = obs_df["n_years"].to_numpy()
    if years.size == 0:
        return

    y_max = int(np.ceil(years.max()))
    bins = np.arange(-0.5, y_max + 1.5, 1)

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=150)
    ax.hist(years, bins=bins, cumulative=True, density=False,
            edgecolor="black", linewidth=0.6, facecolor="#E9ECEF", alpha=1.0)

    ax.set_xlim(-0.5, y_max + 0.5)
    ax.set_xlabel("Series length (years)", fontsize=7)
    ax.set_ylabel("Cumulative number of stations", fontsize=7) if echelle=="quotidien" else ax.set_ylabel("")
    ax.grid(axis="y", which="major", linewidth=0.3, color="#BFBFBF")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="both", labelsize=6, length=3, width=0.6, direction="out")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(0.8)
    fig.tight_layout()

    suffix = "_reduce" if reduce_activate else ""
    out_dir = Path(f"outputs_nr10/hist/dispo/{echelle}{suffix}/{season.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "hist_len_years_cum.svg", format="svg", bbox_inches="tight", pad_inches=0)
    fig.savefig(out_dir / "hist_len_years_cum.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    print(f"{out_dir}/hist_len_years_cum.svg")
    plt.close(fig)






def generate_scatter(
    datasets: dict[str, pd.DataFrame],
    dir_path: Path,
    col_calculate: str,
    echelle: str,
    show_signif: bool,
    scale: str
):
    global logger
    if col_calculate not in ["significant", "model"]:

        metrics = []
        datasets_diff = []

        df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")

        for d in datasets:
            mod = d.get("modelised")    # pl.DataFrame
            obs = d.get("observed")     # pl.DataFrame
            col = d.get("column")
            season = d.get("season")

            if show_signif:
                # mod = mod.filter(pl.col("significant") == True)
                obs = obs.filter(pl.col("significant") == True)
                # toujours se baser sur les stations
                # et calculer avec AROME correspondant même si non significatif

            obs_vs_mod = match_and_compare(obs, mod, col, df_obs_vs_mod)

            obs_vs_mod_diff = obs_vs_mod.select([
                pl.Series("NUM_POSTE", range(1, obs_vs_mod.height + 1)),
                pl.col("lat"),
                pl.col("lon"),
                (pl.col("AROME") - pl.col("Station")).alias(col)
            ])

            res = {
                "modelised": None,
                "observed": obs_vs_mod_diff,
                "column": col,
                "season": season
            }

            datasets_diff.append(res)

            if obs_vs_mod.is_empty():
                logger.warning(f"Aucune donnée après filtrage pour {echelle} {season} - {col_calculate} - signif {show_signif}")
                continue

            # Légende unités
            if col_calculate == "numday":
                legend = "days"
            elif col_calculate == "mean":
                legend = "mm/year"
            elif col_calculate == "z_T_p":
                legend = "%"
            else:
                legend = scale.replace('_', '/')

            # ----- SCATTER -----
            season_dir = dir_path / season
            season_dir.mkdir(parents=True, exist_ok=True)

            x = obs_vs_mod["AROME"].to_numpy()
            y = obs_vs_mod["Station"].to_numpy()

            max_lim = np.max([x.max(), y.max()])

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                x, y,
                label="Données",
                alpha=0.5,
                s=20,
                facecolor="black",
                edgecolor="white",
                linewidth=0.5
            )
            ax.plot([0, max_lim], [0, max_lim], linewidth=1, color="red", label="y = x")
            ax.set_xlabel(f"AROME ({legend})")
            ax.set_ylabel(f"Stations ({legend})")
            ax.set_aspect("equal")
            ax.set_xlim(0, max_lim)
            ax.set_ylim(0, max_lim)
            fig.savefig(season_dir / "scatter.svg", format="svg", bbox_inches="tight", pad_inches=0)
            fig.savefig(season_dir / "scatter.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            logger.info(season_dir / "scatter.svg")

            # ----- METRICS -----
            me = np.mean(x - y)
            corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
            n = len(x)
            mean_mod = obs_vs_mod.select(pl.col("AROME").mean()).item()
            mean_obs = obs_vs_mod.select(pl.col("Station").mean()).item()
            delta = me / np.mean([mean_mod, mean_obs])*100

            metrics.append({
                "echelle": echelle,
                "col_calculate": f"{col_calculate}",
                "season": season,
                "n": n,
                "r": corr,
                "me": me,
                "delta": delta
            })


        suffix = "_signif" if show_signif else ""
        if metrics:
            pd.DataFrame(metrics).to_csv(dir_path / f"metrics{suffix}.csv", index=False)
            logger.info(f"{dir_path}/metrics{suffix}.csv")
        else:
            logger.warning(f"Aucun résultat significatif, fichier metrics{suffix}.csv non créé")

    return datasets_diff

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

            if data_type == "dispo":
                res = import_data_dispo(
                    season=s,
                    echelle=e,
                    reduce_activate=reduce_activate,
                    config_obs=config_obs,
                    scale=scale
                )

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
                
            if col_calculate in ["z_T_p", "model"]:
                SIGNIFICANT_SHOW = [True] # On choisi d'afficher ou non les points significatifs 
            else:
                SIGNIFICANT_SHOW = [False]

            logger.info(f"Echelle {e} - Saison {s} données générées")
            datasets.append(res)    # On range le résultat dans la bonne liste
     
        for signif in SIGNIFICANT_SHOW:
            dir_path, val_col, titles, model_gdfs, obs_gdfs = calculate_data_maps(
                datasets,
                echelle=e,
                data_type=data_type,
                col_calculate=col_calculate,
                reduce_activate=reduce_activate,
                show_signif=signif,
                titles=[s.upper() for s in season],  # titres des 4 sous-cartes
                saturation_col=sat
            )

            # for show in [True, False]:
            #     generate_maps(
            #         dir_path=dir_path,
            #         model_gdfs=model_gdfs,
            #         obs_gdfs=obs_gdfs,
            #         data_type=data_type,
            #         titles=titles,
            #         val_col=val_col,
            #         show_mod=show,
            #         show_obs=not show,
            #         show_signif=signif,
            #         col_calculate=col_calculate,
            #         scale=scale
            #     )

        if data_type == "dispo":
            generate_hist(res, echelle=e, season=s, reduce_activate=reduce_activate)

        if data_type != "dispo" and col_calculate not in ["significant", "model"]:
            for signif in SIGNIFICANT_SHOW:

                datasets_diff = generate_scatter(
                    datasets=datasets,
                    dir_path=dir_path,
                    col_calculate=col_calculate,
                    echelle=e,
                    show_signif=signif,
                    scale=scale
                )
                
                dir_path, val_col, titles, model_gdfs, obs_gdfs = calculate_data_maps(
                    datasets_diff,
                    echelle=e,
                    data_type=data_type,
                    col_calculate=col_calculate,
                    reduce_activate=reduce_activate,
                    show_signif=signif,
                    titles=[s.upper() for s in season],  # titres des 4 sous-cartes
                    saturation_col=sat,
                    diff=True
                )

                # generate_maps(
                #     dir_path=dir_path,
                #     model_gdfs=model_gdfs,
                #     obs_gdfs=obs_gdfs,
                #     data_type=data_type,
                #     titles=titles,
                #     val_col=val_col,
                #     show_mod=False,
                #     show_obs=not False,
                #     show_signif=signif,
                #     col_calculate=col_calculate,
                #     scale=scale,
                #     diff=True,
                #     mesure=mesure
                # )


def str2bool(v):
    if v == "True":
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de génération des représentation")
    parser.add_argument("--data_type", choices=["dispo", "stats", "gev"])
    parser.add_argument("--col_calculate", choices=["n_years", "numday", "mean", "mean-max", "z_T_p", "zTpa", "significant", "model"], default=None)
    parser.add_argument("--echelle", choices=["horaire", "quotidien", "horaire_reduce"], nargs='+', default=["horaire"])
    parser.add_argument("--season", type=str, nargs='+', default=["son"])
    parser.add_argument("--reduce_activate", type=str2bool, default=False)
    parser.add_argument("--sat", type=float, default=100)

    args = parser.parse_args()  
    main(args)
