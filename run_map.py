from __future__ import annotations

from typing import Sequence, Mapping, Tuple
from pathlib import Path

import polars as pl
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm, Normalize, LinearSegmentedColormap

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
from shapely.geometry import box

from app.pipelines.import_quarto import pipeline_data_gev_quarto

import yaml
config_path = Path("app/config/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def plot_four_polars_grids(
    datasets: Sequence[Mapping[str, pl.DataFrame]],
    *,
    echelle: str,
    titles: Sequence[str] | None = None,  # ex: ['SON', 'DJF', 'MAM', 'JJA']
    lat_col: str = "lat",
    lon_col: str = "lon",
    val_col: str = "ERROR",
    # Observations
    obs_key: str = "observed_show",
    show_obs: bool = True,  # n'influence plus que l'affichage
    min_pt: int=6,
    max_pt: int=11,
    obs_edgecolor: str = "#000000",
    obs_facecolor: Optional[str] = None,
    # Grid
    side_km: float = 2.5,
    # Colormap
    cmap_bool: Tuple[str, str] = ("#f0f0f0", "#d73027"),
    # Relief
    relief_path: str = "data/external/niveaux/selection_courbes_niveau_france.shp",
    relief_linewidth: float = 0.3,
    relief_color: str = "#666666",
    figsize: Tuple[int, int] = (6, 6),
    show_mod: bool = True,
    show_signif: bool = False,  # filtre significatif sur obs
    numero: int=1,
    saturation_col: int=99,
    saturation_size: int=99
):
    """Trace quatre cartes polaires + une légende commune (SVG).

    La plage de la légende est calculée sur **l'ensemble** des valeurs (modèle
    et observations), que l'on choisisse ou non de les afficher.
    """

    # ------------------------------------------------------------------
    # Préparation des GeoDataFrames
    # ------------------------------------------------------------------
    model_gdfs, obs_gdfs = [], []
    for d in datasets:
        # -- Modèle
        df_m = d.get("modelised_show")
        if df_m is None:
            model_gdfs.append(None)


        else:
            if not isinstance(df_m, pl.DataFrame):
                raise TypeError("'modelised_show' doit être un pl.DataFrame.")
            gdf_m = gpd.GeoDataFrame(
                df_m.to_pandas(),  # Convertissez df_m en DataFrame pandas
                geometry=gpd.points_from_xy(df_m[lon_col].to_list(), df_m[lat_col].to_list()),
                crs="EPSG:4326",
            ).to_crs("EPSG:2154")
            model_gdfs.append(gdf_m)  # Ajoutez gdf_m à model_gdfs

        # -- Observations (toujours chargées pour la légende)
        if obs_key in d:
            df_o = d[obs_key]
            if not isinstance(df_o, pl.DataFrame):
                raise TypeError("'observed_show' doit être un pl.DataFrame.")
            if "significant" in df_o.columns and show_signif:
                df_o = df_o.filter(pl.col("significant"))
            obs_gdfs.append(
                gpd.GeoDataFrame(
                    df_o.to_pandas(),
                    geometry=gpd.points_from_xy(df_o[lon_col].to_list(), df_o[lat_col].to_list()),
                    crs="EPSG:4326",
                ).to_crs("EPSG:2154")
                if df_o.height > 0
                else None
            )
        else:
            obs_gdfs.append(None)

    # Titres par défaut
    if titles is None:
        titles = [str(i) for i in range(1, len(model_gdfs) + 1)]

    # ------------------------------------------------------------------
    # Grille carrée autour des points modélisés
    # ------------------------------------------------------------------
    half = side_km * 500  # m
    for i, gdf in enumerate(model_gdfs):
        if gdf is not None:
            gdf = gdf.copy()
            gdf.loc[:, "geometry"] = gdf.geometry.apply(
                lambda p: box(p.x - half, p.y - half, p.x + half, p.y + half)
            )
            model_gdfs[i] = gdf

    # ------------------------------------------------------------------
    # Masque France métropolitaine
    # ------------------------------------------------------------------
    deps = (
        gpd.read_file("https://france-geojson.gregoiredavid.fr/repo/departements.geojson")
        .to_crs("EPSG:2154")
    )
    deps_metro = deps[~deps["code"].isin(["2A", "2B"])].copy()
    deps_metro["geometry"] = deps_metro.geometry.simplify(500)
    mask = deps_metro.union_all()

    # Correction : overlay et clip peuvent retourner des vues, donc copie explicite
    model_gdfs = [gpd.overlay(g, deps_metro[["geometry"]], how="intersection").copy() if g is not None else None for g in model_gdfs]
    obs_gdfs = [g.clip(mask).copy() if isinstance(g, gpd.GeoDataFrame) else None for g in obs_gdfs]

    # ------------------------------------------------------------------
    # Relief
    # ------------------------------------------------------------------
    relief = gpd.read_file(Path(relief_path).resolve()).to_crs("EPSG:2154").clip(mask)

    # ------------------------------------------------------------------
    # Saturation des valeurs extrêmes (1 % sup.)
    # ------------------------------------------------------------------

    # 1) Avant la saturation, mémoriser la valeur originale
    for gdf in model_gdfs + [g for g in obs_gdfs if g is not None]:
        gdf = gdf.copy()
        gdf["_val_raw"] = gdf[val_col].copy()         # ← copie intacte

    all_values: list[float] = []
    for gdf in model_gdfs:
        all_values.extend(gdf[val_col].to_numpy())
    for gdf in obs_gdfs:
        if gdf is not None:
            all_values.extend(gdf[val_col].to_numpy())

    is_bool = model_gdfs[0][val_col].dtype == bool
    if all_values and not is_bool:
        # Seuil couleur pour toutes les valeurs, modèle + obs
        seuil = np.percentile(np.abs(all_values), saturation_col)
        for gdf in model_gdfs + [g for g in obs_gdfs if g is not None]:
            gdf = gdf.copy()
            gdf.loc[:, val_col] = np.where(np.abs(gdf[val_col]) > seuil,
                                        np.sign(gdf[val_col]) * seuil,
                                        gdf[val_col])

    # ------------------------------------------------------------------
    # Colormap & Norm communes
    # ------------------------------------------------------------------
    if is_bool:
        cmap = mpl.colors.ListedColormap(list(cmap_bool))
        norm = BoundaryNorm([-0.5, 0.5, 1.5], ncolors=2)
        for g in model_gdfs + [g for g in obs_gdfs if g is not None]:
            g = g.copy()
            g.loc[:, "_val_int"] = g[val_col].astype(int)
        col = "_val_int"
    else:
        all_mins = [g[val_col].min() for g in model_gdfs]
        all_mins += [g[val_col].min() for g in obs_gdfs if g is not None]
        all_maxs = [g[val_col].max() for g in model_gdfs]
        all_maxs += [g[val_col].max() for g in obs_gdfs if g is not None]

        vmin_raw = float(min(all_mins))
        vmax_raw = float(max(all_maxs))
        # rayon symétrique
        max_abs = max(abs(vmin_raw), abs(vmax_raw))
        vmin, vmax = -max_abs, max_abs

        # centre au zéro
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = LinearSegmentedColormap.from_list("bwr_custom", ["red", "#ffffff", "blue"], N=150)
        col = val_col

    # ------------------------------------------------------------------
    # Export cartes
    # ------------------------------------------------------------------
    Path("presentation_files/map").mkdir(parents=True, exist_ok=True)
    modes = {"rast": 0, "norast": 1}

    for title, gdf_m, gdf_o in zip(titles, model_gdfs, obs_gdfs):
        for mode, z in modes.items():
            fig, ax = plt.subplots(figsize=figsize)

            deps_metro.boundary.plot(ax=ax, edgecolor="#AAAAAA", linewidth=0.2, zorder=0)

            if show_mod:
                gdf_m.plot(ax=ax, column=col, cmap=cmap, norm=norm, linewidth=0, zorder=1)

            if show_obs and gdf_o is not None and not gdf_o.empty:
                # Vérifier que la colonne _val_raw existe et lever une erreur sinon
                if "_val_raw" not in gdf_o.columns:
                    raise RuntimeError("La colonne _val_raw devrait exister à ce stade !")

                # ------------------------------------------------------------------
                # Échelle des diamètres de points
                # ------------------------------------------------------------------
                # valeurs brutes abs et max global (pour le mapping linéaire)
                abs_vals = gdf_o["_val_raw"].abs()
                abs_max  = abs_vals.max()
                # taille brute non-saturée
                raw_size = ((abs_vals / abs_max) * (max_pt - min_pt) + min_pt) ** 2
                # masque des points à saturer
                # Seuil taille uniquement les obs
                seuil_size = np.percentile(abs_vals, saturation_size)
                # copie et saturation : tous les points > seuil obtiennent max_pt**2
                sizes = raw_size.copy()
                sizes[abs_vals > seuil_size] = max_pt ** 2
                # on enregistre
                gdf_o = gdf_o.copy()
                gdf_o.loc[:, "_size_pt2"] = sizes

                kw = dict(
                    markersize="_size_pt2",
                    marker="o",
                    edgecolor=obs_edgecolor,
                    linewidth=0.1,
                    zorder=3,
                )

                if obs_facecolor is None:
                    gdf_o.plot(ax=ax, column=col, cmap=cmap, norm=norm, **kw)
                else:
                    gdf_o.plot(ax=ax, facecolor=obs_facecolor, **kw)

            relief.plot(ax=ax, color=relief_color, linewidth=relief_linewidth, alpha=0.8, zorder=2)

            ax.set_axis_off()
            ax.set_rasterization_zorder(z)

            if mode == "rast":
                sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
                if is_bool:
                    cbar.set_ticks([0, 1])
                    cbar.set_ticklabels(["Non", "Oui"])

            suffix_obs = "_obs" if show_obs else ""
            suffix_mod = "_mod" if show_mod else ""
            suffix_signif = "_signif" if show_signif else ""
            name = f"{echelle}_{title}_{mode}{suffix_mod}{suffix_obs}{suffix_signif}"

            fig.savefig(f"presentation_files/map/{name}.svg", format="svg", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"presentation_files/map/{name}.svg")

    # ------------------------------------------------------------------
    # Légende seule
    # ------------------------------------------------------------------
    mpl.rcParams.update({'font.size': 16})
    fig2 = plt.figure(figsize=(1, figsize[1]*2))
    ax2 = fig2.add_axes([0.25, 0.05, 0.5, 0.9])
    sm2 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cb2 = fig2.colorbar(sm2, cax=ax2)
    if is_bool:
        cb2.set_ticks([0, 1])
        cb2.set_ticklabels(["Non", "Oui"])

    fig2.savefig(f"presentation_files/map/{echelle}_legend_{numero}.svg", format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    

##################################################" QUOTIDIEN"
echelles = ["quotidien"] 

seasons  = ["son", "djf", "mam", "jja"]
datasets = {e: [] for e in echelles}

for e in echelles:
    for s in seasons:
        res = pipeline_data_gev_quarto(
            config=config,
            model_name="niveau_retour",
            echelle=e,
            season_choice=s,
            T_choice=10,
            par_X_annees=10,
            min_year=config["years"]["min"],
            max_year=config["years"]["max"],
        )
        print(f"saison {s} données générées")
        datasets[e].append(res)    # On range le résultat dans la bonne liste

for show in [True, False]:
    for e in echelles:
        plot_four_polars_grids(
            datasets[e],
            echelle=e,
            val_col="z_T_p",          # nom de la colonne à colorer / symboliser
            titles=[s.upper() for s in seasons],  # titres des 4 sous-cartes
            show_mod=show,
            show_obs=not show,
            numero=1
        )


seasons  = ["son_bis", "nod", "sep", "oct", "nov", "dec"]
datasets = {e: [] for e in echelles}

for e in echelles:
    for s in seasons:
        res = pipeline_data_gev_quarto(
            config=config,
            model_name="niveau_retour",
            echelle=e,
            season_choice=s if s != "son_bis" else "son",
            T_choice=10,
            par_X_annees=10,
            min_year=config["years"]["min"],
            max_year=config["years"]["max"],
        )
        print(f"saison {s} données générées")
        datasets[e].append(res)    # On range le résultat dans la bonne liste

for show in [True, False]:
    for e in echelles:
        plot_four_polars_grids(
            datasets[e],
            echelle=e,
            val_col="z_T_p",          # nom de la colonne à colorer / symboliser
            titles=[s.upper() for s in seasons],  # titres des 4 sous-cartes
            show_mod=show,
            show_obs=not show,
            numero=2
        )


######################################## HORAIRE
# echelles = ["horaire"] 
# seasons  = [
#     "son", "djf", "mam", "jja",
#     "jan",
#     "fev",
#     "mar",
#     "avr",
#     "mai",
#     "jui",
#     "juill",
#     "aou",
#     "sep",
#     "oct",
#     "nov",
#     "dec"
# ]

# datasets = {e: [] for e in echelles}

# for e in echelles:
#     for s in seasons:
#         res = pipeline_data_gev_quarto(
#             config=config,
#             model_name="niveau_retour",
#             echelle=e,
#             season_choice=s,
#             T_choice=10,
#             par_X_annees=10,
#             min_year=config["years"]["min"],
#             max_year=config["years"]["max"],
#         )
#         print(f"saison {s} données générées")
#         datasets[e].append(res)    # On range le résultat dans la bonne liste

# for show in [True, False]:
#     for e in echelles:
#         plot_four_polars_grids(
#             datasets[e],
#             echelle=e,
#             val_col="z_T_p",          # nom de la colonne à colorer / symboliser
#             titles=[s.upper() for s in seasons],  # titres des 4 sous-cartes
#             show_mod=show,
#             show_obs=not show,
#             numero=1
#         )