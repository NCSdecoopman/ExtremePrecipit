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
    
    # Set data loading parameters
    if reduce_activate:
        _, min_year, max_year, _ = years_to_load("reduce", season, input_dir)
    else:
        _, min_year, max_year, _ = years_to_load(echelle, season, input_dir)
    cols = ["NUM_POSTE", "nan_ratio"]

    observed_load = load_data(input_dir, season, echelle, cols, min_year, max_year)

    # Filter by NaN ratio and aggregate unique years
    nan_limit = 0.10
    observed = (
        observed_load.filter(pl.col("nan_ratio") <= nan_limit)
        .group_by("NUM_POSTE")
        .agg(pl.col("year").n_unique().alias("n_years"))
    )
    
    # Load station metadata
    observed_meta = pl.read_csv(f"data/metadonnees/observed/postes_{echelle}.csv")
    observed_meta = observed_meta.with_columns([
        pl.col("NUM_POSTE").cast(pl.Int32),
        pl.col("lat").cast(pl.Float32),
        pl.col("lon").cast(pl.Float32),
        pl.col("altitude").cast(pl.Int32)
    ])

    observed = observed.with_columns([
        pl.col("NUM_POSTE").cast(pl.Int32)
    ])

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
        # Set data loading parameters
        if reduce_activate:
            _, min_year, max_year, len_serie = years_to_load("reduce", season, input_dir)
        else:
            _, min_year, max_year, len_serie = years_to_load(echelle, season, input_dir)
        cols = ["NUM_POSTE", mesure, "nan_ratio"]

        df = load_data(input_dir, season, echelle, cols, min_year, max_year)

        # Filter stations by max NaN ratio
        df = cleaning_data_observed(df, echelle, len_serie)
        
        # Handle null values
        df = df.drop_nulls(subset=[mesure])

        # Calculate statistics per point
        df = compute_statistic_per_point(df, col_calculate)

        # Add coordinates and altitude metadata
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
    gev_dir_mod = config_mod["gev"]["path"]["outputdir"]
    gev_dir_obs = config_obs["gev"]["path"]["outputdir"]
    suffix_save = "_reduce" if reduce_activate else ""

    for gev_dir in [gev_dir_mod, gev_dir_obs]:
        path_dir = Path(gev_dir) / f"{echelle}{suffix_save}" / season / "niveau_retour.parquet"
        
        if "modelised" in str(gev_dir):
            modelised = pl.read_parquet(path_dir)
            modelised = add_metadata(modelised, scale, type='modelised')
        elif "observed" in str(gev_dir):
            # Verify if path exists before reading
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
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

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
    """Plot maps and common SVG legend.
    The legend range is calculated over all values (model and observations).
    """

    # Ensure output directory exists
    suffix_reduce = "_reduce" if reduce_activate else ""
    suffix_diff = "_diff" if diff else ""
    name_dir = f"outputs/maps/{data_type}_{col_calculate}/{echelle}{suffix_reduce}/compare_{len(titles)}/sat_{saturation_col}"
    dir_path = Path(name_dir)

    # # Si le dossier existe, on le supprime entièrement
    # if dir_path.exists() and dir_path.is_dir():
    #     shutil.rmtree(dir_path)
        
    dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Convert Polars DataFrames to GeoDataFrames
    model_gdfs: list[Optional[gpd.GeoDataFrame]] = []
    obs_gdfs: list[Optional[gpd.GeoDataFrame]] = []

    for d in datasets:
        val_col = d.get("column")

        # Model
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

        # Observations
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



    # Default titles
    if titles is None:
        titles = [str(i) for i in range(1, len(model_gdfs) + 1)]

    # 2. Add square grid around model points
    half = side_km * 500  # meters
    for gdf in (g for g in model_gdfs if g is not None):
        gdf.loc[:, "geometry"] = gdf.geometry.apply(
            lambda p: box(p.x - half, p.y - half, p.x + half, p.y + half)
        )

    # 5. Add _val_raw column (intact copy of values)
    for gdf in (g for g in model_gdfs if g is not None):
        gdf.loc[:, "_val_raw"] = gdf[val_col]
    for gdf in (g for g in obs_gdfs if g is not None):
        gdf.loc[:, "_val_raw"] = gdf[val_col]

    # 6. Saturate extreme values using percentile thresholds
    if col_calculate not in ["significant", "model"]:
        seuils = []

        # 1) Calculate percentile threshold for each table
        for gdf in (g for g in model_gdfs + obs_gdfs if g is not None):
            if val_col not in gdf.columns or gdf[val_col].empty:
                continue
            vals = gdf[val_col].dropna()
            if show_signif and "significant" in gdf.columns:
                vals = gdf.loc[gdf["significant"], val_col].dropna()
            if vals.empty:
                continue
            seuil = np.percentile(np.abs(vals), saturation_col)
            seuils.append(seuil)

        if not seuils:
            return None

        # 2) Global threshold = max(seuils)
        seuil_global = max(seuils)

        # 3) Apply saturation
        for gdf in (g for g in model_gdfs + obs_gdfs if g is not None):
            if val_col not in gdf.columns or gdf[val_col].empty:
                continue
            gdf.loc[:, val_col] = np.where(
                np.abs(gdf[val_col]) > seuil_global,
                np.sign(gdf[val_col]) * seuil_global,
                gdf[val_col]
            )

    # ------------------------------------------------------------------
    # 7. Filter significant data if requested (original commented out block)
    # ------------------------------------------------------------------
    # if show_signif:

    #     for idx, gdf in enumerate(model_gdfs):
    #         if gdf is not None and "significant" in gdf.columns:
    #             model_gdfs[idx] = gdf.loc[gdf["significant"] == True].copy()

    #     for idx, gdf in enumerate(obs_gdfs):
    #         if gdf is not None and "significant" in gdf.columns:
    #             obs_gdfs[idx] = gdf.loc[gdf["significant"] == True].copy()     


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
    show_obs: bool = True,
    show_signif: bool = False,
    saturation_size: int = 100,

    # Obs points
    min_pt: int = 4,
    max_pt: int = 10,
    middle_pt: int = 9,
    obs_edgecolor: str = "#BFBFBF",
    obs_facecolor: Optional[str] = None,

    relief_path: str = "data/external/niveaux/selection_courbes_niveau_france.shp",
    relief_linewidth: float = 0.3,
    relief_color: str = "#000000",
    figsize: Tuple[int, int] = (6, 6),
    diff: bool = False,
    rdiff: bool = False,
    mesure: str = None
) -> None:
    
    if data_type == "dispo":
        min_pt, max_pt = 0.5, 3

    if val_col not in ["significant", "model"]:
        if rdiff:
            legend = "%"
        elif col_calculate == "numday":
            legend = "days"
        elif col_calculate == "mean":
            legend = "mm/year"
        elif col_calculate == "z_T_p":
            legend = "%"
        else:
            legend = scale.replace('_', '/')
            # Normalize units to English
            legend = legend.replace("mm/j", "mm/d")
    else:
        legend = ""

    # 3. Metropolitan France mask
    deps = (
        gpd.read_file("https://france-geojson.gregoiredavid.fr/repo/departements.geojson")
        .to_crs("EPSG:2154")
    )
    deps_metro = deps[~deps["code"].isin(["2A", "2B"])].copy()
    deps_metro["geometry"] = deps_metro.geometry.simplify(500)
    mask = deps_metro.union_all()

    # Clip to national boundary
    model_gdfs = [
        g.clip(mask).copy() if g is not None else None
        for g in model_gdfs
    ]
    obs_gdfs = [
        g.clip(mask).copy() if g is not None else None
        for g in obs_gdfs
    ]

    # 4. Contour lines
    relief = (
        gpd.read_file(Path(relief_path).resolve())
        .to_crs("EPSG:2154")
        .clip(mask)
    )
    relief["geometry"] = relief.geometry.simplify(500)
    # 7. Shared colormap and normalization
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
                "#D55E00",  # rust
                "#0072B2",  # dark blue
                "#8B4513",  # brown
                "#000000",  # black
                "#009E73",  # green
                "#F0E442"   # yellow
            ]

            cmap = ListedColormap(palette)
            # Center integer categories on colors
            norm = BoundaryNorm(np.arange(len(all_cat) + 1) - 0.5,
                                ncolors=len(all_cat))

        else:
            def _vals_for_norm(g):
                if g is None or val_col not in g.columns or g[val_col].empty:
                    return None
                if show_signif and "significant" in g.columns:
                    v = g.loc[g["significant"], val_col]
                    return v if not v.empty else None
                return g[val_col]

            all_mins = [v.min() for g in model_gdfs if (v := _vals_for_norm(g)) is not None]
            all_mins += [v.min() for g in obs_gdfs if (v := _vals_for_norm(g)) is not None]
            all_maxs = [v.max() for g in model_gdfs if (v := _vals_for_norm(g)) is not None]
            all_maxs += [v.max() for g in obs_gdfs if (v := _vals_for_norm(g)) is not None]

            max_abs = max(abs(min(all_mins)), abs(max(all_maxs)))
            vmin, vmax = -max_abs, max_abs

            # Specific forcing for hourly_reduce
            if "horaire_reduce" in str(dir_path):
                seasons_lower = [t.lower() for t in (titles or [])]
                if any("jan" in t for t in seasons_lower):
                    vmin, vmax = -260, 260
                if any("hydro" in t for t in seasons_lower):
                    vmin, vmax = -153, 153

            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

            n_colors = 15 
            # Load RGB triplets (0-255)
            rgb = np.loadtxt("src/colors/prec_div.txt") / 255.0               # 256 × 3 → 0‑1
            # Select equally spaced indices
            idx = np.linspace(0, rgb.shape[0] - 1, n_colors, dtype=int)
            # Convert to hex for from_list()
            hex_colors = [to_hex(rgb[i]) for i in idx]
            hex_colors[len(hex_colors) // 2] = "#808080"
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
        vmin = min(all_mins)
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

        n_colors = 15 
        # Load RGB triplets (0-255)
        rgb = np.loadtxt("src/colors/prec_seq.txt") / 255.0               # 256 × 3 → 0‑1
        # Select equally spaced indices
        idx = np.linspace(0, rgb.shape[0] - 1, n_colors, dtype=int)
        # Convert to hex for from_list()
        hex_colors = [to_hex(rgb[i]) for i in idx]
        cmap = LinearSegmentedColormap.from_list("prec_seq", hex_colors, N=n_colors)

        # Norma spread between 0 and vmax
        levels = np.linspace(vmin, vmax, 16)  # 15 intervals → 16 boundary levels
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # 8. Map export
    modes = {"rast": 5.5, "norast": -1}

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
            True:               "Significant",
            False:           "Non Significant"
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

            # Optional: remove small line artifacts
            coast = coast[coast.length > 2_000]  # 2 km, adjustable

            coast.plot(
                ax=ax,
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )


            # Model
            if show_mod and gdf_m is not None:
                gdf_m_plot = gdf_m
                if show_signif and "significant" in gdf_m.columns:
                    gdf_m_plot = gdf_m.loc[gdf_m["significant"]].copy()
                if not gdf_m_plot.empty:
                    gdf_m_plot.plot(ax=ax, column=val_col, cmap=cmap, norm=norm, linewidth=0, zorder=1)

            # Observations
            if show_obs and gdf_o is not None and not gdf_o.empty:

                # Reviewer request: uniform dot size for all displayed stations
                mean_pt = middle_pt / 2
                gdf_o.loc[:, "_size_pt2"] = mean_pt**2

                kw_hollow = dict(
                    markersize="_size_pt2",
                    marker="o",
                    color="white",
                    edgecolor=obs_edgecolor,
                    linewidth=0.5,
                )

                if show_signif and "significant" in gdf_o.columns:
                    gdf_nonsig = gdf_o.loc[~gdf_o["significant"]]
                    gdf_plot = gdf_o.loc[gdf_o["significant"]].copy()
                else:
                    gdf_nonsig = gdf_o.iloc[0:0]
                    gdf_plot = gdf_o

                if not gdf_nonsig.empty:
                    gdf_nonsig.plot(ax=ax, zorder=3, **kw_hollow)

                if not gdf_plot.empty:
                    # split near-zero vs non-zero among significant stations only
                    vals = gdf_plot[val_col]
                    max_abs = float(vals.abs().max()) or 1.0
                    span = float(vals.max() - vals.min()) or max_abs
                    threshold = max(0.05 * max_abs, span / 15, 1e-5)
                    gdf_zero = gdf_plot[vals.abs() <= threshold]
                    gdf_nonzero = gdf_plot[vals.abs() > threshold]

                    kw_zero = dict(
                        markersize="_size_pt2",
                        marker="o",
                        edgecolor="#333333",
                        linewidth=0.5,
                    )

                    kw_nonzero = dict(
                        markersize="_size_pt2",
                        marker="o",
                        edgecolor="face",
                        linewidth=0.1,
                    )

                    if not gdf_zero.empty:
                        if obs_facecolor is None:
                            gdf_zero.plot(ax=ax, color="#808080", zorder=4, **kw_zero)
                        else:
                            gdf_zero.plot(ax=ax, color=obs_facecolor, zorder=4, **kw_zero)

                    if not gdf_nonzero.empty:
                        if obs_facecolor is None:
                            gdf_nonzero.plot(
                                ax=ax, column=val_col, cmap=cmap, norm=norm,
                                zorder=5, **kw_nonzero,
                            )
                        else:
                            gdf_nonzero.plot(
                                ax=ax, color=obs_facecolor, zorder=5, **kw_nonzero,
                            )


            # Relief
            relief.plot(ax=ax, color=relief_color, linewidth=relief_linewidth, alpha=0.8, zorder=6)

            ax.set_axis_off()
            ax.set_rasterization_zorder(z)

            if False:
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
                    # labels in order of all_cat
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
                    # Force one decimal point
                    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


            suffix_obs = "obs" if show_obs else ""
            suffix_mod = "mod" if show_mod else ""
            suffix_signif = "_signif" if show_signif else ""
            suffix_diff = "_rdiff" if rdiff else ("_diff" if diff else "")
            name_file = f"{suffix_mod}{suffix_obs}{suffix_signif}_{mode}{suffix_diff}"

            subdir = dir_path / title.lower() 
            subdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(subdir / f"{name_file}.svg", format="svg", bbox_inches="tight", pad_inches=0, dpi=180)
            fig.savefig(subdir / f"{name_file}.pdf", format="pdf", bbox_inches="tight", pad_inches=0, dpi=180)
            plt.close(fig)
            logger.info(subdir / f"{name_file}.svg")

    # 9. Legend only
    if diff:
        mpl.rcParams.update({"font.size": 30})
        fig2 = plt.figure(figsize=(2.0, figsize[1] * 2))
        ax2 = fig2.add_axes([0.08, 0.05, 0.32, 0.9])
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
            cb2.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{int(round(x)):+4.0f}")
            )
        else:
            cb2.ax.set_ylabel(f"{legend}", rotation=90, fontsize=22)
            cb2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    suffix_signif = "_signif" if show_signif else ""
    suffix_diff = "_rdiff" if rdiff else ("_diff" if diff else "")
    name_legend = f"legend{suffix_signif}{suffix_diff}"

    fig2.savefig(dir_path / f"{name_legend}.svg", format="svg", bbox_inches="tight", pad_inches=0)
    fig2.savefig(dir_path / f"{name_legend}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig2)

    # --- 9bis. Légende horizontale ---------------------------------------
    mpl.rcParams.update({"font.size": 22})
    figh = plt.figure(figsize=(figsize[0] * 1.4, 1.2))
    axh = figh.add_axes([0.08, 0.45, 0.84, 0.35])

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
        cbh.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

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
    out_dir = Path(f"outputs/hist/dispo/{echelle}{suffix}/{season.lower()}")
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
        datasets_rdiff = []

        df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")

        for d in datasets:
            mod = d.get("modelised")    # pl.DataFrame
            obs = d.get("observed")     # pl.DataFrame
            col = d.get("column")
            season = d.get("season")

            if show_signif:
                # mod = mod.filter(pl.col("significant") == True)
                obs = obs.filter(pl.col("significant") == True)
                # Always base on station data and calculate with corresponding AROME even if non-significant 

            obs_vs_mod = match_and_compare(obs, mod, col, df_obs_vs_mod)

            # Absolute diff (ME)
            obs_vs_mod_diff = obs_vs_mod.select([
                pl.Series("NUM_POSTE", range(1, obs_vs_mod.height + 1)),
                pl.col("lat"),
                pl.col("lon"),
                (pl.col("AROME") - pl.col("Station")).alias(col)
            ])

            # Relative diff (RB %)
            obs_vs_mod_rdiff = obs_vs_mod.select([
                pl.Series("NUM_POSTE", range(1, obs_vs_mod.height + 1)),
                pl.col("lat"),
                pl.col("lon"),
                ((pl.col("AROME") - pl.col("Station")) / pl.col("Station").abs().clip(lower_bound=1e-6) * 100).alias(col)
            ])

            datasets_diff.append({
                "modelised": None,
                "observed": obs_vs_mod_diff,
                "column": col,
                "season": season
            })
            datasets_rdiff.append({
                "modelised": None,
                "observed": obs_vs_mod_rdiff,
                "column": col,
                "season": season
            })

            if obs_vs_mod.is_empty():
                logger.warning(f"No data after filtering for {echelle} {season} - {col_calculate} - signif {show_signif}")
                continue

            # Unit legend
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
                label="Data",
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
            logger.warning(f"No significant results, metrics{suffix}.csv not created")

    return datasets_diff, datasets_rdiff

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
                SIGNIFICANT_SHOW = [True] # Choose whether to display significant points
            else:
                SIGNIFICANT_SHOW = [False]

            logger.info(f"Scale {e} - Season {s} data generated")
            datasets.append(res)    # Store result in dataset list
     
        for signif in SIGNIFICANT_SHOW:
            dir_path, val_col, titles, model_gdfs, obs_gdfs = calculate_data_maps(
                datasets,
                echelle=e,
                data_type=data_type,
                col_calculate=col_calculate,
                reduce_activate=reduce_activate,
                show_signif=signif,
                titles=[s.upper() for s in season],  # subplot titles
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
                    show_signif=signif,
                    col_calculate=col_calculate,
                    scale=scale
                )

        if data_type == "dispo":
            generate_hist(res, echelle=e, season=s, reduce_activate=reduce_activate)

        if data_type != "dispo" and col_calculate not in ["significant", "model"]:
            for signif in SIGNIFICANT_SHOW:

                # Conditions d'appel de generate_scatter :
                # - toujours pour l'échelle "quotidien"
                # - pour l'échelle "horaire" seulement si reduce_activate == False
                if (e == "quotidien") or (e == "horaire" and not reduce_activate):
                    datasets_diff, datasets_rdiff = generate_scatter(
                        datasets=datasets,
                        dir_path=dir_path,
                        col_calculate=col_calculate,
                        echelle=e,
                        show_signif=signif,
                        scale=scale
                    )
                    
                    # --- Absolute diff maps (ME) ---
                    dir_path_d, val_col_d, titles_d, model_gdfs_d, obs_gdfs_d = calculate_data_maps(
                        datasets_diff,
                        echelle=e,
                        data_type=data_type,
                        col_calculate=col_calculate,
                        reduce_activate=reduce_activate,
                        show_signif=signif,
                        titles=[s.upper() for s in season],
                        saturation_col=sat,
                        diff=True
                    )

                    generate_maps(
                        dir_path=dir_path_d,
                        model_gdfs=model_gdfs_d,
                        obs_gdfs=obs_gdfs_d,
                        data_type=data_type,
                        titles=titles_d,
                        val_col=val_col_d,
                        show_mod=False,
                        show_obs=True,
                        show_signif=signif,
                        col_calculate=col_calculate,
                        scale=scale,
                        diff=True,
                        mesure=mesure
                    )

                    # --- Relative diff maps (RB %) ---
                    dir_path_r, val_col_r, titles_r, model_gdfs_r, obs_gdfs_r = calculate_data_maps(
                        datasets_rdiff,
                        echelle=e,
                        data_type=data_type,
                        col_calculate=col_calculate,
                        reduce_activate=reduce_activate,
                        show_signif=signif,
                        titles=[s.upper() for s in season],
                        saturation_col=sat,
                        diff=True
                    )

                    generate_maps(
                        dir_path=dir_path_r,
                        model_gdfs=model_gdfs_r,
                        obs_gdfs=obs_gdfs_r,
                        data_type=data_type,
                        titles=titles_r,
                        val_col=val_col_r,
                        show_mod=False,
                        show_obs=True,
                        show_signif=signif,
                        col_calculate=col_calculate,
                        scale=scale,
                        diff=True,
                        rdiff=True,
                        mesure=mesure
                    )


def str2bool(v):
    if v == "True":
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map generation pipeline")
    parser.add_argument("--data_type", choices=["dispo", "stats", "gev"])
    parser.add_argument("--col_calculate", choices=["n_years", "numday", "mean", "mean-max", "z_T_p", "zTpa", "significant", "model"], default=None)
    parser.add_argument("--echelle", choices=["horaire", "quotidien", "horaire_reduce"], nargs='+', default=["horaire"])
    parser.add_argument("--season", type=str, nargs='+', default=["son"])
    parser.add_argument("--reduce_activate", type=str2bool, default=False)
    parser.add_argument("--sat", type=float, default=100)

    args = parser.parse_args()  
    main(args)
