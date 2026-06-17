"""
Generate RL2 trend maps for daily and hourly data (observed + AROME).
Produces figures similar to Figures 6 and 7 of the manuscript but for T=2.
Uses existing GEV best-model parameters and niveau_retour significance.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from pathlib import Path

# ---- Paths (relative to repo root) ----
REPO = Path(".")
RELIEF_PATH = REPO / "data/external/external/niveaux/selection_courbes_niveau_france.shp"
DEPTS_PATH = REPO / "data/external/external/departements/depts.shp"
COLORS_PATH = REPO / "src/colors/prec_div.txt"
OUTPUT_DIR = REPO / "perso/v2_modifs/Figures"

CONFIGS = {
    "daily": {
        "gev_obs":  REPO / "data/gev/gev/observed/quotidien/hydro",
        "gev_mod":  REPO / "data/gev/gev/modelised/quotidien/hydro",
        "meta_obs": REPO / "data/metadonnees/observed/postes_quotidien.csv",
        "meta_mod": REPO / "data/metadonnees/modelised/postes_quotidien.csv",
        "maxima_obs": REPO / "data/statisticals/statisticals/observed/quotidien",
        "maxima_mod": REPO / "data/statisticals/statisticals/modelised/horaire",  # AROME stores all in horaire/
        "mesure_obs": "max_mm_j",
        "mesure_mod": "max_mm_j",
        "min_year": 1960,  # hydro = +1
        "max_year": 2022,
        "len_serie_obs": 50,
        "len_serie_mod": 25,  # AROME: no strict filtering needed
    },
    "hourly": {
        "gev_obs":  REPO / "data/gev/gev/observed/horaire/hydro",
        "gev_mod":  REPO / "data/gev/gev/modelised/horaire/hydro",
        "meta_obs": REPO / "data/metadonnees/observed/postes_horaire.csv",
        "meta_mod": REPO / "data/metadonnees/modelised/postes_horaire.csv",
        "maxima_obs": REPO / "data/statisticals/statisticals/observed/horaire",
        "maxima_mod": REPO / "data/statisticals/statisticals/modelised/horaire",
        "mesure_obs": "max_mm_h",
        "mesure_mod": "max_mm_h",
        "min_year": 1991,  # hydro = +1 from 1990
        "max_year": 2022,
        "len_serie_obs": 25,
        "len_serie_mod": 25,
    },
}


def load_maxima_years(maxima_dir, season, mesure, min_year, max_year, len_serie):
    """Load maxima data and return df with NUM_POSTE and year columns (filtered)."""
    dfs = []
    for y in range(min_year, max_year + 1):
        p = Path(maxima_dir) / f"{y:04d}" / f"{season}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["NUM_POSTE", mesure, "nan_ratio"])
            df["year"] = y
            dfs.append(df)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["NUM_POSTE"] = df["NUM_POSTE"].astype(str)
    df = df[df["nan_ratio"] <= 0.10].dropna(subset=[mesure])
    # Filter stations with enough years
    counts = df.groupby("NUM_POSTE")["year"].nunique()
    valid = counts[counts >= len_serie].index
    df = df[df["NUM_POSTE"].isin(valid)]
    return df[["NUM_POSTE", "year"]].drop_duplicates()


def compute_rl_trend(gev_df, maxima_years_df, min_year, max_year, T=2):
    """
    Compute the relative trend (%) of the T-year return level over [1992, max_year]
    using the same normalization as pipeline_best_to_niveau_retour.compute_zT_for_years.
    """
    a_year, b_year = 1992, max_year
    results = []
    for _, row in gev_df.iterrows():
        poste = row["NUM_POSTE"]
        mu0, mu1 = row["mu0"], row["mu1"]
        sigma0, sigma1 = row["sigma0"], row["sigma1"]
        xi = row["xi"]

        sub = maxima_years_df[maxima_years_df["NUM_POSTE"] == poste]
        if len(sub) == 0:
            continue
        years_obs = sub["year"].values

        # Same normalization as compute_zT_for_years
        t_tilde_obs_raw = (years_obs - min_year) / (max_year - min_year)
        t_min_ret = t_tilde_obs_raw.min()
        t_max_ret = t_tilde_obs_raw.max()
        if t_max_ret == t_min_ret:
            continue
        res0_obs = t_tilde_obs_raw / (t_max_ret - t_min_ret)
        dx = res0_obs.min() + 0.5

        # Compute t_tilde for years a and b
        t_tilde_retour = []
        for y in [a_year, b_year]:
            t_raw = (y - min_year) / (max_year - min_year)
            res0 = t_raw / (t_max_ret - t_min_ret)
            t_tilde = res0 - dx
            t_tilde_retour.append(t_tilde)

        CT = (-np.log(1 - 1 / T)) ** (-xi) - 1
        zTa = mu0 + mu1 * t_tilde_retour[0] + (sigma0 + sigma1 * t_tilde_retour[0]) / xi * CT
        zTb = mu0 + mu1 * t_tilde_retour[1] + (sigma0 + sigma1 * t_tilde_retour[1]) / xi * CT
        if zTa == 0 or np.isnan(zTa) or np.isnan(zTb):
            continue
        z_T_p = (zTb - zTa) / zTa * 100

        results.append({"NUM_POSTE": poste, "z_T_p": z_T_p})
    return pd.DataFrame(results)


def build_map_elements():
    """Load and prepare shared map elements: coastline, relief, colormap."""
    depts = gpd.read_file(DEPTS_PATH)
    depts = depts[~depts["NUM_DEP"].isin(["2A", "2B"])]
    depts = depts.set_crs("EPSG:27572").to_crs("EPSG:2154")
    depts["geometry"] = depts.geometry.simplify(500)
    mask = depts.union_all() if hasattr(depts, "union_all") else depts.unary_union

    if mask.geom_type == "MultiPolygon":
        polys = list(mask.geoms)
    else:
        polys = [mask]
    coast = gpd.GeoSeries([p.exterior for p in polys], crs="EPSG:2154")
    coast = coast[coast.length > 2000]

    relief = gpd.read_file(RELIEF_PATH).to_crs("EPSG:2154").clip(mask)
    relief["geometry"] = relief.geometry.simplify(500)

    # Colormap
    n_colors = 15
    rgb = np.loadtxt(COLORS_PATH) / 255.0
    idx = np.linspace(0, rgb.shape[0] - 1, n_colors, dtype=int)
    hex_colors = [to_hex(rgb[i]) for i in idx]
    hex_colors[len(hex_colors) // 2] = "#808080"
    cmap = LinearSegmentedColormap.from_list("prec_div", hex_colors, N=n_colors)

    return coast, relief, mask, cmap


def plot_trend_map(ax, gdf, col, coast, relief, cmap, norm, title, is_grid=False):
    """Plot a single trend map panel."""
    coast.plot(ax=ax, edgecolor="black", linewidth=0.6, zorder=1)

    if is_grid:
        # AROME grid: plot as small squares
        from shapely.geometry import box as shapely_box
        half = 2500 / 2  # 2.5 km grid -> 1.25 km half-side
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(
            lambda p: shapely_box(p.x - half, p.y - half, p.x + half, p.y + half)
        )
        # Separate near-zero
        vals = gdf[col]
        max_abs = float(vals.abs().max()) or 1.0
        span = float(vals.max() - vals.min()) or max_abs
        threshold = max(0.05 * max_abs, span / 15, 1e-5)
        gdf_zero = gdf[vals.abs() <= threshold]
        gdf_nonzero = gdf[vals.abs() > threshold]
        if not gdf_zero.empty:
            gdf_zero.plot(ax=ax, color="#808080", linewidth=0, zorder=2)
        if not gdf_nonzero.empty:
            gdf_nonzero.plot(ax=ax, column=col, cmap=cmap, norm=norm, linewidth=0, zorder=2)
    else:
        # Observations: uniform dots
        pt_size = 4.5 ** 2
        vals = gdf[col]
        max_abs = float(vals.abs().max()) or 1.0
        span = float(vals.max() - vals.min()) or max_abs
        threshold = max(0.05 * max_abs, span / 15, 1e-5)
        gdf_zero = gdf[vals.abs() <= threshold]
        gdf_nonzero = gdf[vals.abs() > threshold]
        if not gdf_zero.empty:
            gdf_zero.plot(ax=ax, color="#808080", markersize=pt_size, marker="o",
                         edgecolor="#333333", linewidth=0.5, zorder=2)
        if not gdf_nonzero.empty:
            gdf_nonzero.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                           markersize=pt_size, marker="o", edgecolor="face",
                           linewidth=0.1, zorder=3)

    relief.plot(ax=ax, color="black", linewidth=0.3, alpha=0.8, zorder=5)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, fontweight="bold")


def make_figure(
    gdf_mod, gdf_obs, coast, relief, mask, cmap,
    title_mod, title_obs, output_name, echelle_label
):
    """Generate a 1x2 figure: AROME (left) vs Observations (right)."""
    # Apply significance: set non-significant to 0
    if "significant" in gdf_obs.columns:
        gdf_obs = gdf_obs.copy()
        gdf_obs.loc[gdf_obs["significant"] == False, "z_T_p"] = 0
    if gdf_mod is not None and "significant" in gdf_mod.columns:
        gdf_mod = gdf_mod.copy()
        gdf_mod.loc[gdf_mod["significant"] == False, "z_T_p"] = 0

    # Clip to mask
    gdf_obs = gdf_obs.clip(mask)
    if gdf_mod is not None:
        gdf_mod = gdf_mod.clip(mask)

    # Shared norm
    all_vals = list(gdf_obs["z_T_p"].dropna())
    if gdf_mod is not None:
        all_vals += list(gdf_mod["z_T_p"].dropna())
    max_abs = max(abs(min(all_vals)), abs(max(all_vals))) if all_vals else 50
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    # Compute statistics
    n_obs = len(gdf_obs)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6.5), dpi=300)

    if gdf_mod is not None:
        plot_trend_map(axs[0], gdf_mod, "z_T_p", coast, relief, cmap, norm,
                      title_mod, is_grid=True)
    else:
        axs[0].set_visible(False)

    plot_trend_map(axs[1], gdf_obs, "z_T_p", coast, relief, cmap, norm,
                  title_obs, is_grid=False)

    # Colorbar
    fig.subplots_adjust(bottom=0.18, wspace=0.05)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"Relative trend (%)", fontsize=10)
    cbar.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_png = OUTPUT_DIR / f"{output_name}.png"
    out_pdf = OUTPUT_DIR / f"{output_name}.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved {out_png}")


def process_scale(scale_key, T=2):
    """Process one scale (daily or hourly): compute RL_T trends and generate maps."""
    cfg = CONFIGS[scale_key]
    print(f"\n{'='*60}")
    print(f"Processing {scale_key} (T={T})")
    print(f"{'='*60}")

    # Load GEV parameters
    gev_obs = pd.read_parquet(cfg["gev_obs"] / "gev_param_best_model.parquet")
    gev_obs["NUM_POSTE"] = gev_obs["NUM_POSTE"].astype(str)

    gev_mod = pd.read_parquet(cfg["gev_mod"] / "gev_param_best_model.parquet")
    gev_mod["NUM_POSTE"] = gev_mod["NUM_POSTE"].astype(str)

    # Load existing RL10 significance from niveau_retour.parquet
    nr_obs = pd.read_parquet(cfg["gev_obs"] / "niveau_retour.parquet")
    nr_obs["NUM_POSTE"] = nr_obs["NUM_POSTE"].astype(str)
    signif_obs = nr_obs[["NUM_POSTE", "significant"]].drop_duplicates()

    nr_mod = pd.read_parquet(cfg["gev_mod"] / "niveau_retour.parquet")
    nr_mod["NUM_POSTE"] = nr_mod["NUM_POSTE"].astype(str)
    signif_mod = nr_mod[["NUM_POSTE", "significant"]].drop_duplicates()

    # Load maxima years for t_tilde normalization
    print("Loading observed maxima...")
    maxima_obs = load_maxima_years(
        cfg["maxima_obs"], "hydro", cfg["mesure_obs"],
        cfg["min_year"], cfg["max_year"], cfg["len_serie_obs"]
    )

    # For AROME, load from its statistics directory
    print("Loading AROME maxima...")
    maxima_mod = load_maxima_years(
        cfg["maxima_mod"], "hydro", cfg["mesure_mod"],
        cfg["min_year"], cfg["max_year"], cfg["len_serie_mod"]
    )

    # Compute RL_T trends
    print(f"Computing RL{T} trends for observed stations...")
    trends_obs = compute_rl_trend(gev_obs, maxima_obs, cfg["min_year"], cfg["max_year"], T=T)
    print(f"  -> {len(trends_obs)} stations")

    if maxima_mod is not None:
        print(f"Computing RL{T} trends for AROME...")
        trends_mod = compute_rl_trend(gev_mod, maxima_mod, cfg["min_year"], cfg["max_year"], T=T)
        print(f"  -> {len(trends_mod)} grid points")
    else:
        trends_mod = None

    # Merge with significance
    trends_obs = trends_obs.merge(signif_obs, on="NUM_POSTE", how="left")
    trends_obs["significant"] = trends_obs["significant"].fillna(False)

    if trends_mod is not None and len(trends_mod) > 0:
        trends_mod = trends_mod.merge(signif_mod, on="NUM_POSTE", how="left")
        trends_mod["significant"] = trends_mod["significant"].fillna(False)
    elif trends_mod is not None and len(trends_mod) == 0:
        trends_mod = None

    # Add coordinates
    meta_obs = pd.read_csv(cfg["meta_obs"])
    meta_obs["NUM_POSTE"] = meta_obs["NUM_POSTE"].astype(str)
    trends_obs = trends_obs.merge(meta_obs[["NUM_POSTE", "lat", "lon"]], on="NUM_POSTE", how="inner")

    gdf_obs = gpd.GeoDataFrame(
        trends_obs,
        geometry=gpd.points_from_xy(trends_obs["lon"], trends_obs["lat"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:2154")

    gdf_mod = None
    if trends_mod is not None:
        meta_mod = pd.read_csv(cfg["meta_mod"])
        meta_mod["NUM_POSTE"] = meta_mod["NUM_POSTE"].astype(str)
        trends_mod = trends_mod.merge(meta_mod[["NUM_POSTE", "lat", "lon"]], on="NUM_POSTE", how="inner")
        gdf_mod = gpd.GeoDataFrame(
            trends_mod,
            geometry=gpd.points_from_xy(trends_mod["lon"], trends_mod["lat"]),
            crs="EPSG:4326"
        ).to_crs("EPSG:2154")

    return gdf_mod, gdf_obs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate RL_T trend maps")
    parser.add_argument("--T", type=int, default=2, help="Return period (default: 2)")
    args = parser.parse_args()
    T = args.T

    coast, relief, mask, cmap = build_map_elements()

    # Daily
    gdf_mod_d, gdf_obs_d = process_scale("daily", T=T)
    make_figure(
        gdf_mod_d, gdf_obs_d, coast, relief, mask, cmap,
        title_mod=f"(a) AROME — Daily $RL_{{{T}}}$ trend",
        title_obs=f"(b) Stations — Daily $RL_{{{T}}}$ trend",
        output_name=f"rl{T}_trends_daily",
        echelle_label="Daily"
    )

    # Hourly
    gdf_mod_h, gdf_obs_h = process_scale("hourly", T=T)
    make_figure(
        gdf_mod_h, gdf_obs_h, coast, relief, mask, cmap,
        title_mod=f"(a) AROME — Hourly $RL_{{{T}}}$ trend",
        title_obs=f"(b) Stations — Hourly $RL_{{{T}}}$ trend",
        output_name=f"rl{T}_trends_hourly",
        echelle_label="Hourly"
    )

    print(f"\nDone! Figures saved to {OUTPUT_DIR}/rl{T}_trends_*.png")


if __name__ == "__main__":
    main()

