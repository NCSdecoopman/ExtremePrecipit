"""
Generate RL_T trend maps for daily and hourly data (observed + AROME).
Produces combined figures (RL2 + RL5) with AROME | Stations panels and a
shared colour scale, similar to Figures 4, 6 and 7 layout.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from concurrent.futures import ThreadPoolExecutor
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from matplotlib.gridspec import GridSpec
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


def _read_maxima_year(path, mesure):
    df = pd.read_parquet(path, columns=["NUM_POSTE", mesure, "nan_ratio"])
    df["year"] = int(path.parent.name)
    return df


def load_maxima_years(maxima_dir, season, mesure, min_year, max_year, len_serie):
    """Load maxima data and return df with NUM_POSTE and year columns (filtered)."""
    maxima_dir = Path(maxima_dir)
    paths = [
        maxima_dir / f"{y:04d}" / f"{season}.parquet"
        for y in range(min_year, max_year + 1)
        if (maxima_dir / f"{y:04d}" / f"{season}.parquet").exists()
    ]
    if not paths:
        return None

    with ThreadPoolExecutor() as pool:
        dfs = list(pool.map(lambda p: _read_maxima_year(p, mesure), paths))
    df = pd.concat(dfs, ignore_index=True)
    df["NUM_POSTE"] = df["NUM_POSTE"].astype(str)
    df = df[df["nan_ratio"] <= 0.10].dropna(subset=[mesure])
    counts = df.groupby("NUM_POSTE", sort=False)["year"].nunique()
    valid = counts[counts >= len_serie].index
    return df.loc[df["NUM_POSTE"].isin(valid), ["NUM_POSTE", "year"]].drop_duplicates()


def _time_norm_table(maxima_years_df, min_year, max_year, a_year, b_year):
    """Per-station normalized time for return-level endpoints (vectorized)."""
    my = maxima_years_df[["NUM_POSTE", "year"]].copy()
    my["t_raw"] = (my["year"] - min_year) / (max_year - min_year)
    agg = my.groupby("NUM_POSTE", sort=False)["t_raw"].agg(["min", "max"])
    agg = agg[agg["min"] < agg["max"]]
    agg["span"] = agg["max"] - agg["min"]
    my = my.join(agg[["span"]], on="NUM_POSTE")
    my["res0"] = my["t_raw"] / my["span"]
    dx = my.groupby("NUM_POSTE", sort=False)["res0"].min() + 0.5
    norm = agg.join(dx.rename("dx"))
    for y, col in ((a_year, "t_a"), (b_year, "t_b")):
        t_raw_y = (y - min_year) / (max_year - min_year)
        norm[col] = t_raw_y / norm["span"] - norm["dx"]
    return norm.reset_index()[["NUM_POSTE", "t_a", "t_b"]]


def compute_rl_trend(gev_df, maxima_years_df, min_year, max_year, T=2):
    """
    Compute the relative trend (%) of the T-year return level over [1992, max_year]
    using the same normalization as pipeline_best_to_niveau_retour.compute_zT_for_years.
    """
    a_year, b_year = 1992, max_year
    norm = _time_norm_table(maxima_years_df, min_year, max_year, a_year, b_year)
    df = gev_df.merge(norm, on="NUM_POSTE", how="inner")

    xi = df["xi"].to_numpy()
    CT = (-np.log(1 - 1 / T)) ** (-xi) - 1
    t_a = df["t_a"].to_numpy()
    t_b = df["t_b"].to_numpy()
    mu0 = df["mu0"].to_numpy()
    mu1 = df["mu1"].to_numpy()
    sigma0 = df["sigma0"].to_numpy()
    sigma1 = df["sigma1"].to_numpy()

    zTa = mu0 + mu1 * t_a + (sigma0 + sigma1 * t_a) / xi * CT
    zTb = mu0 + mu1 * t_b + (sigma0 + sigma1 * t_b) / xi * CT
    valid = (zTa != 0) & np.isfinite(zTa) & np.isfinite(zTb)
    z_T_p = np.full(len(df), np.nan)
    z_T_p[valid] = (zTb[valid] - zTa[valid]) / zTa[valid] * 100

    out = pd.DataFrame({"NUM_POSTE": df["NUM_POSTE"].values, "z_T_p": z_T_p})
    return out.dropna(subset=["z_T_p"]).reset_index(drop=True)


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


def _significant_only(gdf):
    if gdf is None or gdf.empty:
        return gdf
    if "significant" in gdf.columns:
        return gdf.loc[gdf["significant"]].copy()
    return gdf


def _split_zero_nonzero(gdf, col):
    if gdf is None or gdf.empty:
        return gdf, gdf
    vals = gdf[col]
    max_abs = float(vals.abs().max()) or 1.0
    span = float(vals.max() - vals.min()) or max_abs
    threshold = max(0.05 * max_abs, span / 15, 1e-5)
    return gdf[vals.abs() <= threshold], gdf[vals.abs() > threshold]


def plot_trend_map(ax, gdf, col, coast, relief, cmap, norm, title, is_grid=False):
    """Plot a single trend map panel."""
    coast.plot(ax=ax, edgecolor="black", linewidth=0.6, zorder=1)

    if gdf is None or gdf.empty:
        relief.plot(ax=ax, color="black", linewidth=0.3, alpha=0.8, zorder=5)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    if is_grid:
        gdf_sig = _significant_only(gdf)
        if gdf_sig is not None and not gdf_sig.empty:
            from shapely.geometry import box as shapely_box
            half = 2500 / 2
            gdf_sig = gdf_sig.copy()
            gdf_sig["geometry"] = gdf_sig.geometry.apply(
                lambda p: shapely_box(p.x - half, p.y - half, p.x + half, p.y + half)
            )
            gdf_zero, gdf_nonzero = _split_zero_nonzero(gdf_sig, col)
            if not gdf_zero.empty:
                gdf_zero.plot(ax=ax, color="#808080", linewidth=0, zorder=2)
            if not gdf_nonzero.empty:
                gdf_nonzero.plot(ax=ax, column=col, cmap=cmap, norm=norm, linewidth=0, zorder=2)
    else:
        pt_size = 4.5 ** 2
        if "significant" in gdf.columns:
            gdf_nonsig = gdf.loc[~gdf["significant"]]
            gdf_sig = gdf.loc[gdf["significant"]]
            if not gdf_nonsig.empty:
                gdf_nonsig.plot(
                    ax=ax, color="white", markersize=pt_size, marker="o",
                    edgecolor="#BFBFBF", linewidth=0.5, zorder=2,
                )
        else:
            gdf_sig = gdf

        if gdf_sig is not None and not gdf_sig.empty:
            gdf_zero, gdf_nonzero = _split_zero_nonzero(gdf_sig, col)
            if not gdf_zero.empty:
                gdf_zero.plot(
                    ax=ax, color="#808080", markersize=pt_size, marker="o",
                    edgecolor="#333333", linewidth=0.5, zorder=3,
                )
            if not gdf_nonzero.empty:
                gdf_nonzero.plot(
                    ax=ax, column=col, cmap=cmap, norm=norm,
                    markersize=pt_size, marker="o", edgecolor="face",
                    linewidth=0.1, zorder=4,
                )

    relief.plot(ax=ax, color="black", linewidth=0.3, alpha=0.8, zorder=5)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, fontweight="bold")


def _sig_values(gdf, col="z_T_p"):
    if gdf is None or gdf.empty:
        return []
    g = _significant_only(gdf)
    return list(g[col].dropna()) if g is not None and not g.empty else []


def _shared_norm(gdfs, col="z_T_p"):
    all_vals = []
    for gdf in gdfs:
        all_vals.extend(_sig_values(gdf, col))
    max_abs = max(abs(min(all_vals)), abs(max(all_vals))) if all_vals else 50
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def make_combined_figure(
    panels,
    coast,
    relief,
    mask,
    cmap,
    output_name,
    col_header=("AROME", "Stations"),
):
    """
    panels: list of (T, gdf_mod, gdf_obs) — one row per return period.
    Layout: 2 columns (AROME | Stations), one shared colour bar per RL row.
    """
    clipped = []
    for T, gdf_mod, gdf_obs in panels:
        gobs = gdf_obs.clip(mask)
        gmod = gdf_mod.clip(mask) if gdf_mod is not None else None
        clipped.append((T, gmod, gobs))

    n_rows = len(clipped)
    fig_h = 5.5 * n_rows + 0.35
    fig = plt.figure(figsize=(12, fig_h), dpi=300)
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([1.0, 0.045])
    gs = GridSpec(
        n_rows * 2, 2,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.05,
        left=0.07,
        right=0.98,
        top=0.96,
        bottom=0.03,
    )

    fig.text(0.28, 0.985, col_header[0], ha="center", va="top", fontsize=10, fontweight="bold")
    fig.text(0.74, 0.985, col_header[1], ha="center", va="top", fontsize=10, fontweight="bold")

    for i, (T, gdf_mod, gdf_obs) in enumerate(clipped):
        map_gs_row = i * 2
        cbar_gs_row = i * 2 + 1
        row_gdfs = [g for g in (gdf_mod, gdf_obs) if g is not None]
        norm = _shared_norm(row_gdfs)

        ax_mod = fig.add_subplot(gs[map_gs_row, 0])
        ax_obs = fig.add_subplot(gs[map_gs_row, 1])

        if gdf_mod is not None and not gdf_mod.empty:
            plot_trend_map(
                ax_mod, gdf_mod, "z_T_p", coast, relief, cmap, norm,
                title="", is_grid=True,
            )
        else:
            ax_mod.set_visible(False)

        plot_trend_map(
            ax_obs, gdf_obs, "z_T_p", coast, relief, cmap, norm,
            title="", is_grid=False,
        )

        cbar_gs = gs[cbar_gs_row, :].subgridspec(1, 3, width_ratios=[0.14, 0.72, 0.14])
        cax = fig.add_subplot(cbar_gs[0, 1])
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
        cax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))
        cax.tick_params(labelsize=7, length=2, width=0.5, direction="out", pad=1)
        cb.outline.set_linewidth(0.5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_png = OUTPUT_DIR / f"{output_name}.png"
    out_pdf = OUTPUT_DIR / f"{output_name}.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved {out_png} ({n_rows} RL row(s): {[T for T, _, _ in clipped]})")


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
    parser = argparse.ArgumentParser(description="Generate RL_T trend maps (combined RL2 + RL5)")
    parser.add_argument(
        "--T", type=int, nargs="+", default=[2, 5],
        help="Return periods on separate rows (default: 2 5 — run once, not separately)",
    )
    args = parser.parse_args()
    periods = sorted(set(args.T))
    if len(periods) == 1:
        print(
            f"WARNING: single T={periods[0]} — figure will have one row only. "
            "For RL2+RL5 combined layout, run: uv run python .../generate_rl2_trend_maps.py"
        )

    coast, relief, mask, cmap = build_map_elements()

    for scale_key, output_name in [("daily", "rl2_rl5_trends_daily"), ("hourly", "rl2_rl5_trends_hourly")]:
        panels = []
        for T in periods:
            gdf_mod, gdf_obs = process_scale(scale_key, T=T)
            panels.append((T, gdf_mod, gdf_obs))
        make_combined_figure(panels, coast, relief, mask, cmap, output_name)

    print(f"\nDone! Figures saved to {OUTPUT_DIR}/rl2_rl5_trends_*.png")


if __name__ == "__main__":
    main()

