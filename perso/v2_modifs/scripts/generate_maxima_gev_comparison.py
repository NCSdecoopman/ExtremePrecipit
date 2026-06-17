import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex

def save_figure_atomically(fig, file_path, dpi=300, format=None):
    """
    Saves a figure to a temporary file first, then atomically replaces the target file.
    This prevents live-reloading tools like Quarto from reading a partially-written file.
    """
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    temp_path = os.path.join(dir_name, f"temp_{base_name}")
    
    # Save to temp file
    if format:
        fig.savefig(temp_path, bbox_inches='tight', dpi=dpi, format=format)
    else:
        fig.savefig(temp_path, bbox_inches='tight', dpi=dpi)
        
    # Atomic replace
    if os.path.exists(file_path):
        os.remove(file_path)
    os.rename(temp_path, file_path)

def main():
    # Paths (relative to repo root)
    gev_path = 'data/gev/gev/observed/quotidien/hydro/gev_param_best_model.parquet'
    maxima_dir = 'data/statisticals/statisticals/observed/quotidien'
    postes_path = 'data/metadonnees/metadonnees/observed/postes_quotidien.csv'
    depts_path = 'data/external/external/departements/depts.shp'
    relief_path = 'data/external/external/niveaux/selection_courbes_niveau_france.shp'
    output_dir = 'perso/v2_modifs/Figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    print("Loading GEV parameters...")
    gev_df = pd.read_parquet(gev_path)
    
    print("Loading annual maxima...")
    dfs = []
    for y in range(1960, 2023):
        p = os.path.join(maxima_dir, f"{y:04d}", "hydro.parquet")
        if os.path.exists(p):
            dfs.append(pd.read_parquet(p, columns=['NUM_POSTE', 'max_mm_j', 'nan_ratio']).assign(year=y))
    maxima_df = pd.concat(dfs, ignore_index=True)
    maxima_df = maxima_df[maxima_df['nan_ratio'] <= 0.10].dropna(subset=['max_mm_j'])
    
    print("Loading station coordinates...")
    postes_df = pd.read_csv(postes_path, usecols=['NUM_POSTE', 'lat', 'lon'])
    postes_df['NUM_POSTE'] = postes_df['NUM_POSTE'].astype(str)
    
    # 2. Calculate trends per station
    print("Calculating trends...")
    results = []
    for _, row in gev_df.iterrows():
        poste = row['NUM_POSTE']
        model = row['model']
        mu0 = row['mu0']
        mu1 = row['mu1']
        sigma0 = row['sigma0']
        sigma1 = row['sigma1']
        xi = row['xi']
        
        # Station observations
        sub = maxima_df[maxima_df['NUM_POSTE'] == poste].sort_values('year')
        if len(sub) < 50:
            continue
        
        years = sub['year'].values
        vals = sub['max_mm_j'].values
        
        tmin, tmax = years.min(), years.max()
        has_break = '_break_year' in model
        
        # delta_year
        delta_year = (tmax - 1985) if has_break else (tmax - tmin)
        
        # --- GEV-estimated RL2 Trend ---
        T = 2
        CT = (-np.log(1 - 1/T))**(-xi) - 1
        gev_slope_decade = (mu1 + (sigma1/xi)*CT) * (10 / delta_year)
        
        t_tilde_start = -0.5
        t_tilde_end = 0.5
        z_start = mu0 + mu1*t_tilde_start + (sigma0 + sigma1*t_tilde_start)/xi * CT
        z_end = mu0 + mu1*t_tilde_end + (sigma0 + sigma1*t_tilde_end)/xi * CT
        gev_trend_pct = (z_end - z_start) / z_start * 100
        
        # --- Simple Linear Regression on Maxima ---
        slope_full, intercept_full, r_full, p_full, stderr_full = linregress(years, vals)
        slope_decade_full = slope_full * 10
        pred_start_full = intercept_full + slope_full * tmin
        pred_end_full = intercept_full + slope_full * tmax
        pct_full = (pred_end_full - pred_start_full) / pred_start_full * 100
        
        results.append({
            'NUM_POSTE': poste,
            'gev_slope_decade': gev_slope_decade,
            'gev_trend_pct': gev_trend_pct,
            'slope_decade_full': slope_decade_full,
            'pct_full': pct_full
        })
        
    res_df = pd.DataFrame(results)
    # Filter out Corsica from stations (historically starting with '20')
    res_df = res_df[~res_df['NUM_POSTE'].str.startswith(('20', '2A', '2B'))]
    
    res_df = res_df.merge(postes_df, on='NUM_POSTE', how='inner')
    print(f"Computed comparison for {len(res_df)} stations (excluding Corsica).")
    
    # Load departments and filter Corsica
    print("Loading departments map...")
    depts = gpd.read_file(depts_path)
    depts = depts[~depts['NUM_DEP'].isin(['2A', '2B'])]
    depts = depts.set_crs("EPSG:27572").to_crs("EPSG:2154")
    
    # Build simplified metropolitan mask and coastline like in the paper
    print("Constructing coastline outline...")
    depts_simple = depts.copy()
    depts_simple['geometry'] = depts_simple.geometry.simplify(500)
    mask = depts_simple.union_all() if hasattr(depts_simple, 'union_all') else depts_simple.unary_union
    
    if mask.geom_type == "MultiPolygon":
        polys = list(mask.geoms)
    else:
        polys = [mask]
    exteriors = [poly.exterior for poly in polys]
    coast = gpd.GeoSeries(exteriors, crs="EPSG:2154")
    coast = coast[coast.length > 2000]
    
    # Load and clip relief contours
    print("Loading and clipping relief contours...")
    relief = gpd.read_file(relief_path).to_crs("EPSG:2154").clip(mask)
    relief['geometry'] = relief.geometry.simplify(500)
    
    # Convert results to GeoDataFrame projected to Lambert 93
    stations_gdf = gpd.GeoDataFrame(
        res_df,
        geometry=gpd.points_from_xy(res_df['lon'], res_df['lat']),
        crs="EPSG:4326"
    ).to_crs("EPSG:2154")
    
    # Matplotlib styling parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })
    
    # Diverging colormap limits and custom paper colormap
    vmin_pct, vmax_pct = -30, 30
    n_colors = 15
    rgb = np.loadtxt("src/colors/prec_div.txt") / 255.0
    idx = np.linspace(0, rgb.shape[0] - 1, n_colors, dtype=int)
    hex_colors = [to_hex(rgb[i]) for i in idx]
    cmap = LinearSegmentedColormap.from_list("prec_div", hex_colors, N=n_colors)
    norm = TwoSlopeNorm(vmin=vmin_pct, vcenter=0.0, vmax=vmax_pct)
    
    # Shared maximum absolute trend value to make sizes comparable between maps
    abs_max_shared = max(
        stations_gdf['pct_full'].abs().max(),
        stations_gdf['gev_trend_pct'].abs().max()
    ) or 1
    
    def plot_style_map(ax, gdf, col, title):
        # 1. Plot coastline boundary
        coast.plot(ax=ax, edgecolor="black", linewidth=0.6, zorder=1)
        
        # 2. Scale marker sizes proportionally to absolute trend value using the shared maximum
        abs_vals = gdf[col].abs()
        min_pt, max_pt = 4, 10
        sizes = ((abs_vals / abs_max_shared) * (max_pt - min_pt) + min_pt) ** 2
        
        # 3. Plot stations
        gdf.plot(
            ax=ax, column=col, cmap=cmap, norm=norm,
            markersize=sizes, marker="o", edgecolor="face", linewidth=0.1, zorder=3
        )
        
        # 4. Plot relief contours
        relief.plot(ax=ax, color="black", linewidth=0.3, alpha=0.8, zorder=5)
        
        ax.set_axis_off()
        ax.set_title(title)

    # ==========================================
    # 3. Figure A: 4-panel Overview
    # ==========================================
    print("Generating 4-panel overview figure...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 11), dpi=300)
    
    plot_style_map(axs[0, 0], stations_gdf, 'pct_full', '(a) Observed Annual Maxima Trend (%)')
    plot_style_map(axs[0, 1], stations_gdf, 'gev_trend_pct', r'(b) GEV $\mathrm{RL}_2$ Trend (%)')
    
    # Shared colorbar for maps
    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    cbar_ax = fig.add_axes([0.15, 0.52, 0.7, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Relative Trend over 1960–2022 (%)')
    
    # --- Subplot (c): Scatter plot of absolute trends (mm/decade) ---
    ax_sc_abs = axs[1, 0]
    x_abs = res_df['slope_decade_full'].values
    y_abs = res_df['gev_slope_decade'].values
    r_abs = res_df['slope_decade_full'].corr(res_df['gev_slope_decade'])
    
    ax_sc_abs.scatter(x_abs, y_abs, color='#1f77b4', alpha=0.4, edgecolors='none', s=15, label='Stations')
    lims_abs = [
        min(ax_sc_abs.get_xlim()[0], ax_sc_abs.get_ylim()[0]),
        max(ax_sc_abs.get_xlim()[1], ax_sc_abs.get_ylim()[1])
    ]
    ax_sc_abs.plot(lims_abs, lims_abs, color='#555555', linestyle='--', linewidth=1.0, label='1:1 Line')
    slope_fit, intercept_fit, _, _, _ = linregress(x_abs, y_abs)
    x_grid = np.linspace(lims_abs[0], lims_abs[1], 100)
    ax_sc_abs.plot(x_grid, intercept_fit + slope_fit*x_grid, color='#d62728', linestyle='-', linewidth=1.2, label='Linear Fit')
    ax_sc_abs.set_xlabel('Annual Maxima Linear Trend (mm/decade)')
    ax_sc_abs.set_ylabel(r'GEV $\mathrm{RL}_2$ Trend (mm/decade)')
    ax_sc_abs.set_title('(c) Absolute Trends')
    ax_sc_abs.grid(True, linestyle=':', alpha=0.5)
    ax_sc_abs.set_xlim(lims_abs)
    ax_sc_abs.set_ylim(lims_abs)
    ax_sc_abs.set_aspect('equal', adjustable='box')
    ax_sc_abs.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax_sc_abs.text(0.95, 0.05, f"$r$ = {r_abs:.3f}", transform=ax_sc_abs.transAxes,
                  verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                  fontsize=11, fontweight='bold')
    
    # --- Subplot (d): Scatter plot of relative trends (%) ---
    ax_sc_pct = axs[1, 1]
    x_pct = res_df['pct_full'].values
    y_pct = res_df['gev_trend_pct'].values
    r_pct = res_df['pct_full'].corr(res_df['gev_trend_pct'])
    
    ax_sc_pct.scatter(x_pct, y_pct, color='#2ca02c', alpha=0.4, edgecolors='none', s=15, label='Stations')
    lims_pct = [
        min(ax_sc_pct.get_xlim()[0], ax_sc_pct.get_ylim()[0]),
        max(ax_sc_pct.get_xlim()[1], ax_sc_pct.get_ylim()[1])
    ]
    ax_sc_pct.plot(lims_pct, lims_pct, color='#555555', linestyle='--', linewidth=1.0, label='1:1 Line')
    slope_fit_pct, intercept_fit_pct, _, _, _ = linregress(x_pct, y_pct)
    x_grid_pct = np.linspace(lims_pct[0], lims_pct[1], 100)
    ax_sc_pct.plot(x_grid_pct, intercept_fit_pct + slope_fit_pct*x_grid_pct, color='#d62728', linestyle='-', linewidth=1.2, label='Linear Fit')
    ax_sc_pct.set_xlabel('Annual Maxima Linear Trend (%)')
    ax_sc_pct.set_ylabel(r'GEV $\mathrm{RL}_2$ Trend (%)')
    ax_sc_pct.set_title(r'(d) Relative Trends')
    ax_sc_pct.grid(True, linestyle=':', alpha=0.5)
    ax_sc_pct.set_xlim(lims_pct)
    ax_sc_pct.set_ylim(lims_pct)
    ax_sc_pct.set_aspect('equal', adjustable='box')
    ax_sc_pct.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax_sc_pct.text(0.95, 0.05, f"$r$ = {r_pct:.3f}", transform=ax_sc_pct.transAxes,
                  verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                  fontsize=11, fontweight='bold')
    
    save_figure_atomically(fig, os.path.join(output_dir, "comparison_maxima_gev.png"), dpi=300)
    save_figure_atomically(fig, os.path.join(output_dir, "comparison_maxima_gev.pdf"), format='pdf')
    plt.close(fig)
    
    # ==========================================
    # 4. Figure B: Maps only (Side-by-side)
    # ==========================================
    print("Generating maps-only figure...")
    fig_maps, axs_maps = plt.subplots(1, 2, figsize=(12, 6.5), dpi=300)
    
    plot_style_map(axs_maps[0], stations_gdf, 'pct_full', '(a) Observed Annual Maxima Trend (%)')
    plot_style_map(axs_maps[1], stations_gdf, 'gev_trend_pct', r'(b) GEV $\mathrm{RL}_2$ Trend (%)')
    
    fig_maps.subplots_adjust(bottom=0.18, wspace=0.05)
    cbar_ax_maps = fig_maps.add_axes([0.25, 0.08, 0.5, 0.03])
    cbar_maps = fig_maps.colorbar(sm, cax=cbar_ax_maps, orientation='horizontal')
    cbar_maps.set_label('Relative Trend over 1960–2022 (%)')
    
    save_figure_atomically(fig_maps, os.path.join(output_dir, "comparison_maxima_gev_maps.png"), dpi=300)
    save_figure_atomically(fig_maps, os.path.join(output_dir, "comparison_maxima_gev_maps.pdf"), format='pdf')
    plt.close(fig_maps)
    
    # ==========================================
    # 5. Figure C: Scatter plots only (Side-by-side)
    # ==========================================
    print("Generating scatter-plots-only figure...")
    fig_sc, axs_sc = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    
    ax_sc_abs = axs_sc[0]
    ax_sc_abs.scatter(x_abs, y_abs, color='#1f77b4', alpha=0.4, edgecolors='none', s=15, label='Stations')
    ax_sc_abs.plot(lims_abs, lims_abs, color='#555555', linestyle='--', linewidth=1.0, label='1:1 Line')
    ax_sc_abs.plot(x_grid, intercept_fit + slope_fit*x_grid, color='#d62728', linestyle='-', linewidth=1.2, label='Linear Fit')
    ax_sc_abs.set_xlabel('Annual Maxima Linear Trend (mm/decade)')
    ax_sc_abs.set_ylabel(r'GEV $\mathrm{RL}_2$ Trend (mm/decade)')
    ax_sc_abs.set_title('(a) Absolute Trends')
    ax_sc_abs.grid(True, linestyle=':', alpha=0.5)
    ax_sc_abs.set_xlim(lims_abs)
    ax_sc_abs.set_ylim(lims_abs)
    ax_sc_abs.set_aspect('equal', adjustable='box')
    ax_sc_abs.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax_sc_abs.text(0.95, 0.05, f"$r$ = {r_abs:.3f}", transform=ax_sc_abs.transAxes,
                  verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                  fontsize=11, fontweight='bold')
    
    ax_sc_pct = axs_sc[1]
    ax_sc_pct.scatter(x_pct, y_pct, color='#2ca02c', alpha=0.4, edgecolors='none', s=15, label='Stations')
    ax_sc_pct.plot(lims_pct, lims_pct, color='#555555', linestyle='--', linewidth=1.0, label='1:1 Line')
    ax_sc_pct.plot(x_grid_pct, intercept_fit_pct + slope_fit_pct*x_grid_pct, color='#d62728', linestyle='-', linewidth=1.2, label='Linear Fit')
    ax_sc_pct.set_xlabel('Annual Maxima Linear Trend (%)')
    ax_sc_pct.set_ylabel(r'GEV $\mathrm{RL}_2$ Trend (%)')
    ax_sc_pct.set_title(r'(b) Relative Trends')
    ax_sc_pct.grid(True, linestyle=':', alpha=0.5)
    ax_sc_pct.set_xlim(lims_pct)
    ax_sc_pct.set_ylim(lims_pct)
    ax_sc_pct.set_aspect('equal', adjustable='box')
    ax_sc_pct.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax_sc_pct.text(0.95, 0.05, f"$r$ = {r_pct:.3f}", transform=ax_sc_pct.transAxes,
                  verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                  fontsize=11, fontweight='bold')
    
    save_figure_atomically(fig_sc, os.path.join(output_dir, "comparison_maxima_gev_scatter.png"), dpi=300)
    save_figure_atomically(fig_sc, os.path.join(output_dir, "comparison_maxima_gev_scatter.pdf"), format='pdf')
    plt.close(fig_sc)
    
    print("All figures saved successfully (excluding Corsica with shared size scale).")

if __name__ == "__main__":
    main()
