
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import geopandas as gpd

# Configuration
INPUT_ROOT = Path("data/statisticals/observed/horaire")
META_PATH = Path("data/metadonnees/observed/postes_horaire.csv")
RELIEF_PATH = Path("data/external/niveaux/selection_courbes_niveau_france.shp")
OUTPUT_DIR = Path("add_fig")
OUTPUT_DIR.mkdir(exist_ok=True)

MONTHS = ["jan", "fev", "mar"]
YEARS = range(1990, 2023)

def load_jfm_data():
    data = []
    print("Loading precipitation data...")
    for year in YEARS:
        for m in MONTHS:
            p = INPUT_ROOT / str(year) / f"{m}.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(p, columns=["NUM_POSTE", "max_mm_h"])
                    df["month"] = m
                    df["year"] = year
                    data.append(df)
                except Exception as e:
                    print(f"Error loading {p}: {e}")
    return pd.concat(data) if data else pd.DataFrame()

def main():
    # 1. Load Data
    df_all = load_jfm_data()
    if df_all.empty:
        print("No data found!")
        return

    # 2. Find dominant month per station
    df_pivot = df_all.drop_duplicates(["NUM_POSTE", "year", "month"]).pivot(
        index=["NUM_POSTE", "year"], columns="month", values="max_mm_h"
    )
    df_pivot = df_pivot.dropna(subset=MONTHS)
    df_pivot["winner"] = df_pivot[MONTHS].idxmax(axis=1)
    
    # Mapping for categories
    month_to_int = {"jan": 0, "fev": 1, "mar": 2}
    df_pivot["winner_int"] = df_pivot["winner"].map(month_to_int)
    
    station_stats = df_pivot.groupby("NUM_POSTE")["winner_int"].agg(lambda x: x.value_counts().idxmax()).reset_index()
    station_stats.columns = ["NUM_POSTE", "dominant_month_int"]
    
    # 3. Load Metadata and Geographize
    print(f"Loading metadata from {META_PATH}...")
    df_meta = pd.read_csv(META_PATH, usecols=["NUM_POSTE", "lat", "lon"])
    
    print(f"Type NUM_POSTE station_stats: {station_stats['NUM_POSTE'].dtype}")
    print(f"Type NUM_POSTE df_meta: {df_meta['NUM_POSTE'].dtype}")
    
    # Force both to string to be safe
    station_stats["NUM_POSTE"] = station_stats["NUM_POSTE"].astype(str)
    df_meta["NUM_POSTE"] = df_meta["NUM_POSTE"].astype(str)
    
    final_df = station_stats.merge(df_meta, on="NUM_POSTE", how="inner")
    print(f"Rows after merge: {len(final_df)}")
    
    if final_df.empty:
        print("Final DF is empty after merge!")
        return

    gdf = gpd.GeoDataFrame(
        final_df, 
        geometry=gpd.points_from_xy(final_df.lon, final_df.lat),
        crs="EPSG:4326"
    ).to_crs("EPSG:2154")
    print(f"Rows in GDF before clip: {len(gdf)}")

    # 4. Map Preparation (Project Style)
    print("Preparing map background...")
    deps = gpd.read_file("https://france-geojson.gregoiredavid.fr/repo/departements.geojson").to_crs("EPSG:2154")
    deps_metro = deps[~deps["code"].isin(["2A", "2B"])].copy()
    deps_metro["geometry"] = deps_metro.geometry.simplify(500)
    mask = deps_metro.union_all()
    
    # Clip stations to France
    gdf = gdf.clip(mask)
    print(f"Rows in GDF after clip: {len(gdf)}")

    # Relief
    if RELIEF_PATH.exists():
        relief = gpd.read_file(RELIEF_PATH).to_crs("EPSG:2154").clip(mask)
        relief["geometry"] = relief.geometry.simplify(500)
    else:
        relief = None

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Coastline
    if mask.geom_type == "MultiPolygon":
        polys = list(mask.geoms)
    else:
        polys = [mask]
    exteriors = [poly.exterior for poly in polys]
    coast = gpd.GeoSeries(exteriors, crs=deps_metro.crs)
    coast[coast.length > 2000].plot(ax=ax, edgecolor="black", linewidth=0.6, zorder=1)

    # Discrete Colormap
    # Jan: Blueish, Feb: Greenish, Mar: Orangish (consistent with previous choice but discrete)
    colors = ["#2b6cb0", "#48bb78", "#f6ad55"] 
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)

    # Stations
    gdf.plot(
        ax=ax, 
        column="dominant_month_int", 
        cmap=cmap, 
        norm=norm, 
        markersize=15, 
        edgecolor="black", 
        linewidth=0.1, 
        zorder=3
    )

    # Relief
    if relief is not None:
        relief.plot(ax=ax, color="#000000", linewidth=0.3, alpha=0.5, zorder=5)

    ax.set_axis_off()
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='January', markerfacecolor=colors[0], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='February', markerfacecolor=colors[1], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='March', markerfacecolor=colors[2], markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='upper left', title="Dominant month for JFM max", fontsize=10, title_fontsize=11)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "map_jfm_dominant_month_standard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches='tight')
    print(f"Standardized map saved to {output_path}")

if __name__ == "__main__":
    main()
