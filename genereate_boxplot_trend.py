
import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path

# Force non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Configuration
GEV_ROOT = Path("data/gev")
OUTPUT_DIR = Path("add_fig")
SEASONS = ["ond", "jfm", "amj", "jas", "hydro"]
SEASON_LABELS = ["OND", "JFM", "AMJ", "JAS", "YEAR"]
TREND_COL = "z_T_p"

def load_trends(source, echelle):
    all_data = []
    base_path = GEV_ROOT / source / echelle
    for s in SEASONS:
        p = base_path / s / "niveau_retour.parquet"
        if p.exists():
            try:
                df = pl.read_parquet(p)
                # Filter for significant trends
                if "significant" not in df.columns:
                    print(f"Warning: 'significant' column missing in {p}")
                    continue
                
                # Check for duplicates or multiple rows per station
                if df["NUM_POSTE"].n_unique() < len(df):
                    # If there are duplicates, we might be loading multiple models or return levels
                    # Let's take the first one or filter if needed. 
                    # Usually niveau_retour.parquet from best_model should be unique per station.
                    df = df.unique(subset=["NUM_POSTE"])

                df_sig = df.filter(pl.col("significant") == True)
                if df_sig.is_empty():
                    continue
                trends = df_sig[TREND_COL].to_numpy()
                season_label = s.upper() if s != "hydro" else "YEAR"
                for t in trends:
                    all_data.append({
                        "Season": season_label,
                        "Trend": t,
                        "Type": "Full" if echelle == "quotidien" else "Restricted"
                    })
            except Exception as e:
                print(f"Error reading {p}: {e}")
                continue
        else:
            print(f"Warning: {p} not found")
    return pd.DataFrame(all_data)

def plot_robustness(ax, source_name):
    print(f"Processing {source_name}...")
    source_key = "observed" if source_name == "STATIONS" else "modelised"
    df_full = load_trends(source_key, "quotidien")
    df_reduce = load_trends(source_key, "quotidien_reduce")
    
    print(f"  {source_name} - Full: {len(df_full)} rows, Reduce: {len(df_reduce)} rows")
    
    if df_full.empty and df_reduce.empty:
        print(f"Warning: No data found for {source_name}.")
        return False

    pos = np.arange(len(SEASON_LABELS))
    width = 0.35
    
    data_f = [df_full[df_full["Season"] == s]["Trend"] if not df_full.empty else [] for s in SEASON_LABELS]
    data_r = [df_reduce[df_reduce["Season"] == s]["Trend"] if not df_reduce.empty else [] for s in SEASON_LABELS]
    
    # Colors matching fig4/fig8 style (Black vs Light Gray)
    box_f = ax.boxplot(
        data_f,
        positions=pos - width / 2,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="black", color="black"),
        medianprops=dict(color="white", linewidth=1.5),
        showfliers=False,
    )
    box_r = ax.boxplot(
        data_r,
        positions=pos + width / 2,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", color="black"),
        medianprops=dict(color="black", linewidth=1.5),
        showfliers=False,
    )
    
    ax.set_xticks(pos)
    ax.set_xticklabels(SEASON_LABELS)
    ax.set_ylabel("Significant relative trends (%)")
    ax.set_title(source_name)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    # Add n counts above boxplots using whisker positions
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.02
    
    for i in range(len(SEASON_LABELS)):
        n_f = len(data_f[i])
        n_r = len(data_r[i])
        
        # Get whisker tops from the boxplot objects
        if n_f > 0:
            whisker_high_f = box_f['whiskers'][2*i + 1].get_ydata()[1]
            ax.text(i - width/2, whisker_high_f + y_offset, f"n={n_f}", 
                    ha="center", va="bottom", fontsize=8, color="black")
        
        if n_r > 0:
            whisker_high_r = box_r['whiskers'][2*i + 1].get_ydata()[1]
            ax.text(i + width/2, whisker_high_r + y_offset, f"n={n_r}", 
                    ha="center", va="bottom", fontsize=8, color="gray")
    
    return True

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    success_obs = plot_robustness(ax1, "STATIONS")
    success_mod = plot_robustness(ax2, "AROME")
    
    if not (success_obs or success_mod):
        print("Error: No data plotted.")
        return

    # Global legend
    legend_elements = [
        Patch(facecolor="black", label="Complete series (1959-2022)"),
        Patch(facecolor="lightgray", label="Restricted period (1990-2022)")
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    output_path = OUTPUT_DIR / "daily_robustness_comparison_dual.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Dual plot saved to {output_path}")

if __name__ == "__main__":
    run()
