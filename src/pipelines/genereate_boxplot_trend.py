import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# Force non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Configuration
GEV_ROOT = Path("data/gev")
OUTPUT_DIR = Path("add_fig")
TREND_COL = "z_T_p"

def load_trends(source, echelle, periods):
    all_data = []
    base_path = GEV_ROOT / source / echelle
    for s in periods:
        p = base_path / s / "niveau_retour.parquet"
        if p.exists():
            try:
                df = pl.read_parquet(p)
                if "significant" not in df.columns:
                    print(f"Warning: 'significant' column missing in {p}")
                    continue
                df = df.unique(subset=["NUM_POSTE"])
                df_sig = df.filter(pl.col("significant") == True)
                if df_sig.is_empty():
                    continue
                trends = df_sig[TREND_COL].to_numpy()
                
                # Label logic
                if s == "hydro":
                    label = "YEAR"
                else:
                    label = s.upper()
                
                # Type label based on timeframe (simplified logic)
                for t in trends:
                    all_data.append({
                        "Period": label,
                        "Trend": t
                    })
            except Exception as e:
                print(f"Error reading {p}: {e}")
        else:
            print(f"Warning: {p} not found")
    return pd.DataFrame(all_data)

def plot_robustness(ax, df_full, df_reduce, labels, rotation=0):
    pos = np.arange(len(labels))
    width = 0.35

    data_f = [df_full[df_full["Period"] == s]["Trend"] if not df_full.empty else [] for s in labels]
    data_r = [df_reduce[df_reduce["Period"] == s]["Trend"] if not df_reduce.empty else [] for s in labels]

    # Boxplots: Full (Black, 1959-2022) vs Reduce (Gray, 1990-2022)
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
    ax.set_xticklabels(labels, rotation=rotation)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Add n counts
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.02
    for i in range(len(labels)):
        n_f = len(data_f[i])
        n_r = len(data_r[i])
        if n_f > 0:
            whisker_high_f = box_f['whiskers'][2*i + 1].get_ydata()[1]
            ax.text(i - width/2, whisker_high_f + y_offset, f"n={n_f}", ha="center", va="bottom", fontsize=8, color="black")
        if n_r > 0:
            whisker_high_r = box_r['whiskers'][2*i + 1].get_ydata()[1]
            ax.text(i + width/2, whisker_high_r + y_offset, f"n={n_r}", ha="center", va="bottom", fontsize=8, color="gray")

def run_plot(periods, labels, output_filename, fig_width=18, rotation=0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    sources = ["Stations", "AROME"]
    echelles = ["quotidien", "horaire"]
    
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 12), sharey=True)
    
    nomenclature_echelle = {
        "quotidien": "daily",
        "horaire": "hourly"
    }
    
    for row, echelle in enumerate(echelles):
        label_echelle = nomenclature_echelle.get(echelle, echelle)
        for col, source_name in enumerate(sources):
            source_key = "observed" if source_name.upper() == "STATIONS" else "modelised"
            
            # Application de la logique de périodes
            if echelle == "quotidien":
                # Noir = 1959-2022 (dossier standard) | Gris = 1990-2022 (dossier _reduce)
                df_full = load_trends(source_key, "quotidien", periods)
                df_reduce = load_trends(source_key, "quotidien_reduce", periods)
            else:
                # Horaire: Noir = 1959-2022 (dossier _reduce) | Gris = 1990-2022 (dossier standard)
                df_full = load_trends(source_key, "horaire_reduce", periods)
                df_reduce = load_trends(source_key, "horaire", periods)
            
            print(f"Echelle: {echelle}, Source: {source_key}")
            print(f"  Full (1959-2022): {len(df_full)} points")
            print(f"  Reduce (1990-2022): {len(df_reduce)} points")

            ax = axes[row, col]
            plot_robustness(ax, df_full, df_reduce, labels, rotation=rotation)
            ax.set_title(f"{source_name}")
            if col == 0:
                ax.set_ylabel(f"Significant relative {label_echelle} trends (%)")

    # Global legend
    legend_elements = [
        Patch(facecolor="black", label="Full period (1959-2022)"),
        Patch(facecolor="lightgray", label="Restricted period (1990-2022)")
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # 1. Seasons
    run_plot(
        periods=["ond", "jfm", "amj", "jas", "hydro"],
        labels=["OND", "JFM", "AMJ", "JAS", "YEAR"],
        output_filename="robustness_comparison_seasons.pdf",
        rotation=0
    )
    
    # 2. Months
    run_plot(
        periods=["jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec"],
        labels=["JAN", "FEV", "MAR", "AVR", "MAI", "JUI", "JUILL", "AOU", "SEP", "OCT", "NOV", "DEC"],
        output_filename="robustness_comparison_months.pdf",
        fig_width=24,
        rotation=45
    )