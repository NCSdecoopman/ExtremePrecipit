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
SEASONS = ["ond", "jfm", "amj", "jas", "hydro"]
SEASON_LABELS = ["OND", "JFM", "AMJ", "JAS", "YEAR"]

def get_significance_pct(source, echelle, periods):
    results = {}
    base_path = GEV_ROOT / source / echelle
    for s in periods:
        p = base_path / s / "niveau_retour.parquet"
        label = s.upper() if s != "hydro" else "YEAR"
        if p.exists():
            try:
                df = pl.read_parquet(p)
                if "significant" not in df.columns:
                    results[label] = 0.0
                    continue
                df = df.unique(subset=["NUM_POSTE"])
                total = len(df)
                if total == 0:
                    results[label] = 0.0
                    continue
                significant = df.filter(pl.col("significant") == True).height
                results[label] = (significant / total) * 100
            except Exception as e:
                print(f"Error reading {p}: {e}")
                results[label] = 0.0
        else:
            results[label] = 0.0
    return results

def plot_panel(ax, pcts_full, pcts_restricted, labels, title, y_label="", rotation=0):
    pos = np.arange(len(labels))
    width = 0.35
    
    vals_f = [pcts_full.get(s, 0) if pcts_full else 0 for s in labels]
    vals_r = [pcts_restricted.get(s, 0) if pcts_restricted else 0 for s in labels]
    
    ax.bar(pos - width/2, vals_f, width=width, color="black", label="Complete series (1959-2022)")
    ax.bar(pos + width/2, vals_r, width=width, color="lightgray", label="Restricted period (1990-2022)")
    
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_title(title)
    if y_label:
        ax.set_ylabel(y_label)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on top
    for i, v in enumerate(vals_f):
        if pcts_full:
            ax.text(i - width/2, v + 1, f'{v:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
    for i, v in enumerate(vals_r):
        if pcts_restricted:
            ax.text(i + width/2, v + 1, f'{v:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

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
            ax = axes[row, col]
            
            # Special case for Hourly Stations: only restricted period exists
            if echelle == "horaire" and source_key == "observed":
                pcts_full = None
                pcts_restricted = get_significance_pct(source_key, echelle, periods)
            else:
                pcts_full = get_significance_pct(source_key, echelle, periods)
                pcts_restricted = get_significance_pct(source_key, f"{echelle}_reduce", periods)
            
            y_label = f"Significant {label_echelle} points (%)" if col == 0 else ""
            plot_panel(ax, pcts_full, pcts_restricted, labels, source_name, y_label, rotation=rotation)

    # Global legend
    legend_elements = [
        Patch(facecolor="black", label="Complete series (1959-2022)"),
        Patch(facecolor="lightgray", label="Restricted period (1990-2022)")
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Significance panels saved to {output_path}")

if __name__ == "__main__":
    # 1. Seasons
    run_plot(
        periods=["ond", "jfm", "amj", "jas", "hydro"],
        labels=["OND", "JFM", "AMJ", "JAS", "YEAR"],
        output_filename="significance_comparison_seasons.pdf",
        rotation=0
    )
    
    # 2. Months
    run_plot(
        periods=["jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec"],
        labels=["JAN", "FEV", "MAR", "AVR", "MAI", "JUI", "JUILL", "AOU", "SEP", "OCT", "NOV", "DEC"],
        output_filename="significance_comparison_months.pdf",
        fig_width=24,
        rotation=45
    )
