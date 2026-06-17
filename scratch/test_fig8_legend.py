import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import os

# Set font styling matching fig4.qmd
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

# Path to the metrics CSVs in outputs_nr10
base_dir = Path("outputs_nr10")

daily_path = base_dir / "maps/gev_z_T_p/quotidien/compare_12/sat_99.0/metrics_signif.csv"
hourly_path = base_dir / "maps/gev_z_T_p/horaire/compare_12/sat_90.0/metrics_signif.csv"

# Load dataframes
df_daily = pd.read_csv(daily_path)
df_hourly = pd.read_csv(hourly_path)

# Months order mapping
months_fr = ["jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec"]
months_en = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

df_daily["season"] = df_daily["season"].str.lower()
df_hourly["season"] = df_hourly["season"].str.lower()

# Reindex to force chronological month order
df_daily = df_daily.set_index("season").reindex(months_fr).reset_index()
df_hourly = df_hourly.set_index("season").reindex(months_fr).reset_index()

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

x = np.arange(len(months_en))
width_4 = 0.2  # bar width for 4 bars
width_2 = 0.35 # bar width for 2 bars

# Legend styling arguments for clean, boxed legends matching LaTeX style, placed above the subplots
legend_opts_a = dict(
    loc="lower right",
    bbox_to_anchor=(1.0, 1.02),
    ncol=4,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1.0,
    fancybox=False,
    fontsize=9
)

legend_opts_bc = dict(
    loc="lower right",
    bbox_to_anchor=(1.0, 1.02),
    ncol=2,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1.0,
    fancybox=False,
    fontsize=9
)

# ---------------------------------------------
# Panel (a): Median GEV relative trend (%)
# ---------------------------------------------
ax1.bar(x - 1.5*width_4, df_daily["mean_obs"], width=width_4, color="black", label="Daily Stations")
ax1.bar(x - 0.5*width_4, df_daily["mean_mod"], width=width_4, color="dimgray", label="Daily AROME")
ax1.bar(x + 0.5*width_4, df_hourly["mean_obs"], width=width_4, color="gray", label="Hourly Stations")
ax1.bar(x + 1.5*width_4, df_hourly["mean_mod"], width=width_4, color="lightgray", label="Hourly AROME")

ax1.set_ylabel("Median GEV relative trend (%)")
ax1.set_ylim(-40, 150)
ax1.grid(axis="y", which="major", linestyle="--", alpha=0.8)
ax1.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
ax1.set_title("(a) Monthly median GEV relative trend (stations vs AROME)", loc="left")
ax1.legend(**legend_opts_a)

# ---------------------------------------------
# Panel (b): Median GEV trend bias (%)
# ---------------------------------------------
ax2.bar(x - 0.5*width_2, df_daily["me"], width=width_2, color="dimgray", label="Daily Bias")
ax2.bar(x + 0.5*width_2, df_hourly["me"], width=width_2, color="lightgray", label="Hourly Bias")

ax2.set_ylabel("Median GEV trend bias (%)")
ax2.set_ylim(-110, 40)
ax2.grid(axis="y", which="major", linestyle="--", alpha=0.8)
ax2.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
ax2.set_title("(b) Monthly median GEV trend bias (AROME - stations)", loc="left")
ax2.legend(**legend_opts_bc)

# ---------------------------------------------
# Panel (c): Spatial correlation (r)
# ---------------------------------------------
ax3.bar(x - 0.5*width_2, df_daily["r"], width=width_2, color="dimgray", label="Daily ($r$)")
ax3.bar(x + 0.5*width_2, df_hourly["r"], width=width_2, color="lightgray", label="Hourly ($r$)")

ax3.set_ylabel("Spatial correlation ($r$)")
ax3.set_ylim(-0.15, 0.95)
ax3.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax3.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax3.grid(axis="y", which="major", linestyle="--", alpha=0.8)
ax3.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
ax3.set_title("(c) Monthly spatial correlation of GEV trends", loc="left")
ax3.legend(**legend_opts_bc)

# Shared x-axis ticks
plt.xticks(x, months_en)

# Adjust layout and save PNG to check
plt.tight_layout()
# Let's add extra spacing to prevent legend overlap
fig.subplots_adjust(hspace=0.35)

out_dir = r"C:\Users\nicod\Documents\GitHub\ExtremePrecipit\scratch"
fig.savefig(os.path.join(out_dir, "test_fig8.png"), bbox_inches="tight")
plt.close(fig)
print("Test figure generated successfully in scratch/test_fig8.png")
