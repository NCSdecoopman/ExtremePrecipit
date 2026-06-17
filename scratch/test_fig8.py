import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

# Legend styling arguments for clean, boxed legends matching LaTeX style
legend_opts = dict(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1.0,
    fancybox=False,
    fontsize=9
)

# ---------------------------------------------
# Panel (a): Mean GEV relative trend (%)
# ---------------------------------------------
# 4 bars per month: Daily Obs, Daily AROME, Hourly Obs, Hourly AROME
ax1.bar(x - 1.5*width_4, df_daily["mean_obs"], width=width_4, color="black", label="Daily Stations")
ax1.bar(x - 0.5*width_4, df_daily["mean_mod"], width=width_4, color="dimgray", label="Daily AROME")
ax1.bar(x + 0.5*width_4, df_hourly["mean_obs"], width=width_4, color="gray", label="Hourly Stations")
ax1.bar(x + 1.5*width_4, df_hourly["mean_mod"], width=width_4, color="lightgray", label="Hourly AROME")

ax1.set_ylabel("Mean relative trend (%)")
ax1.set_ylim(-30, 260) # Leave headroom at the top for the legend
ax1.grid(axis="y", which="major", linestyle="--", alpha=0.8)
ax1.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
ax1.set_title("(a) Monthly mean GEV relative trend (stations vs AROME)", loc="left")
ax1.legend(ncol=2, **legend_opts)

# ---------------------------------------------
# Panel (b): Bias in relative trend (%)
# ---------------------------------------------
# 2 bars per month: Daily Bias, Hourly Bias
ax2.bar(x - 0.5*width_2, df_daily["me"], width=width_2, color="dimgray", label="Daily Bias")
ax2.bar(x + 0.5*width_2, df_hourly["me"], width=width_2, color="lightgray", label="Hourly Bias")

ax2.set_ylabel("Bias in relative trend (%)")
ax2.set_ylim(-230, 50) # Leave headroom at the top for the legend
ax2.grid(axis="y", which="major", linestyle="--", alpha=0.8)
ax2.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
ax2.set_title("(b) Monthly mean GEV trend bias (AROME - stations)", loc="left")
ax2.legend(ncol=2, **legend_opts)

# ---------------------------------------------
# Panel (c): Spatial correlation (r)
# ---------------------------------------------
# 2 bars per month: Daily Correlation, Hourly Correlation
ax3.bar(x - 0.5*width_2, df_daily["r"], width=width_2, color="dimgray", label="Daily ($r$)")
ax3.bar(x + 0.5*width_2, df_hourly["r"], width=width_2, color="lightgray", label="Hourly ($r$)")

ax3.set_ylabel("Spatial correlation ($r$)")
ax3.set_ylim(-0.15, 0.95) # Leave headroom at the top for the legend
ax3.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax3.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax3.grid(axis="y", which="major", linestyle="--", alpha=0.8)
ax3.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
ax3.set_title("(c) Monthly spatial correlation of GEV trends", loc="left")
ax3.legend(ncol=2, **legend_opts)

# Shared x-axis ticks
plt.xticks(x, months_en)

# Adjust layout
plt.tight_layout()
plt.savefig("scratch/fig8_test.pdf", bbox_inches="tight")
print("Saved scratch/fig8_test.pdf successfully!")
