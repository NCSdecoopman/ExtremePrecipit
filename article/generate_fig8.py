import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# Set font styling matching fig4.qmd
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

COLOR_DAILY = "black"
COLOR_HOURLY = "dimgray"
LABEL_DAILY = "Daily (1959-2022)"
LABEL_HOURLY = "Hourly (1990-2022)"


def format_n_label(value: float) -> str:
    return f" {int(value)}"


def style_y_axis(ax, show_left_labels=True, show_right_labels=False):
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.8)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
    ax.tick_params(
        labelleft=show_left_labels,
        labelright=show_right_labels,
        left=True,
        right=show_right_labels,
    )


def style_panel_a_xaxis(ax, show_month_labels=True, rotation=45):
    ax.set_xlim(-0.5, len(months_en) - 0.5)
    ax.set_xticks(x)
    ha = "center" if rotation == 0 else "right"
    ax.set_xticklabels(
        months_en if show_month_labels else [],
        fontsize=8,
        rotation=rotation,
        ha=ha,
    )
    ax.xaxis.set_major_locator(mticker.FixedLocator(x))
    ax.tick_params(axis="x", which="major", length=3)


def add_bar_labels(ax, bars, values, label_fn, rotation=90):
    for bar, val in zip(bars, values):
        if pd.isna(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label_fn(val),
            rotation=rotation,
            ha="center",
            va="bottom",
            fontsize=8,
        )

# Path to the metrics CSVs in outputs_nr10
base_dir = Path("../outputs_nr10")
if not base_dir.exists():
    base_dir = Path("outputs_nr10")

daily_path = base_dir / "maps/gev_z_T_p/quotidien/compare_12/sat_99.0/metrics_signif.csv"
hourly_path = base_dir / "maps/gev_z_T_p/horaire/compare_12/sat_90.0/metrics_signif.csv"

df_daily = pd.read_csv(daily_path)
df_hourly = pd.read_csv(hourly_path)

months_fr = ["jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec"]
months_en = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

df_daily["season"] = df_daily["season"].str.lower()
df_hourly["season"] = df_hourly["season"].str.lower()

df_daily = df_daily.set_index("season").reindex(months_fr).reset_index()
df_hourly = df_hourly.set_index("season").reindex(months_fr).reset_index()

x = np.arange(len(months_en))
width_2 = 0.35
width_a = 0.28

fig = plt.figure(figsize=(12, 11))
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1.05, 1, 1],
    hspace=0.50,
    wspace=0.18,
    top=0.94,
    bottom=0.06,
    left=0.08,
    right=0.98,
)

ax_stations = fig.add_subplot(gs[0, 0])
ax_arome = fig.add_subplot(gs[0, 1], sharey=ax_stations)
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])

# Panel (a): Stations | AROME
ax_stations.bar(x - 0.5 * width_a, df_daily["mean_obs"], width=width_a, color=COLOR_DAILY)
ax_stations.bar(x + 0.5 * width_a, df_hourly["mean_obs"], width=width_a, color=COLOR_HOURLY)
ax_stations.set_ylabel("Median GEV relative trend (%)")
ax_stations.set_ylim(-50, 115)
style_y_axis(ax_stations, show_left_labels=True, show_right_labels=False)
style_panel_a_xaxis(ax_stations, show_month_labels=True)
ax_stations.set_title("(a) Monthly median GEV relative trend", loc="left", pad=10)
ax_stations.text(
    0.03, 0.97, "Stations",
    transform=ax_stations.transAxes,
    ha="left", va="top",
    fontsize=11,
)

ax_arome.bar(x - 0.5 * width_a, df_daily["mean_mod"], width=width_a, color=COLOR_DAILY)
ax_arome.bar(x + 0.5 * width_a, df_hourly["mean_mod"], width=width_a, color=COLOR_HOURLY)
style_y_axis(ax_arome, show_left_labels=False, show_right_labels=True)
ax_arome.yaxis.tick_right()
style_panel_a_xaxis(ax_arome, show_month_labels=True)
ax_arome.text(
    0.03, 0.97, "AROME",
    transform=ax_arome.transAxes,
    ha="left", va="top",
    fontsize=11,
)

# Panel (b): Mean error (ME) between relative trends
me_min = min(df_daily["me"].min(), df_hourly["me"].min())
me_max = max(df_daily["me"].max(), df_hourly["me"].max())
ax2.bar(x - 0.5 * width_2, df_daily["me"], width=width_2, color=COLOR_DAILY)
ax2.bar(x + 0.5 * width_2, df_hourly["me"], width=width_2, color=COLOR_HOURLY)
ax2.set_ylabel("Mean error (ME) (%)")
ax2.set_ylim(me_min - 20, me_max + 20)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(25))
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(5))
ax2.grid(axis="y", which="major", linestyle="--", alpha=0.8)
ax2.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
ax2.set_title("(b) Mean error (ME) between AROME and Météo-France stations", loc="left")
style_panel_a_xaxis(ax2, show_month_labels=True, rotation=0)

# Panel (c): Spatial correlation (r)
bars_daily = ax3.bar(x - 0.5 * width_2, df_daily["r"], width=width_2, color=COLOR_DAILY)
bars_hourly = ax3.bar(x + 0.5 * width_2, df_hourly["r"], width=width_2, color=COLOR_HOURLY)
add_bar_labels(ax3, bars_daily, df_daily["n"], format_n_label)
add_bar_labels(ax3, bars_hourly, df_hourly["n"], format_n_label)
ax3.set_ylabel("Spatial correlation ($r$)")
ax3.set_ylim(0, 0.75)
ax3.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax3.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
ax3.grid(axis="y", which="major", linestyle="--", alpha=0.8)
ax3.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
ax3.set_title("(c) Monthly spatial correlation of GEV relative trends", loc="left")
ax3.set_xlim(-0.5, len(months_en) - 0.5)
ax3.set_xticks(x)
ax3.set_xticklabels(months_en)
ax3.xaxis.set_major_locator(mticker.FixedLocator(x))

legend_elements = [
    Patch(facecolor=COLOR_DAILY, label=LABEL_DAILY),
    Patch(facecolor=COLOR_HOURLY, label=LABEL_HOURLY),
]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.01),
    frameon=True,
)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/histo_z_T_p.pdf", bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print("Figure 8 PDF component generated successfully in figures/histo_z_T_p.pdf")
