import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64
import polars as pl
import datetime as dt

def convert_custom_colorscale(custom_colorscale):
    positions, colors = zip(*custom_colorscale)
    return LinearSegmentedColormap.from_list("custom_colormap", list(zip(positions, colors)))

def get_stat_column_name(stat_key: str, scale_key: str) -> str:
    if stat_key == "mean":
        return f"mean_all_{scale_key}"
    elif stat_key == "max":
        return f"max_all_{scale_key}"
    elif stat_key == "mean-max":
        return f"max_mean_{scale_key}"
    elif stat_key == "date":
        return "date_max_h" if scale_key == "mm_h" else "date_max_j"
    elif stat_key == "month":
        return "mois_pluvieux_h" if scale_key == "mm_h" else "mois_pluvieux_j"
    elif stat_key == "numday":
        return "jours_pluie_moyen"
    else:
        raise ValueError(f"Statistique inconnue : {stat_key}")

def get_stat_unit(stat_key: str, scale_key: str) -> str:
    if stat_key in ["mean", "max", "mean-max"]:
        return "mm/h" if scale_key == "mm_h" else "mm/j"
    elif stat_key == "sum":
        return "mm"
    elif stat_key == "numday":
        return "jours"
    else:
        return ""

def formalised_legend(df: pl.DataFrame, column_to_show: str, colormap, vmin=None, vmax=None):
    df = df.clone()
    column = df[column_to_show]

    if column_to_show.startswith("date"):
        df = df.with_columns(pl.col(column_to_show).str.strptime(pl.Datetime, fmt="%Y-%m-%d"))
        vmin = df[column_to_show].min() if vmin is None else pl.datetime(vmin)
        vmax = df[column_to_show].max() if vmax is None else pl.datetime(vmax)

        vmin_ts = vmin.timestamp()
        vmax_ts = vmax.timestamp()
        value_norm = df[column_to_show].cast(pl.Datetime).dt.timestamp().alias("value_norm")
        value_norm = ((value_norm - vmin_ts) / (vmax_ts - vmin_ts)).clip(0.0, 1.0)
        df = df.with_columns(value_norm)

        val_fmt_func = lambda x: x.strftime("%Y-%m-%d")

    elif column_to_show.startswith("mois_pluvieux"):
        df = df.with_columns(pl.col(column_to_show).cast(pl.Int32))
        value_norm = ((df[column_to_show] - 1) / 11).clip(0.0, 1.0)
        df = df.with_columns(value_norm.alias("value_norm"))

        mois_labels = [
            "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
            "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
        ]
        val_fmt_func = lambda x: mois_labels[int(x) - 1] if 1 <= int(x) <= 12 else "Inconnu"

        vmin, vmax = 1, 12

    else:
        vmin = 0 if vmin is None else vmin
        vmax = df[column_to_show].max() if vmax is None else vmax
        value_norm = ((df[column_to_show] - vmin) / (vmax - vmin)).clip(0.0, 1.0)
        df = df.with_columns(value_norm.alias("value_norm"))

        val_fmt_func = lambda x: f"{x:.2f}"

    # Application de la colormap
    fill_color = df["value_norm"].map_elements(
        lambda v: [int(255 * c) for c in colormap(v)[:3]] + [255],
        return_dtype=pl.List(pl.UInt8),
        skip_nulls=True  # ou False selon ton besoin
    )

    df = df.with_columns([
        pl.Series("fill_color", fill_color),
        df[column_to_show].map_elements(val_fmt_func, return_dtype=pl.String).alias("val_fmt"),
        df["lat"].map_elements(lambda x: f"{x:.3f}", return_dtype=pl.String).alias("lat_fmt"),
        df["lon"].map_elements(lambda x: f"{x:.3f}", return_dtype=pl.String).alias("lon_fmt")
    ])

    return df, vmin, vmax

def display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=5, label=""):
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(vmin, int) and isinstance(vmax, int) and (1 <= vmin <= 12) and (1 <= vmax <= 12):
        mois_labels = [
            "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
            "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
        ]
        color_boxes = ""
        for mois in range(vmin, vmax + 1):
            rgba = colormap((mois - 1) / 11)
            rgb = [int(255 * c) for c in rgba[:3]]
            color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
            label_mois = mois_labels[mois - 1]
            color_boxes += (
                f'<div style="display: flex; align-items: center; margin-bottom: 4px;">'
                f'  <div style="width: 14px; height: 14px; background-color: {color}; '
                f'border: 1px solid #ccc; margin-right: 6px;"></div>'
                f'  <div style="font-size: 12px;">{label_mois}</div>'
                f'</div>'
            )
        html_mois = (
            f'<div style="text-align: left; font-size: 13px; margin-bottom: 4px;">{label}</div>'
            f'<div style="display: flex; flex-direction: column;">{color_boxes}</div>'
        )
        st.markdown(html_mois, unsafe_allow_html=True)
        return

    gradient = np.linspace(1, 0, 256).reshape(256, 1)
    fig, ax = plt.subplots(figsize=(1, 5), dpi=100)
    ax.imshow(gradient, aspect='auto', cmap=colormap)
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    base64_img = base64.b64encode(buf.getvalue()).decode()

    if isinstance(vmin, dt.datetime) and isinstance(vmax, dt.datetime):
        ticks_seconds = np.linspace(vmax.timestamp(), vmin.timestamp(), n_ticks)
        ticks = [dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d") for t in ticks_seconds]
    else:
        ticks_vals = np.linspace(vmax, vmin, n_ticks)
        ticks = [f"{val:.1f}" for val in ticks_vals]

    st.markdown(
        f"""
        <div style="text-align: left; font-size: 13px;">{label}</div>
        <div style="display: flex; flex-direction: row; align-items: center; height: {height-30}px;">
            <img src="data:image/png;base64,{base64_img}"
                 style="height: 100%; width: 20px; border: 1px solid #ccc; border-radius: 5px;"/>
            <div style="display: flex; flex-direction: column; justify-content: space-between; 
                        margin-left: 8px; height: 100%; font-size: 12px;">
                {''.join(f'<div>{tick}</div>' for tick in ticks)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
