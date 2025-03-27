import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def get_stat_column_name(stat_key: str, scale_key: str) -> str:
    if stat_key == "mean":
        return f"mean_{scale_key}"
    elif stat_key == "max":
        return f"max_all_{scale_key}"
    elif stat_key == "mean-max":
        return f"max_mean_{scale_key}"
    elif stat_key == "date":
        return f"date_max_{scale_key[-1]}"  # "h" ou "j"
    elif stat_key == "month":
        return f"mois_pluvieux_{scale_key[-1]}"  # "h" ou "j"
    elif stat_key == "numday":
        return "jours_pluie_moyen"
    else:
        raise ValueError(f"Statistique inconnue : {stat_key}")

def get_stat_unit(stat_key: str, scale_key: str) -> str:
    if stat_key in ["mean", "max", "mean-max"]:
        if scale_key == "mm_h":
            return f"mm/h"
        elif scale_key == "mm_j":
            return f"mm/j"
        else:
            return "ERROR"
    elif stat_key == "sum":
        return "mm"
    elif stat_key == "numday":
        return "jours"
    else:
        return ""
    
def formalised_legend(df, column_to_show, colormap, vmin=None, vmax=None):
    if pd.api.types.is_datetime64_any_dtype(df[column_to_show]) or column_to_show.startswith("date_"):
        # Convertir en datetime si ce n'est pas déjà le cas
        df[column_to_show] = pd.to_datetime(df[column_to_show])
        vmin = pd.to_datetime(df[column_to_show].min()) if vmin is None else pd.to_datetime(vmin)
        vmax = pd.to_datetime(df[column_to_show].max()) if vmax is None else pd.to_datetime(vmax)

        # Transformation en nombre de secondes pour normaliser
        value_norm = (df[column_to_show] - vmin).dt.total_seconds() / (vmax - vmin).total_seconds()
        val_fmt_func = lambda x: x.strftime("%Y-%m-%d")
    else:
        vmin = df[column_to_show].min() if vmin is None else vmin
        vmax = df[column_to_show].max() if vmax is None else vmax
        value_norm = (df[column_to_show] - vmin) / (vmax - vmin)
        val_fmt_func = lambda x: f"{x:.1f}"

    value_norm = value_norm.clip(0, 1).fillna(0)

    df["fill_color"] = value_norm.apply(
        lambda v: [int(255 * c) for c in colormap(v)[:3]] + [255]  # RGBA
    )
    df["val_fmt"] = df[column_to_show].map(val_fmt_func)

    return df, vmin, vmax


def display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=5, label=""):
    gradient = np.linspace(1, 0, 256).reshape(256, 1)
    fig, ax = plt.subplots(figsize=(1, 5), dpi=100)
    ax.imshow(gradient, aspect='auto', cmap=colormap)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    base64_img = base64.b64encode(buf.getvalue()).decode()

    # Corriger la génération des ticks pour datetime
    if isinstance(vmin, pd.Timestamp) and isinstance(vmax, pd.Timestamp):
        ticks_seconds = np.linspace(vmax.timestamp(), vmin.timestamp(), n_ticks)
        ticks = [pd.to_datetime(t, unit='s').strftime("%Y-%m-%d") for t in ticks_seconds]
    else:
        ticks = np.linspace(vmax, vmin, n_ticks)
        ticks = [f"{tick:.1f}" for tick in ticks]

    st.markdown(
        f"""
        <div style="text-align: left; font-size: 13px;">
            <b>{label}</b>
        </div>
        <div style="display: flex; flex-direction: row; align-items: center; height: {height-30}px;">
            <img src="data:image/png;base64,{base64_img}" style="height: 100%; width: 20px; border: 1px solid #ccc; border-radius: 5px;"/>
            <div style="display: flex; flex-direction: column; justify-content: space-between; margin-left: 8px; height: 100%; font-size: 12px;">
                {''.join(f'<div>{tick}</div>' for tick in ticks)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
