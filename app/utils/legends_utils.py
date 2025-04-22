import streamlit as st
from io import BytesIO
import base64
import polars as pl
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

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
        # Conversion correcte en datetime (Polars)
        df = df.with_columns(
            pl.col(column_to_show).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.6f", strict=False)
        )

        # Récupération min/max en datetime Python natif
        min_dt = df[column_to_show].min()
        max_dt = df[column_to_show].max()

        if isinstance(min_dt, dt.date):
            min_dt = dt.datetime.combine(min_dt, dt.time.min)
        if isinstance(max_dt, dt.date):
            max_dt = dt.datetime.combine(max_dt, dt.time.min)

        vmin = min_dt if vmin is None else vmin
        vmax = max_dt if vmax is None else vmax

        # Gestion safe des timestamps sur Windows (pré-1970)
        def safe_timestamp(d):
            epoch = dt.datetime(1970, 1, 1)
            return (d - epoch).total_seconds()

        vmin_ts = safe_timestamp(vmin)
        vmax_ts = safe_timestamp(vmax)

        # Ajout de la colonne normalisée dans Polars
        df = df.with_columns([
            ((pl.col(column_to_show).cast(pl.Datetime).dt.timestamp() - vmin_ts) / (vmax_ts - vmin_ts))
            .clip(0.0, 1.0)
            .alias("value_norm")
        ])

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
    # Étape 1 : extraire les valeurs (en NumPy)
    vals = df["value_norm"].to_numpy()

    # Étape 2 : appliquer le colormap sur tout le tableau (résultat : Nx4 array RGBA)
    colors = (255 * np.array(colormap(vals))[:, :3]).astype(np.uint8)

    # Étape 3 : ajouter l'alpha (255)
    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    rgba = np.hstack([colors, alpha])

    # Étape 4 : réinjecter dans Polars
    fill_color = pl.Series("fill_color", rgba.tolist(), dtype=pl.List(pl.UInt8))

    df = df.with_columns([
        pl.Series("fill_color", fill_color),
        df[column_to_show].map_elements(val_fmt_func, return_dtype=pl.String).alias("val_fmt"), # val_fmt optimisé si float
        pl.col("lat").round(3).cast(pl.Utf8).alias("lat_fmt"),
        pl.col("lon").round(3).cast(pl.Utf8).alias("lon_fmt")
    ])

    return df, vmin, vmax

def display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=5, label=""):
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
        return html_mois

    gradient = np.linspace(1, 0, 64).reshape(64, 1)
    fig, ax = plt.subplots(figsize=(1, 3), dpi=30)
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

    html_gradient = f"""
        <div style="text-align: left; font-size: 13px;">{label}</div>
        <div style="display: flex; flex-direction: row; align-items: center; height: {height-30}px;">
            <img src="data:image/png;base64,{base64_img}"
                 style="height: 100%; width: 20px; border: 1px solid #ccc; border-radius: 5px;"/>
            <div style="display: flex; flex-direction: column; justify-content: space-between; 
                        margin-left: 8px; height: 100%; font-size: 12px;">
                {''.join(f'<div>{tick}</div>' for tick in ticks)}
            </div>
        </div>
    """
    return html_gradient
