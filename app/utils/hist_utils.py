import polars as pl
import pandas as pd
import plotly.express as px
import streamlit as st

def plot_histogramme(df: pl.DataFrame, var: str, stat: str, stat_key: str, unit: str, height: int):
    month_order = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                   'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']

    if stat_key == 'month':
        df = df.with_columns([
            pl.col(var).cast(pl.Int32).map_dict({i + 1: month_order[i] for i in range(12)}).alias(var)
        ])

        counts = df.select(var).to_series().value_counts().sort("")

        # Création d'une série avec tous les mois, remplis à 0 si absents
        freq_percent = {
            month: (counts.filter(pl.col(var) == month)["count"].sum() / counts["count"].sum() * 100).item()
            if month in counts[var].to_list() else 0.0
            for month in month_order
        }

        hist_df = pd.DataFrame({var: list(freq_percent.keys()), 'Pourcentage': list(freq_percent.values())})

        fig = px.bar(
            hist_df,
            x=var,
            y='Pourcentage'
        )
        fig.update_layout(
            bargap=0.1,
            xaxis_title="",
            yaxis_title="Pourcentage de stations",
            height=height,
            xaxis=dict(
                categoryorder='array',
                categoryarray=month_order
            )
        )
    else:
        # Conversion nécessaire car px.histogram ne gère que pandas
        df_pd = df.to_pandas()
        fig = px.histogram(
            df_pd,
            x=var,
            nbins=50,
            histnorm='percent'
        )
        fig.update_layout(
            bargap=0.1,
            xaxis_title=f"{stat} ({unit})" if unit else f"{stat}",
            yaxis_title="Pourcentage de stations",
            height=height
        )
    st.plotly_chart(fig, use_container_width=True)



def plot_histogramme_comparatif(
    df_observed: pl.DataFrame,
    df_modelised: pl.DataFrame,
    var: str,
    stat: str,
    stat_key: str,
    unit: str,
    height: int
):
    month_order = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                   'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']

    def prepare_df(df: pl.DataFrame, label: str) -> pd.DataFrame:
        df = df.with_columns([
            pl.col(var).cast(pl.Int32).map_dict({i + 1: month_order[i] for i in range(12)}).alias(var)
        ])
        counts = df.select(var).to_series().value_counts().sort("")

        freq_percent = {
            month: (counts.filter(pl.col(var) == month)["count"].sum() / counts["count"].sum() * 100).item()
            if month in counts[var].to_list() else 0.0
            for month in month_order
        }

        return pd.DataFrame({
            var: list(freq_percent.keys()),
            'Pourcentage': list(freq_percent.values()),
            'Source': label
        })

    if stat_key == 'month':
        df_obs = prepare_df(df_observed, "Observé")
        df_mod = prepare_df(df_modelised, "Modélisé")
        hist_df = pd.concat([df_obs, df_mod], ignore_index=True)

        fig = px.bar(
            hist_df,
            x=var,
            y='Pourcentage',
            color='Source',
            barmode='group'
        )
        fig.update_layout(
            bargap=0.15,
            xaxis_title="",
            yaxis_title="Pourcentage de stations",
            height=height,
            xaxis=dict(
                categoryorder='array',
                categoryarray=month_order
            )
        )
    else:
        df_observed = df_observed.with_columns(pl.lit("Observé").alias("Source"))
        df_modelised = df_modelised.with_columns(pl.lit("Modélisé").alias("Source"))
        df_all = pl.concat([df_observed, df_modelised])
        df_all_pd = df_all.to_pandas()

        fig = px.histogram(
            df_all_pd,
            x=var,
            color='Source',
            nbins=50,
            histnorm='percent',
            barmode='overlay'
        )
        fig.update_layout(
            bargap=0.1,
            xaxis_title=f"{stat} ({unit})" if unit else f"{stat}",
            yaxis_title="Pourcentage de stations",
            height=height
        )

    st.plotly_chart(fig, use_container_width=True)
