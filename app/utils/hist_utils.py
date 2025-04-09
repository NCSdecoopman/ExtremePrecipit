import plotly.express as px
import streamlit as st
import polars as pl
import pandas as pd

def plot_histogramme(df: pl.DataFrame, var, stat, stat_key, unit, height):
    df = df.to_pandas()
    # Définir l’ordre complet des mois
    month_order = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                   'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
    
    if stat_key == 'month':
        # Copie de sécurité
        df = df.copy()
        
        # Convertir le numéro du mois (1-12) en label texte
        df[var] = df[var].astype(int)
        df[var] = df[var].map({i+1: month_order[i] for i in range(12)})

        # Calculer la répartition (pourcentage) par mois
        counts = df[var].value_counts()                  # nb de lignes par mois présent
        counts = counts.reindex(month_order, fill_value=0)  # forcer l’existence de tous les mois, avec 0 pour les absents

        # Convertir en pourcentage
        total = counts.sum()
        freq_percent = (counts / total * 100) if total > 0 else counts
        
        # Construire un nouveau DF pour Plotly
        hist_df = pd.DataFrame({var: freq_percent.index, 'Pourcentage': freq_percent.values})

        # Plot en barres
        fig = px.bar(
            hist_df,
            x=var,
            y='Pourcentage'
        )
        fig.update_layout(
            bargap=0.1,           # Espacement entre barres
            xaxis_title="",       # Pas de titre horizontal
            yaxis_title="Pourcentage de stations",
            height=height,
            xaxis=dict(
                categoryorder='array',
                categoryarray=month_order
            )
        )
    else:
        # Cas normal : on garde px.histogram
        fig = px.histogram(
            df,
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


def plot_histogramme_comparatif(df_observed: pl.DataFrame, df_modelised: pl.DataFrame, var, stat, stat_key, unit, height):
    df_observed = df_observed.to_pandas()
    df_modelised = df_modelised.to_pandas()
    month_order = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                   'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']

    if stat_key == 'month':
        def prepare_df(df, label):
            df = df.copy()
            df[var] = df[var].astype(int)
            df[var] = df[var].map({i + 1: month_order[i] for i in range(12)})
            counts = df[var].value_counts()
            counts = counts.reindex(month_order, fill_value=0)
            total = counts.sum()
            freq_percent = (counts / total * 100) if total > 0 else counts
            return pd.DataFrame({
                var: freq_percent.index,
                'Pourcentage': freq_percent.values,
                'Source': label
            })

        df_obs = prepare_df(df_observed, "Observé")
        df_mod = prepare_df(df_modelised, "Modélisé")
        hist_df = pd.concat([df_obs, df_mod], ignore_index=True)

        fig = px.bar(
            hist_df,
            x=var,
            y='Pourcentage',
            color='Source',
            barmode='group'  # Affichage côte à côte
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
        # Affichage standard pour les autres stats
        df_observed['Source'] = "Observé"
        df_modelised['Source'] = "Modélisé"
        df_all = pd.concat([df_observed, df_modelised], ignore_index=True)

        fig = px.histogram(
            df_all,
            x=var,
            color='Source',
            nbins=50,
            histnorm='percent',
            barmode='overlay'  # ou 'group' si tu veux les voir côte à côte
        )
        fig.update_layout(
            bargap=0.1,
            xaxis_title=f"{stat} ({unit})" if unit else f"{stat}",
            yaxis_title="Pourcentage de stations",
            height=height
        )

    st.plotly_chart(fig, use_container_width=True)
