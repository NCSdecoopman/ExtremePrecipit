import pandas as pd
import streamlit as st
import calendar

def compute_statistic_per_point(df: pd.DataFrame, stat_key: str) -> pd.DataFrame:   
    if stat_key == "mean":
        df = df.copy()
        df["mean_mm_j"] = df["mean_mm_h"] * 24
        return df.groupby(["lat", "lon"]).agg(
            mean_all_mm_h=("mean_mm_h", "mean"),
            mean_all_mm_j=("mean_mm_j", "mean")
        ).reset_index()

    elif stat_key == "max":
        return df.groupby(["lat", "lon"]).agg(
            max_all_mm_h=("max_mm_h", "max"),
            max_all_mm_j=("max_mm_j", "max")
        ).reset_index()

    elif stat_key == "mean-max":
        return df.groupby(["lat", "lon"]).agg(
            max_mean_mm_h=("max_mm_h", "mean"),
            max_mean_mm_j=("max_mm_j", "mean")
        ).reset_index()

    elif stat_key == "date":
        # Date du maximum horaire et journalier
        idx_h = df.groupby(["lat", "lon"])["max_mm_h"].idxmax()
        idx_j = df.groupby(["lat", "lon"])["max_mm_j"].idxmax()

        df_h = df.loc[idx_h, ["lat", "lon", "max_date_mm_h"]].rename(columns={"max_date_mm_h": "date_max_h"})
        df_j = df.loc[idx_j, ["lat", "lon", "max_date_mm_j"]].rename(columns={"max_date_mm_j": "date_max_j"})
        return pd.merge(df_h, df_j, on=["lat", "lon"])

    elif stat_key == "month":
        # On commence par convertir les colonnes de date
        df = df.copy()
        df["max_date_mm_h"] = pd.to_datetime(df["max_date_mm_h"], errors="coerce")
        df["max_date_mm_j"] = pd.to_datetime(df["max_date_mm_j"], errors="coerce")

        # Extraction du mois
        df["mois_max_h"] = df["max_date_mm_h"].dt.month
        df["mois_max_j"] = df["max_date_mm_j"].dt.month

        # Suppression des valeurs NaN éventuelles
        df = df.dropna(subset=["mois_max_h", "mois_max_j"])

        # Calcul du mois le plus fréquent par point (en tenant compte des années)
        mois_h = (
            df.groupby(["lat", "lon", "mois_max_h"])
            .size()
            .reset_index(name="count_h")
            .sort_values("count_h", ascending=False)
            .drop_duplicates(subset=["lat", "lon"])
            .rename(columns={"mois_max_h": "mois_pluvieux_h"})
            [["lat", "lon", "mois_pluvieux_h"]]
        )

        mois_j = (
            df.groupby(["lat", "lon", "mois_max_j"])
            .size()
            .reset_index(name="count_j")
            .sort_values("count_j", ascending=False)
            .drop_duplicates(subset=["lat", "lon"])
            .rename(columns={"mois_max_j": "mois_pluvieux_j"})
            [["lat", "lon", "mois_pluvieux_j"]]
        )

        return pd.merge(mois_h, mois_j, on=["lat", "lon"])


    elif stat_key == "numday":
        # Moyenne du nombre de jours de pluie sur la période sélectionnée
        n_years = df["max_date_mm_h"].str[:4].astype(int).nunique()
        return df.groupby(["lat", "lon"])["n_days_gt1mm"] \
                 .sum() \
                 .div(n_years) \
                 .reset_index(name="jours_pluie_moyen")

    else:
        raise ValueError(f"Statistique inconnue : {stat_key}")