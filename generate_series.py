from pathlib import Path
import polars as pl
from src.utils.logger import get_logger
from src.utils.data_utils import load_data
import numpy as np
from sklearn.metrics import r2_score

def pipeline_gev_from_statisticals():
    global logger
    logger = get_logger(__name__)
   

    echelle="quotidien"   
    mesure = "max_mm_h" if echelle =="horaire" else "max_mm_j"
    min_year = 1959
    max_year = 2022
    season = "mam"

    cols = ["NUM_POSTE", mesure]

    logger.info(f"--- Traitement : {echelle.upper()}--- {season.upper()}")  
    logger.info(f"Chargement des données de {min_year} à {max_year}")

    df_obs = load_data(Path(f"data/statisticals/observed/{echelle}"), season, echelle, cols, min_year, max_year)
    df_mod = load_data(Path("data/statisticals/modelised/horaire"), season, echelle, cols, min_year, max_year)
    df_meta = pl.read_csv("data/metadonnees/obs_vs_mod/obs_vs_mod_quotidien.csv")

    # Assure-toi que les types sont OK
    df_obs = df_obs.with_columns(pl.col("NUM_POSTE").cast(pl.Int64))
    df_mod = df_mod.with_columns(pl.col("NUM_POSTE").cast(pl.Int64))
    df_meta = df_meta.with_columns(
        pl.col("NUM_POSTE_obs").cast(pl.Int64),
        pl.col("NUM_POSTE_mod").cast(pl.Int64)
    )

    # --- 1. Créer la grille complète (toutes années × tous postes d'observations)

    # Toutes les années de min_year à max_year
    years = pl.DataFrame({"year": list(range(min_year, max_year+1))})

    # Produit cartésien années × métadonnées
    full_grid = years.join(df_meta, how="cross")

    # --- 2. Ajouter les données d'observation (possiblement manquantes)

    df_obs_join = df_obs.rename({"NUM_POSTE": "NUM_POSTE_obs"})

    full_grid = full_grid.join(
        df_obs_join.select(["NUM_POSTE_obs", "year", mesure]).rename({mesure: "max_obs"}),
        on=["NUM_POSTE_obs", "year"],
        how="left"  # important : pour garder tout même si max_obs est manquant
    )

    # --- 3. Ajouter les données de modélisation (toujours présentes)

    df_mod_join = df_mod.rename({"NUM_POSTE": "NUM_POSTE_mod"})

    full_grid = full_grid.join(
        df_mod_join.select(["NUM_POSTE_mod", "year", mesure]).rename({mesure: "max_arome"}),
        on=["NUM_POSTE_mod", "year"],
        how="left"  # normalement toutes les valeurs devraient exister ici
    )

    n_obs = full_grid["NUM_POSTE_obs"].n_unique()
    n_mod = full_grid["NUM_POSTE_mod"].n_unique()

    print(f"Nombre de NUM_POSTE_obs uniques : {n_obs}")
    print(f"Nombre de NUM_POSTE_mod uniques : {n_mod}")

    # --- 4. Garde uniquement les colonnes finales
    df_final = full_grid.select(
        "year",
        pl.col("NUM_POSTE_obs").alias("NUM_POSTE"),
        pl.col("lat_obs").alias("lat"),
        pl.col("lon_obs").alias("lon"),
        "max_obs",
        "max_arome",
    )

    df_final = df_final.filter(
        pl.col("NUM_POSTE").is_in([6149001, 35115001, 81220001])
    )

    print(df_final)

    df_final.write_csv(f"obs_vs_arome_{mesure}.csv")

    # # --- 5. Calcul de la moyenne des maximas par poste
    # df_mean = df_final.group_by("NUM_POSTE").agg([
    #     pl.col("max_obs").max().alias("mean_obs"),
    #     pl.col("max_arome").max().alias("mean_arome")
    # ])

    # # --- 6. Calcul du R²
    # # On filtre les lignes sans NaN
    # df_mean_clean = df_mean.drop_nulls()

    # r2 = r2_score(
    #     df_mean_clean["mean_obs"].to_numpy(),
    #     df_mean_clean["mean_arome"].to_numpy()
    # )

    # print(f"R² entre les moyennes des maxima observés et modélisés : {r2:.3f}")

if __name__ == "__main__":
    pipeline_gev_from_statisticals()