import os
import polars as pl


def years_to_load(echelle: str, season: str, input_dir: str):
    # Fixation de l'échelle pour le choix des colonnes à lire
    mesure = "max_mm_h" if "horaire" in echelle else "max_mm_j"
    
    # Liste des années disponibles
    years = [
        int(name) for name in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, name)) and name.isdigit() and len(name) == 4
    ]

    if years:
        min_year = min(years) if echelle in ["quotidien", "horaire_reduce"] else 1990 # Année minimale
        max_year = max(years)
    else:
        print("Aucune année valide trouvée.")

    len_serie = 50 if echelle in ["quotidien", "horaire_reduce"] else 25 # Longueur minimale d'une série valide

    if season in ["hydro", "djf"]:
        min_year+=1 # On commence en 1960

    return mesure, min_year, max_year, len_serie



def load_season(year: int, cols: tuple, season_key: str, base_path: str) -> pl.DataFrame:
    filename = f"{base_path}/{year:04d}/{season_key}.parquet"
    if cols is None:
        # pas de filtrage : on ne passe pas l'argument columns
        return pl.read_parquet(filename)
    else:
        # filtrage sur les colonnes spécifiées
        return pl.read_parquet(filename, columns=cols)

def load_data(intputdir: str, season: str, echelle: str, cols: tuple, min_year: int, max_year: int) -> pl.DataFrame:
    dataframes = []
    errors = []

    for year in range(min_year, max_year + 1):
        try:
            df = load_season(year, cols, season, intputdir)
            df = df.with_columns([
                pl.lit(year).alias("year")
            ])
            
            dataframes.append(df)

        except Exception as e:
            errors.append(f"{year} ({season}) : {e}")

    if errors:
        for err in errors:
            raise ValueError(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")

    return pl.concat(dataframes, how="vertical")


def cleaning_data_observed(
    df: pl.DataFrame,
    echelle: str | None = None,
    len_serie: int = None,
    nan_limit: float = 0.10
) -> pl.DataFrame:
    """
    Filtre les maxima par deux critères :
      1) on annule les valeurs d’une année si nan_ratio > nan_limit
      2) on ne garde que les stations ayant au moins n années valides :
         n = 25   si echelle == "horaire"
         n = 50   si echelle == "quotidien"
    """
    # ——— règles dépendant de l’échelle ———
    if len_serie is None:
        raise ValueError('Paramètre len_serie à préciser')
    
    # Selection des saisons avec nan_limit au maximum
    df_filter = df.filter(pl.col("nan_ratio") <= nan_limit)

    # Calcul du nombre d'années valides par station NUM_POSTE
    station_counts = (
        df_filter.group_by("NUM_POSTE")
        .agg(pl.col("year").n_unique().alias("num_years"))
    )

    # Sélection des NUM_POSTE avec au moins len_serie d'années valides
    valid_stations = station_counts.filter(pl.col("num_years") >= len_serie)

    # Jointure pour ne garder que les stations valides
    df_final = df_filter.filter(
        pl.col("NUM_POSTE").is_in(valid_stations["NUM_POSTE"])
    )

    return df_final

