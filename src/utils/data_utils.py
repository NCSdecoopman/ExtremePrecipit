import polars as pl

def load_season(year: int, cols: tuple, season_key: str, base_path: str) -> pl.DataFrame:
    filename = f"{base_path}/{year:04d}/{season_key}.parquet"
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
    if echelle.startswith("horaire") or echelle.startswith("w"):
        min_valid_years = 25
    elif echelle == "quotidien":
        min_valid_years = 50
    else:
        raise ValueError('Paramètre "echelle" doit être "horaire" ou "quotidien"')
    
    # Selection des saisons avec nan_limit au maximum
    df_filter = df.filter(pl.col("nan_ratio") <= nan_limit)

    # Calcul du nombre d'années valides par station NUM_POSTE
    station_counts = (
        df_filter.group_by("NUM_POSTE")
        .agg(pl.col("year").n_unique().alias("num_years"))
    )

    # Sélection des NUM_POSTE avec au moins min_valid_years d'années valides
    valid_stations = station_counts.filter(pl.col("num_years") >= min_valid_years)

    # Jointure pour ne garder que les stations valides
    df_final = df_filter.filter(
        pl.col("NUM_POSTE").is_in(valid_stations["NUM_POSTE"])
    )

    return df_final

