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


def cleaning_data_observed(df: pl.DataFrame, nan_limit: float = 0.15) -> pl.DataFrame:
    # Moyenne du ratio de NaN par station (lat, lon)
    station_counts = (
        df.group_by(["NUM_POSTE"])
        .agg(pl.col("nan_ratio").mean().alias("nan_ratio"))
    )

    # Stations valides selon le seuil
    valid = station_counts.filter(pl.col("nan_ratio") <= nan_limit)

    # Jointure pour ne garder que les stations valides
    df_filtered = df.join(valid.select(["NUM_POSTE"]), on=["NUM_POSTE"], how="inner")

    return df_filtered