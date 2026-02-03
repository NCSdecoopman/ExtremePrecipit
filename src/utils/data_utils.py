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


def export_station_month_series_to_csv(
    station_id: str | int,
    month: str | int,
    echelle: str = "horaire",
    zarr_dir: str = "data/processed/observed/zarr",
    out_csv: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    config_path: str = "config/observed_settings.yaml",
    var_name: str = "pr",
):
    """
    Exporte la série temporelle d'une station pour un mois donné (toutes années)
    vers un CSV trié par temps.
    """
    import yaml
    import numpy as np
    import pandas as pd
    import xarray as xr
    from pathlib import Path

    def month_to_int(value) -> int:
        if isinstance(value, int):
            if 1 <= value <= 12:
                return value
            raise ValueError(f"Mois invalide: {value}")
        text = str(value).strip().lower()
        if text.isdigit():
            num = int(text)
            if 1 <= num <= 12:
                return num
            raise ValueError(f"Mois invalide: {value}")
        mapping = {
            "janvier": 1, "janv": 1, "jan": 1,
            "fevrier": 2, "fev": 2, "feb": 2,
            "mars": 3, "mar": 3,
            "avril": 4, "avr": 4, "apr": 4,
            "mai": 5, "may": 5,
            "juin": 6, "jun": 6,
            "juillet": 7, "juil": 7, "jul": 7,
            "aout": 8, "aug": 8,
            "septembre": 9, "sept": 9, "sep": 9,
            "octobre": 10, "oct": 10,
            "novembre": 11, "nov": 11,
            "decembre": 12, "dec": 12,
        }
        if text not in mapping:
            raise ValueError(f"Mois invalide: {value}")
        return mapping[text]

    month_int = month_to_int(month)
    station_key = str(station_id)

    zarr_base = Path(zarr_dir) / echelle
    if not zarr_base.exists():
        raise FileNotFoundError(f"Zarr introuvable: {zarr_base}")

    years = sorted(
        int(p.stem) for p in zarr_base.glob("*.zarr")
        if p.stem.isdigit()
    )
    if start_year is not None:
        years = [y for y in years if y >= start_year]
    if end_year is not None:
        years = [y for y in years if y <= end_year]
    if not years:
        raise ValueError("Aucune année trouvée dans la plage demandée.")

    scale_factor = 1.0
    unit_conversion = 1.0
    fill_value = None
    cfg_path = Path(config_path)
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        pr_cfg = cfg.get("zarr", {}).get("variables", {}).get("pr", {})
        scale_factor = float(pr_cfg.get("scale_factor", 1.0))
        unit_conversion = float(pr_cfg.get("unit_conversion", 1.0))
        fill_value = pr_cfg.get("fill_value", None)

    frames = []
    for year in years:
        ds = xr.open_zarr(zarr_base / f"{year}.zarr", chunks=None)
        try:
            if "NUM_POSTE" not in ds.coords:
                raise ValueError(f"Coordonnée NUM_POSTE absente dans {year}.zarr")
            poste_values = ds["NUM_POSTE"].values
            if poste_values.dtype.kind in "iu":
                try:
                    station_sel = int(station_key)
                except ValueError:
                    ds.close()
                    continue
                if station_sel not in poste_values:
                    ds.close()
                    continue
            else:
                station_sel = station_key
                if station_sel not in poste_values.astype(str):
                    ds.close()
                    continue

            da = ds[var_name].sel(NUM_POSTE=station_sel)
            da = da.where(da["time"].dt.month == month_int, drop=True)
            df = da.to_dataframe(name=var_name).reset_index()

            if fill_value is not None:
                df[var_name] = df[var_name].replace(fill_value, np.nan)
            df[var_name] = df[var_name] / (unit_conversion * scale_factor)
            df = df.dropna(subset=[var_name])

            frames.append(df[["time", "NUM_POSTE", var_name]])
        finally:
            ds.close()

    if not frames:
        raise ValueError("Aucune donnée trouvée pour cette station/mois.")

    df_final = pd.concat(frames, ignore_index=True).sort_values("time")
    df_final = df_final.rename(columns={var_name: "pr"})

    if out_csv is None:
        out_dir = Path("outputs/series")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"station_{station_key}_{month_int:02d}_{echelle}.csv"
    else:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_final.to_csv(out_csv, index=False)
    return out_csv


def export_station_monthly_maxima_to_csv(
    station_id: str | int,
    month: str | int,
    echelle: str = "horaire",
    stats_dir: str = "data/statisticals/observed",
    out_csv: str | None = None,
    nan_limit: float = 0.10,
):
    """
    Exporte la série des maxima mensuels (une valeur par année) utilisée pour la GEV,
    avant normalisation.
    """
    from pathlib import Path
    import pandas as pd

    def month_to_season_key(value) -> str:
        if isinstance(value, int):
            mapping = {
                1: "jan", 2: "fev", 3: "mar", 4: "avr", 5: "mai", 6: "jui",
                7: "juill", 8: "aou", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
            }
            if value not in mapping:
                raise ValueError(f"Mois invalide: {value}")
            return mapping[value]
        text = str(value).strip().lower()
        mapping = {
            "janvier": "jan", "janv": "jan", "jan": "jan",
            "fevrier": "fev", "fev": "fev", "feb": "fev",
            "mars": "mar", "mar": "mar",
            "avril": "avr", "avr": "avr", "apr": "avr",
            "mai": "mai", "may": "mai",
            "juin": "jui", "jun": "jui",
            "juillet": "juill", "juil": "juill", "jul": "juill",
            "aout": "aou", "aug": "aou",
            "septembre": "sep", "sept": "sep", "sep": "sep",
            "octobre": "oct", "oct": "oct",
            "novembre": "nov", "nov": "nov",
            "decembre": "dec", "dec": "dec",
        }
        if text not in mapping:
            raise ValueError(f"Mois invalide: {value}")
        return mapping[text]

    season = month_to_season_key(month)
    station_key = str(station_id)

    input_dir = Path(stats_dir) / echelle
    mesure, min_year, max_year, len_serie = years_to_load(echelle, season, str(input_dir))
    cols = ["NUM_POSTE", mesure, "nan_ratio"]

    df = load_data(input_dir, season, echelle, cols, min_year, max_year)
    df = cleaning_data_observed(df, echelle, len_serie=len_serie, nan_limit=nan_limit)
    df = df.drop_nulls(subset=[mesure])
    df = df.filter(pl.col("NUM_POSTE") == station_key)

    if df.is_empty():
        raise ValueError("Aucune donnée trouvée pour cette station/mois.")

    df_out = (
        df.select(["year", "NUM_POSTE", mesure])
        .sort("year")
        .to_pandas()
    )
    df_out[mesure] = df_out[mesure].round(2)

    if out_csv is None:
        out_dir = Path("outputs/series")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"station_{station_key}_{season}_{echelle}_monthly_max.csv"
    else:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(out_csv, index=False)
    return out_csv
