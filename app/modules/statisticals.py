import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from app.utils import load_config

def load_parquet_from_huggingface(year: int, month: int, config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Charge un fichier Parquet depuis Hugging Face en utilisant une configuration YAML.

    Parameters:
        year (int): Année (ex: 2022)
        month (int): Mois (ex: 3)
        config_path (str): Chemin vers le fichier YAML de configuration

    Returns:
        pd.DataFrame: Données du fichier .parquet
    """
    config = load_config(config_path)
    repo_id = config["repo_id"]
    base_path = config["statisticals"]["path"]

    file_relative_path = f"{base_path}/{year:04d}/{month:02d}.parquet"

    hf_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_relative_path,
        repo_type="dataset"
    )

    return pd.read_parquet(hf_path)

def show(config_path):
    config = load_config(config_path)