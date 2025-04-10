# src/config/config_loader.py

import yaml
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str = "config/settings.yaml") -> dict:
    """
    Charge un fichier de configuration YAML.

    Args:
        config_path (str): Chemin vers le fichier YAML.

    Returns:
        dict: Configuration sous forme de dictionnaire.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config non trouvée : {config_path}")
        raise FileNotFoundError(f"Config introuvable : {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config chargée depuis {config_path}")
    return config
