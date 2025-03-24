# src/utils/logger.py

import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)  # Crée ou récupère un logger du nom du module appelant

    if not logger.handlers:  # Évite d'ajouter plusieurs handlers si le logger est déjà configuré
        handler = logging.StreamHandler()  # Handler par défaut qui écrit dans la console

        # Format des logs (ex: 2025-03-21 10:00:00 | src.process.preprocess | INFO | Message)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)  # Ajout du handler
        logger.setLevel(logging.INFO)  # Définit le niveau global (INFO par défaut)
    
    return logger
