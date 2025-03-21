import os
import glob

from typing import List

def find_files(path: str, pattern: str) -> List[str]:
    """Trouve les fichiers correspondant au pattern donné dans le dossier spécifié."""
    search_path = os.path.join(path, pattern)
    files = glob.glob(search_path)
    return files