from huggingface_hub import HfApi, upload_file
import os

import warnings
warnings.filterwarnings("ignore", message="It seems that you are about to commit a data file*")

# Config
SPACE_ID = "ncsdecoopman/ExtremePrecipit"  # nom du Space
LOCAL_DIR = "C:/Users/nicod/Documents/GitHub/app/data"  # dossier local
REMOTE_DIR = "data"  # dossier de destination sur le repo Hugging Face

# Authentification (ouvre une fenêtre navigateur si token non dispo)
api = HfApi()

# Boucle sur tous les fichiers dans le dossier local
for root, _, files in os.walk(LOCAL_DIR):
    for file in files:
        local_path = os.path.join(root, file)
        
        # Calculer le chemin relatif
        relative_path = os.path.relpath(local_path, LOCAL_DIR)
        repo_path = f"{REMOTE_DIR}/{relative_path}".replace("\\", "/")

        print(f"Uploading {local_path} to {repo_path}...")

        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=SPACE_ID,
            repo_type="space",
        )

print("✅ Upload terminé.")