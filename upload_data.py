import os
import shutil
from huggingface_hub import create_repo, HfApi, HfFolder, whoami
from subprocess import run
from pathlib import Path

# === Configuration ===
local_data_path = Path(r"C:\Users\nicod\Documents\GitHub\app\data")
repo_id = "ncsdecoopman/ExtremePrecipit"
local_clone_path = Path.cwd() / "ExtremePrecipit"
repo_url = f"https://huggingface.co/datasets/{repo_id}"

# === Étape 1 : Créer le repo s'il n'existe pas ===
api = HfApi()
user = whoami()
if not any(repo.id == repo_id for repo in api.list_datasets(author=user['name'])):
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# === Étape 2 : Cloner le repo avec Git LFS ===
run(["git", "lfs", "install"])
if local_clone_path.exists():
    shutil.rmtree(local_clone_path)
run(["git", "clone", repo_url, str(local_clone_path)])

# === Étape 3 : Copier les données en gardant l'arborescence ===
target_data_path = local_clone_path / "data"
shutil.copytree(local_data_path, target_data_path, dirs_exist_ok=True)

# === Étape 4 : Suivre les fichiers lourds avec Git LFS ===
os.chdir(local_clone_path)
# Exemples d'extensions à suivre avec git-lfs
extensions = ["*.zarr", "*.parquet", "*.nc", "*.csv", "*.xz"]
for ext in extensions:
    run(["git", "lfs", "track", ext])

# === Étape 5 : Commit et push ===
run(["git", "add", ".gitattributes"])
run(["git", "add", "."])
run(["git", "commit", "-m", "Ajout des données avec arborescence complète"])
run(["git", "push"])
