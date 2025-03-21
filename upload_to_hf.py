import os
import shutil
from huggingface_hub import HfApi

## GENERER LES SOUS FICHIER DE BIN

# input_folder = r"C:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\binaires\individuels"
# output_folder = r"C:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\binaires\split"

# # processed = 0
# # skipped = 0

# # for filename in os.listdir(input_folder):
# #     if filename.endswith(".bin.xz"):
# #         try:
# #             # Exemple de nom : ts_45.12_3.24.bin.xz ou autre format
# #             base = filename.replace("ts_", "").replace(".bin.xz", "")
# #             lat, lon = base.split("_")  # lat = "45.12", lon = "3.24"

# #             lat_prefix = lat.split(".")[0]  # "45"
# #             lon_prefix = lon.split(".")[0]  # "3"

# #             latlon_folder = os.path.join(output_folder, f"lat_{lat_prefix}_lon_{lon_prefix}")
# #             os.makedirs(latlon_folder, exist_ok=True)

# #             src_file = os.path.join(input_folder, filename)
# #             dst_file = os.path.join(latlon_folder, filename)

# #             shutil.copy2(src_file, dst_file)
# #             processed += 1
# #         except Exception as e:
# #             print(f"❌ Fichier ignoré : {filename} | Erreur : {e}")
# #             skipped += 1

# # print(f"✅ {processed} fichiers déplacés")
# # print(f"⚠️ {skipped} fichiers ignorés (problème de nom ou parsing)")

# # # Afficher le nombre de fichiers dans un dossier
# # max_files = 0
# # max_folder = ""

# # for dirpath, dirnames, filenames in os.walk(output_folder):
# #     num_files = len([f for f in filenames if f.endswith(".bin.xz")])
# #     if num_files > max_files:
# #         max_files = num_files
# #         max_folder = dirpath

# # print(f"📂 Dossier avec le plus de fichiers : {max_folder}")
# # print(f"📦 Nombre de fichiers : {max_files}")

# # Upload Hugging Face
# api = HfApi()
repo_id = "ncsdecoopman/extreme-precip-binaires"

# api.upload_folder(
#     repo_id=repo_id,
#     folder_path=r"C:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\result",
#     path_in_repo="result",
#     repo_type="dataset"
# )

# print("🚀 Upload terminé result sur Hugging Face Hub")


import time
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

api = HfApi()

max_retries = 10
retry_delay = 2700  # 45 minutes
attempt = 0

while attempt < max_retries:
    try:
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=r"C:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\binaires\split",
            repo_type="dataset",
            num_workers=4
        )
        print("Upload and commit successful!")
        break
    except HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print(f"Rate limited. Retrying in {retry_delay / 60} minutes (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(retry_delay)
            attempt += 1
            print(attempt)
        else:
            raise e
else:
    print("Max retries reached, upload aborted.")

print("🚀 Upload terminé binaires sur Hugging Face Hub")