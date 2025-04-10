from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    folder_path="data",  # on uploade tout `data/` pour inclure le dossier `statisticals/`
    repo_id="ncsdecoopman/ExtremePrecipit",
    repo_type="dataset",
    allow_patterns=["statisticals/**"]  # upload uniquement ce dossier
)
