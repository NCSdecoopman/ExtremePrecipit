import yaml
import os

base_dir = "data"
extensions = [".nc", ".shp"]  # fichiers simples
dir_extensions = [".zarr"]    # répertoires Zarr

manifest = {"datasets": []}

for root, dirs, files in os.walk(base_dir):
    # Fichiers classiques (nc, shp)
    for file in files:
        if any(file.endswith(ext) for ext in extensions):
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            manifest["datasets"].append({
                "path": file_path,
                "size": f"{size} bytes",
                "type": file.split(".")[-1]
            })
    
    # Répertoires .zarr
    for d in dirs:
        if any(d.endswith(ext) for ext in dir_extensions):
            zarr_path = os.path.join(root, d)
            total_size = 0
            # calcul récursif de la taille de tout le dossier Zarr
            for dirpath, _, filenames in os.walk(zarr_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            manifest["datasets"].append({
                "path": zarr_path,
                "size": f"{total_size} bytes",
                "type": "zarr"
            })

# YAML final
with open("data/dataset_manifest.yaml", "w") as f:
    yaml.dump(manifest, f, default_flow_style=False)

print("Manifest YAML généré : dataset_manifest.yaml")
