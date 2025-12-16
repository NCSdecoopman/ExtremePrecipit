# ExtremePrecipit

## Description

**ExtremePrecipit** est un projet dédié à l’analyse et à la visualisation des précipitations extrêmes. Il propose :

* Un **pipeline de traitement** des données brutes (métadonnées, NetCDF, Zarr, statistiques, ajustements GEV…).
* Une **application Streamlit** interactive pour explorer les séries temporelles, cartes et statistiques relatives aux événements de précipitation extrême (observés et modélisés).
* Une organisation modulaire facilitant l’extension (fonctions utilitaires, configurations).

Ce README détaille l’installation, la structure du projet, l’utilisation de l’application et des pipelines, ainsi que les dépendances nécessaires.

---

## Prérequis

1. **Environnement Python ≥ 3.8**
2. Gestionnaire de paquets `pip` (ou `uv`, `conda` selon la convenance).

### Dépendances Python

| Bibliothèque                                                | Rôle principal                                 |
| ----------------------------------------------------------- | ---------------------------------------------- |
| `streamlit`                                                 | Interface web interactive                      |
| `huggingface_hub`                                           | Téléchargement des données depuis Hugging Face |
| `numpy`, `pandas`                                           | Calculs numériques et gestion de tables        |
| `xarray`, `zarr`, `dask`                                    | Traitement de données multidimensionnelles     |
| `geopandas`, `shapely`, `pydeck`                            | Cartographie et visualisation spatiale         |
| `plotly`                                                    | Graphiques interactifs                         |
| `matplotlib`                                                | Tracés statistiques                            |
| `pyyaml`                                                    | Lecture des fichiers YAML de configuration     |
| `netCDF4`                                                   | Lecture/écriture de fichiers NetCDF            |
| `tqdm`                                                      | Barres de progression                          |


>
> ```bash
> python3 -m venv venv
> source venv/bin/activate   # macOS/Linux
> venv\Scripts\activate      # Windows
> ```

---

## Installation

1. **Cloner le dépôt** :

   ```bash
   git clone https://github.com/votre-utilisateur/ExtremePrecipit.git
   cd ExtremePrecipit
   ```

2. **Installer les dépendances** :
   Créez un fichier `requirements.txt` ou installez directement :

   ```bash
   pip install streamlit huggingface_hub numpy pandas xarray zarr dask geopandas shapely pydeck plotly matplotlib pyyaml netCDF4 tqdm
   ```

   > Si vous préférez un fichier `requirements.txt`, voici un exemple minimal :
   >
   > ```
   > streamlit
   > huggingface_hub
   > numpy
   > pandas
   > xarray
   > zarr
   > dask
   > geopandas
   > shapely
   > pydeck
   > plotly
   > matplotlib
   > pyyaml
   > netCDF4
   > tqdm
   > ```

3. **Structure générale du projet**

   ```
   ExtremePrecipit/
   ├─ .gitignore
   ├─ download_data.py          # Script pour récupérer les données depuis Hugging Face
   ├─ main.py                   # Point d’entrée Streamlit
   ├─ config/                   # Configurations globales (YAML)
   │   ├─ modelised_settings.yaml
   │   └─ observed_settings.yaml
   ├─ app/
   │   ├─ __init__.py
   │   ├─ modules/              # Modules Python (fonctions, affichages, calculs GEV…)
   │   ├─ config/               # Configurations internes à l’app
   │   │   └─ config.yaml
   │   ├─ pipelines/            # Scripts de pipeline pour le traitement des données
   │   ├─ upload/               # Scripts pour l’upload vers un stockage distant
   │   └─ utils/                # Fonctions utilitaires (lecture de config, logs, etc.)
   ├─ src/                      # Version packagée des mêmes modules (pour CLI ou intégration)
   ├─ data/                     # Répertoire cible pour les jeux de données (créé par download_data.py)
   ├─ logs/                     # Dossier de sortie des fichiers de journalisation (logs)
   └─ README.md                 # Ce fichier
   ```

   * Les dossiers `app/` et `src/` contiennent des modules Python similaires ; `app/` est orienté Streamlit, `src/` peut être utilisé pour exécuter les pipelines en ligne de commande.
   * `data/` stocke les données générées par les pipelines.
   * `logs/` stocke les fichiers `.log` générés par les pipelines.

---

## Téléchargement des données

Toutes les données transformées (métadonnées, NetCDF bruts, résultats GEV, etc.) se trouvent sur Hugging Face. Le script `download_data.py` s’appuie sur l’API `huggingface_hub` pour copier l’intégralité du dataset localement sous `data/`.

1. **Exécution du script** :

   ```bash
   python download_data.py
   ```

2. **Arborescence résultante**  :

   ```
   data/
   ├─ metadonnees/
   ├─ gev/
   │   ├─ modelised/         # Paramètres GEV (modélisé)
   │   └─ observed/          # Paramètres GEV (observé)
   ├─ statisticals/
   │   ├─ modelised/         # Statistiques extraites (modélisé)
   │   └─ observed/          # Statistiques extraites (observé)
   └─ … (autres dossiers selon pipelines)
   ```

---

## Structure du projet

### 1. `app/`

* **`app/modules/`** : contient les fonctions de calcul, d’affichage et de visualisation (ex. : calcul de quantiles GEV, générateurs de cartes, graphiques de dispersion…).
* **`app/config/config.yaml`** : paramètres internes à l’application (ex. plages d’années, chemins relatifs aux dossiers `data/`).
* **`app/pipelines/`** : scripts Python séquentiels pour :

  * `pipeline_obs_to_zarr.py` : convertir les NetCDF observés en format Zarr.
  * `pipeline_nc_to_zarr.py` : convertir les NetCDF modélisés en Zarr.
  * `pipeline_zarr_to_stats.py` : extraire les statistiques (par ex. maxima annuel) des Zarr et les enregistrer en Parquet.
  * `pipeline_stats_to_gev.py` : ajuster un modèle GEV aux séries statistiques et enregistrer les paramètres.
  * `pipeline_obs_vs_mod.py` : comparer observé vs modélisé.
  * `pipeline_gev_min_loglike.py` : recherche du modèle GEV optimal sur critère log-vraisemblance.
  * `pipeline_best_gev.py` : sélection du meilleur modèle GEV, génération de cartes de paramètres.
  * `run_all.py` : exécute l’ensemble des étapes dans l’ordre automatiquement.
* **`app/utils/`** : fonctions communes (lecture de fichiers YAML, gestion des chemins, journalisation, manipulations de données, création de menus Streamlit).

### 2. `src/`

* Contient une structure similaire à `app/`, organisée comme un package Python afin de :

  * Fournir des pipelines exécutables en CLI.
  * Permettre une installation éventuelle en tant que module (`pip install -e src/`).

### 3. `download_data.py`

* Script autonome qui récupère l’ensemble des données brutes et intermédiaires depuis Hugging Face.
* À lancer **avant** d’exécuter les pipelines ou l’application Streamlit.

### 4. `main.py`

* Fichier de lancement de l’application Streamlit.
* Ouvre une interface web pour :

  * Visualiser les cartes de précipitations extrêmes (observées et modélisées).
  * Explorer les séries temporelles GEV et statistiques.
  * Générer des graphiques interactifs (histogrammes, nuages de points, etc.).

* Lancez-le avec :

  ```bash
  streamlit run main.py
  ```

### 5. `config/`

* Fichiers YAML (`modelised_settings.yaml` & `observed_settings.yaml`) décrivent :

  * Les chemins d’accès aux données brutes (`data/raw/...`).
  * Les paramètres d’extraction (échelles temporelles, filtres spatiaux).
  * Les répertoires de sortie (`data/statisticals/…`, `data/gev/…`).
  * Les réglages de journalisation (dossier `logs/…`).

---

## Utilisation

### A. Préparation des données

1. **Télécharger** les données (métadonnées, NetCDF, GEV pré-calculés) :

   ```bash
   python download_data.py
   ```
2. **Configurer** (si besoin) les chemins ou paramètres dans `config/modelised_settings.yaml` et `config/observed_settings.yaml`.

### B. Lancer les pipelines

> **Option 1 – Séquentiel**
> Exécutez chaque pipeline pas à pas, dans l’ordre recommandé :
>
> ```bash
> python app/pipelines/pipeline_nc_to_zarr.py
> python app/pipelines/pipeline_zarr_to_stats.py
> python app/pipelines/pipeline_stats_to_gev.py
> python app/pipelines/pipeline_obs_to_zarr.py
> python app/pipelines/pipeline_obs_vs_mod.py
> python app/pipelines/pipeline_gev_min_loglike.py
> python app/pipelines/pipeline_best_gev.py
> ```
>
> * Chaque script écrit ses logs dans `logs/pipeline/...`
> * Les sorties (Parquet, Zarr, fichiers CSV) se trouvent dans `data/statisticals/…` et `data/gev/…`.

> **Option 2 – Tout-en-un**
> Le script `run_all.py` orchestrera l’ensemble des étapes automatiquement :
>
> ```bash
> python app/pipelines/run_all.py
> ```

### C. Lancer l’application Streamlit

1. À la racine du projet :

   ```bash
   streamlit run main.py
   ```
2. Ouvrez votre navigateur à l’URL indiquée (par défaut [http://localhost:8501](http://localhost:8501)).
3. Choisissez dans les menus :

   * **Période** (1959–2015 par défaut)
   * **Type de données** : Observées vs Modélisées
   * **Statistiques** : Max annuels, analyses GEV, cartes spatiales, etc.


## Structure détaillée des répertoires

```
ExtremePrecipit/
├─ .gitignore
├─ download_data.py  
├─ main.py
├─ config/
│   ├─ modelised_settings.yaml   # Paramètres pour les données modélisées
│   └─ observed_settings.yaml    # Paramètres pour les données observées
├─ app/
│   ├─ modules/                  # Fonctions de calcul et de visualisation
│   │   ├─ all_max.py
│   │   ├─ gev.py
│   │   ├─ niveau_retour.py
│   │   ├─ periode_retour.py
│   │   ├─ scatter_plot.py
│   │   └─ … (autres modules utilitaires)
│   ├─ config/
│   │   └─ config.yaml           # Réglages internes à l’app (menus, chemins)
│   ├─ pipelines/                # Scripts de pipeline (conversion, statistiques, GEV)
│   │   ├─ pipeline_nc_to_zarr.py
│   │   ├─ pipeline_zarr_to_stats.py
│   │   ├─ pipeline_stats_to_gev.py
│   │   ├─ pipeline_obs_to_zarr.py
│   │   ├─ pipeline_obs_vs_mod.py
│   │   ├─ pipeline_gev_min_loglike.py
│   │   ├─ pipeline_best_gev.py
│   │   └─ run_all.py
│   ├─ upload/                   # Upload des résultats (ex. : vers un cloud)
│   │   └─ upload_statisticals.py
│   └─ utils/                    # Fonctions utilitaires (config_tools, data_utils, logger…)
│       ├─ config_tools.py
│       ├─ data_utils.py
│       ├─ logger.py
│       └─ … (autres utilitaires)
├─ src/                          # Version packagée du code (modules, pipelines, utils)
│   ├─ config/
│   ├─ pipelines/
│   ├─ upload/
│   └─ utils/
├─ data/                         # Dossier cible pour les données téléchargées (via download_data.py)
│   ├─ metadonnees/
│   ├─ raw/
│   ├─ zarr/ (→ générés par pipelines)
│   ├─ statisticals/ (→ résultats Parquet)
│   └─ gev/ (→ paramètres GEV)
└─ logs/                         # Fichiers de journalisation des pipelines
```

---

## Contribution

Toute contribution (signalement de bug, demande d’amélioration, pull request) est la bienvenue !

---

## Licence

Ce projet est distribué sous licence **MIT**.
