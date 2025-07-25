# Dockerfile pour l'environnement 'stage'
FROM continuumio/miniconda3:latest

# 1. Définir les variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive

# 2. Copier le fichier environment_stage.yml dans le conteneur
COPY environment_stage.yml /tmp/environment_stage.yml

# 3. Créer l'environnement conda
RUN conda env create -f /tmp/environment_stage.yml && \
    conda clean -afy

# 4. Activer l'environnement par défaut
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/envs/stage/bin:$PATH

# 5. Installer les dépendances privées (themis_maths et hades_stats)
# Remplacer <TOKEN> par votre token personnel GitLab
RUN source activate stage && \
    pip install --no-cache-dir git+https://<TOKEN>@gricad-gitlab.univ-grenoble-alpes.fr/mnemosyne/themis_maths.git && \
    pip install --no-cache-dir git+https://www.gricad-gitlab.univ-grenoble-alpes.fr/mnemosyne/hades_stats.git

# 6. Définir l'environnement par défaut
ENV CONDA_DEFAULT_ENV=stage

# 7. Définir le dossier de travail
WORKDIR /workspace

# 8. (Optionnel) Copier le code source
# COPY . /workspace

# 9. (Optionnel) Commande par défaut
CMD ["/bin/bash"] 