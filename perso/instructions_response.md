# Instructions pour répondre aux reviewers (HESS / Copernicus)

Ce fichier regroupe les règles de mise en forme et de rédaction à suivre pour répondre aux reviewers.

## 1. Rédaction de la réponse dans le QMD

* **Notation cohérente** :
  - Garder la notation originale du reviewer dans le texte de la réponse (ex. $RL_{10y}$ et $RL_{2y}$).
  - Utiliser la notation formelle du manuscrit dans les figures, légendes et axes (ex. $\mathrm{RL}_{10}$, $\mathrm{RL}_2$).
* **Précision des sections** :
  - Indiquer précisément les sous-sections modifiées en incluant leur nom complet (ex. `Section 4.2.1 (Daily)` plutôt que la section globale `Section 4.2`).
* **Modifications fictives** :
  - Décrire dans la réponse ce qui est modifié dans le manuscrit sans éditer directement le fichier `.tex` principal (`main.tex`), sauf instruction contraire.

## 2. Production des figures et scripts

* **Scripts dédiés** : Ranger les nouveaux codes Python dans le dossier `perso/v2_modifs/scripts/`.
* **Figures séparées** : Préférer diviser les analyses complexes en plusieurs figures (ex. cartes spatiales d'un côté, scatter plots de l'autre) si cela améliore la lisibilité.
* **Design des cartes** :
  - Utiliser les contours côtiers simplifiés nationaux (sans les départements).
  - Superposer les courbes de niveau du relief national (`selection_courbes_niveau_france.shp`).
  - Utiliser la palette de couleurs officielle `prec_div.txt`.
* **Sauvegarde atomique** :
  - Sauvegarder les figures en PNG via un fichier temporaire renommé ensuite (`os.replace`). Cela évite que les outils de live-reload (comme Quarto preview) ne chargent des fichiers PNG corrompus ou vides pendant l'écriture.

Article à prendre en compte uniquement (pré-print lu par les reviewers) : `perso\v1_1_preprint\main.tex`
