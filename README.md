# Projet de fin de module — Deep Learning (EMSI Casablanca, 2025–2026)

**Conception, implémentation, comparaison et analyse critique de modèles de deep
learning pour données tabulaires, images et séquences.**

Ce dépôt contient le travail complet demandé par le cahier des charges
(`Projet_Deep-Learning_EMSI.pdf`). Chaque partie est livrée sous forme d'un
**notebook Jupyter autonome** : import des données, configuration, théorie,
implémentation PyTorch, expériences, analyse critique et question de synthèse
sont **entièrement contenus dans le `.ipynb`**.

## Structure

```
Projet-deeplearning/
├── README.md
├── requirements.txt
├── RAPPORT.md / RAPPORT.html                   # Rapport scientifique (HTML = imprimable en PDF)
├── notebooks/
│   ├── 01_Partie1_MLP_BreastCancer.ipynb      # Partie I  — MLP / données tabulaires
│   ├── 02_Partie2_CNN_FashionMNIST.ipynb      # Partie II — CNN / images
│   ├── 03_Partie3_RNN_Seq2Seq_fra_eng.ipynb   # Partie III— RNN/LSTM/GRU/Seq2Seq
│   └── 04_Synthese_transversale.ipynb         # Question transversale finale
├── tools/
│   ├── generate_figures.py                     # reproduit toutes les figures du rapport
│   └── report_to_html.py                       # RAPPORT.md -> RAPPORT.html auto-portant
├── artifacts/figures/                          # figures (.png) + results.json
└── data/                                       # (créé automatiquement) données téléchargées
```

## Rapport

Le **rapport scientifique structuré** (introduction, objectifs, méthodologie,
implémentation, résultats, interprétation, limites, conclusion) est dans
[`RAPPORT.md`](RAPPORT.md) et [`RAPPORT.html`](RAPPORT.html). La version HTML est
**auto-portante** (figures intégrées) : l'ouvrir dans un navigateur puis
*Imprimer → Enregistrer en PDF* produit le PDF du rapport. Les figures sont
reproductibles via `python tools/generate_figures.py`.

## Jeux de données (Kaggle)

| Partie | Dataset | Identifiant Kaggle |
|--------|---------|--------------------|
| I  — MLP  | Breast Cancer Wisconsin | `uciml/breast-cancer-wisconsin-data` |
| II — CNN  | Fashion-MNIST           | `zalando-research/fashionmnist` |
| III— RNN  | Anglais → Français (Tatoeba) | `myksust/fra-eng` |

Chaque notebook télécharge automatiquement son dataset via **`kagglehub`**.
Deux modes sont gérés sans rien coder en dur :

1. **Automatique** — `kagglehub.dataset_download(...)`. Nécessite une clé Kaggle
   (`~/.kaggle/kaggle.json`, obtenue depuis *Kaggle → Account → Create New Token*).
2. **Manuel (repli)** — déposer les fichiers du dataset dans `data/<nom>/` ;
   le notebook les détecte automatiquement.

## Principes de conception

- **Rien n'est codé en dur.** Chaque notebook commence par une cellule de
  configuration (`@dataclass Config`) regroupant chemins, hyperparamètres et
  graine aléatoire. Tout le reste référence cette configuration.
- **Reproductibilité.** Graine fixée pour `random`, `numpy` et `torch`.
- **Portabilité CPU/GPU.** Sélection automatique du *device* (`try_gpu`) ;
  modèle et données placés sur le même *device*.
- **Code réutilisable.** Fonctions génériques (`train_model`, `evaluate`, …)
  paramétrées par la configuration, jamais par des constantes en dur.

## Comment exécuter

```bash
python -m pip install -r requirements.txt
jupyter notebook            # puis ouvrir les notebooks dans l'ordre 01 → 04
```

Exécuter *Run All*. Sur CPU, l'entraînement reste raisonnable car les budgets
d'époques sont volontairement modestes et réglables depuis la `Config`.
