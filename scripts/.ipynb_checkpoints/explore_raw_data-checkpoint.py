"""
Exploration des données brutes EUROSTAT 

Ce script n'est PAS appelé par main.py : c'est un outil d'exploration, à lancer manuellement quand on veut inspecter les données brutes
(avant nettoyage), pas à chaque exécution du pipeline.

Deux façons de l'utiliser dans VS Code :

1. En terminal :
       python scripts/explore_raw_data.py.

2. En mode interactif Jupyter (équivalent de display() sur Colab) : les blocs "# %%" ci-dessous sont reconnus par
   l'extension Jupyter de VS Code comme des cellules. Ouvre ce fichier,
   clique sur "Run Cell" au-dessus de chaque bloc, et raw_all s'affichera
   dans un tableau interactif (Data Viewer), comme display()
   sur Colab.
"""

# %%
import sys
from pathlib import Path

# Permet d'exécuter ce script depuis scripts/ tout en important src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collect import collect_raw_data, inspect_raw_data, get_series_df

# %%
raw_all = collect_raw_data()

# %%
inspect_raw_data(raw_all, n_rows=10)

# %%
# Affichage interactif (fonctionne uniquement en cellule Jupyter VS Code,
# équivalent de display(raw_all) sur Colab). En script classique, cette
# ligne est silencieuse.
raw_all

# %%
# Inspection libre, à adapter selon les besoins :
# raw_all[raw_all["geo"] == "LU"].head(20)
# raw_all["geo"].value_counts()