"""Configuration de la collecte pytest.

Ce fichier existe pour une seule raison : rendre la racine du projet importable,
quels que soient l'interpréteur et le répertoire courant.

Le problème qu'il règle
------------------------
Les tests importent `config`, `app` et leurs doublures partagées, qui vivent à la
racine du dépôt. Sans `__init__.py` dans `tests/`, pytest insère **le dossier des
tests** dans `sys.path`, pas la racine. L'import ne fonctionne alors que par
accident — parce que `python -m pytest` ajoute le répertoire courant — et cet
accident ne se produit pas de la même façon d'une version de Python à l'autre :
la suite passait sur 3.11 et échouait à la collecte sur 3.14.

La présence d'un `conftest.py` à la racine suffit à ce que pytest la traite comme
`rootdir` ; les insertions explicites ci-dessous rendent la garantie indépendante
de cette convention. Le projet se teste indifféremment par `pytest`,
`python -m pytest`, ou depuis un IDE qui choisit son propre répertoire courant.

Pourquoi les doublures s'importent sans préfixe
------------------------------------------------
`tests` est un nom si banal qu'il existe déjà comme paquet installé dans certains
environnements — c'était le cas ici sur Python 3.14, où
`from tests.zone_doubles import ...` résolvait vers un `site-packages/tests/`
sans rapport avec le projet. Le dossier des tests est donc ajouté à `sys.path`
et les doublures s'importent par `from zone_doubles import ...`, sans préfixe :
un nom court et local vaut mieux qu'un nom qualifié qui peut désigner autre chose.
"""

from __future__ import annotations

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent

for chemin in (RACINE, RACINE / "tests"):
    if str(chemin) not in sys.path:
        sys.path.insert(0, str(chemin))
