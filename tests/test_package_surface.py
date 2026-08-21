"""La surface publique du paquet décrit le pipeline entier, pas son état d'hier.

`sentinel/__init__.py` exportait `Detection`, `Detector`, `Tracker` et les
exceptions — la photographie du projet à l'étape 3. Ni `Event`, ni `ZoneManager`,
ni `ReportGenerator`, ni `VideoSource`, ni `Timeline` n'y figuraient, alors que
tous existaient et étaient testés.

Ce n'est pas un détail de forme : `__all__` est la première chose qu'on lit pour
comprendre un paquet, et un `__all__` figé raconte un projet plus petit qu'il
n'est.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import sentinel

PAQUET = Path(sentinel.__file__).parent

# Une classe par étage du pipeline, dans le sens de lecture. Le jour où un étage
# s'ajoute, ce test échoue tant que sa classe n'est pas exposée — c'est
# exactement ce qu'on veut.
ETAGES = [
    ("source", "VideoSource"),
    ("perception", "Detector"),
    ("mémoire temporelle", "Tracker"),
    ("géométrie", "ZoneManager"),
    ("règles métier", "EventEngine"),
    ("chronologie", "Timeline"),
    ("restitution", "ReportGenerator"),
    ("rapport de session", "SessionReportGenerator"),
]


@pytest.mark.parametrize("etage, nom", ETAGES, ids=[e for e, _ in ETAGES])
def test_every_stage_of_the_pipeline_is_exported(etage: str, nom: str) -> None:
    """Chaque étage du pipeline est accessible depuis `sentinel`."""
    assert nom in sentinel.__all__, f"L'étage « {etage} » n'est pas exposé."
    assert hasattr(sentinel, nom)


def test_every_exported_name_actually_exists() -> None:
    """Un nom listé dans `__all__` mais absent casse `from sentinel import *`."""
    manquants = [nom for nom in sentinel.__all__ if not hasattr(sentinel, nom)]

    assert manquants == [], f"Noms annoncés mais absents : {manquants}"


def test_the_data_contract_is_exported() -> None:
    """`Detection` circule entre tous les modules : elle appartient au contrat."""
    assert "Detection" in sentinel.__all__


def test_every_business_exception_is_exported() -> None:
    """L'interface attrape `SentinelError` : toute la hiérarchie doit être visible.

    Une exception non exportée oblige à importer un sous-module pour l'attraper,
    ce qui revient à faire fuiter l'organisation interne du paquet.
    """
    from sentinel import exceptions

    definies = {
        noeud.name
        for noeud in ast.parse(
            Path(exceptions.__file__).read_text(encoding="utf-8")
        ).body
        if isinstance(noeud, ast.ClassDef)
    }

    assert definies <= set(sentinel.__all__), definies - set(sentinel.__all__)


def test_no_private_name_is_exported() -> None:
    """`_write` et `_to_latin1` sont partagés entre modules, pas publiés.

    Ils règlent un piège interne de fpdf2 ; les exposer ferait croire à un
    service offert par le paquet.
    """
    prives = [nom for nom in sentinel.__all__ if nom.startswith("_") and nom != "__version__"]

    assert prives == []


def test_the_export_list_has_no_duplicate() -> None:
    """Un doublon dans `__all__` signale une fusion mal relue."""
    assert len(sentinel.__all__) == len(set(sentinel.__all__))


@pytest.mark.parametrize(
    "module",
    sorted(chemin.stem for chemin in PAQUET.glob("*.py") if chemin.stem != "__init__"),
)
def test_every_module_of_the_package_imports(module: str) -> None:
    """Chaque module s'importe seul, sans dépendre de l'ordre d'import du paquet."""
    __import__(f"sentinel.{module}")


def test_importing_the_package_loads_no_model() -> None:
    """Importer `sentinel` ne doit charger ni poids YOLO ni interpréteur torch.

    Le paquet est importé par tous les tests et par `download_models.py` ; s'il
    tirait Ultralytics à l'import, la suite passerait de 3 secondes à plusieurs
    dizaines, et le script de téléchargement exigerait ce qu'il est censé
    installer.
    """
    import sys

    assert "ultralytics" not in sys.modules
