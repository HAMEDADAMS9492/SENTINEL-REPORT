"""Hygiène des dépendances — ce que le projet installe doit être ce qu'il utilise.

Ces tests n'exercent aucune logique métier : ils vérifient une **cohérence entre
le discours et la facture**. Le README consacre un paragraphe à refuser
`supervision` au motif qu'elle tire ~30 Mo (dont PyAV, un décodeur vidéo
complet) pour un service que `cv2.pointPolygonTest` rend déjà. Tant que la
dépendance restait déclarée dans `requirements.txt`, cet argumentaire était faux
en pratique : le coût était payé, le bénéfice nul.

Un test rend la régression impossible. Sans lui, la ligne reviendrait au premier
copier-coller d'un `requirements.txt` d'un autre projet, et personne ne le
remarquerait — une dépendance en trop ne casse rien, elle coûte seulement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent

# Fichiers de code du projet, tests inclus : une dépendance réintroduite dans un
# test est aussi coûteuse à installer qu'une dépendance réintroduite en
# production.
SOURCES = sorted(
    [*RACINE.glob("*.py"), *(RACINE / "sentinel").glob("*.py"), *(RACINE / "tests").glob("*.py")]
)

# Librairies explicitement refusées, avec la raison du refus. Le message
# d'échec doit rappeler l'argument : un test qui dit seulement « interdit »
# invite à lever l'interdiction.
REFUSEES: dict[str, str] = {
    "supervision": (
        "~30 Mo (dont PyAV) pour ce que cv2.pointPolygonTest fait déjà — "
        "voir l'en-tête de sentinel/zones.py."
    ),
}


@pytest.mark.parametrize("chemin", SOURCES, ids=lambda p: p.name)
def test_no_refused_library_is_imported(chemin: Path) -> None:
    """Aucun fichier n'importe une librairie que le projet a refusée."""
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        nettoyee = ligne.strip()
        for nom, raison in REFUSEES.items():
            interdits = (f"import {nom}", f"from {nom} import", f"from {nom}.")
            assert not nettoyee.startswith(interdits), (
                f"{chemin.name} importe « {nom} », refusée : {raison}"
            )


def test_refused_libraries_are_absent_from_requirements() -> None:
    """`requirements.txt` ne déclare aucune librairie refusée.

    La recherche ignore les lignes de commentaire : le fichier **explique**
    pourquoi `supervision` a été retirée, et cette explication a autant de valeur
    que le retrait lui-même.
    """
    lignes = [
        ligne.strip()
        for ligne in (RACINE / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if ligne.strip() and not ligne.strip().startswith("#")
    ]

    for nom, raison in REFUSEES.items():
        for ligne in lignes:
            assert not ligne.lower().startswith(nom), f"« {nom} » est déclarée : {raison}"


def test_conversion_helper_is_gone() -> None:
    """`to_supervision()` n'existe plus.

    C'était la seule porte d'entrée vers la librairie. La laisser « au cas où »
    aurait obligé à garder la dépendance déclarée, donc installée : une fonction
    sans appelant ne justifie pas 30 Mo.
    """
    from sentinel import detection

    assert not hasattr(detection, "to_supervision")
