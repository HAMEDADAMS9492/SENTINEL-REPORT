"""Les valeurs métier vivent dans `config.py`, et nulle part ailleurs.

Le docstring d'`app.py` promet « aucune logique métier ici — pas de calcul de
durée, pas de règle d'événement, pas de seuil en dur ». La promesse était fausse :
l'interface réécrivait les règles, validait la cadence contre deux constantes
codées sur place, et rangeait les gravités dans son propre ordre.

Ces tests couvrent les deux moitiés du problème. Les premiers vérifient que la
logique déplacée **fonctionne** ; les derniers vérifient qu'elle n'est pas
**revenue** dans l'interface. Un seuil recopié dans `app.py` ne casse rien le
jour où on l'écrit — il diverge silencieusement de `config.py` six mois plus
tard, et le rapport cesse de dire ce que la configuration annonce.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import app
import config

APP = Path(__file__).resolve().parents[1] / "app.py"
ARBRE = ast.parse(APP.read_text(encoding="utf-8"))


def _fonction(nom: str) -> ast.FunctionDef:
    """Nœud AST d'une fonction de premier niveau d'`app.py`."""
    for noeud in ARBRE.body:
        if isinstance(noeud, ast.FunctionDef) and noeud.name == nom:
            return noeud
    raise AssertionError(f"Fonction « {nom} » introuvable dans app.py.")


# ---------------------------------------------------------------------------
# Ordre de gravité
# ---------------------------------------------------------------------------


def test_every_severity_has_a_rank() -> None:
    """Ajouter un niveau sans lui donner de rang doit échouer tout de suite.

    Sans ce test, un `Severity.CRITICAL` ajouté demain lèverait un `KeyError` au
    premier incident — c'est-à-dire en production, pas au chargement.
    """
    for niveau in config.Severity:
        assert isinstance(niveau.rank, int)


def test_ranks_are_strictly_ordered() -> None:
    """Faible < moyenne < élevée, indépendamment de l'ordre de déclaration."""
    assert config.Severity.LOW.rank < config.Severity.MEDIUM.rank
    assert config.Severity.MEDIUM.rank < config.Severity.HIGH.rank


def test_the_worst_severity_is_the_highest_ranked() -> None:
    """L'ordre de la liste fournie n'influe pas sur le résultat."""
    niveaux = [config.Severity.MEDIUM, config.Severity.HIGH, config.Severity.LOW]

    assert config.worst_severity(niveaux) is config.Severity.HIGH
    assert config.worst_severity(list(reversed(niveaux))) is config.Severity.HIGH


def test_an_empty_set_has_no_worst_severity() -> None:
    """Aucun incident n'est différent d'un incident bénin.

    Rendre `LOW` par défaut ferait afficher « gravité faible » sur une session
    sans le moindre incident — une information fausse là où « — » est exact.
    """
    assert config.worst_severity([]) is None


# ---------------------------------------------------------------------------
# Facteur de durées
# ---------------------------------------------------------------------------


def test_a_neutral_factor_returns_the_configuration_untouched() -> None:
    """Facteur 1 : la configuration d'origine, sans copie ni recalcul."""
    assert config.scaled_rules(1.0) is config.EVENT_RULES


@pytest.mark.parametrize("facteur", [0.25, 0.5, 2.0, 3.0])
def test_scaling_preserves_the_anti_bounce_invariant(facteur: float) -> None:
    """`cooldown_s > min_duration_s` doit survivre à tout facteur.

    C'est la raison d'être de la fonction. Ne multiplier que le seuil ferait
    passer le rôdage à 180 s de seuil pour 180 s de garde au facteur 3 : un même
    objet redéclencherait en boucle, et le système produirait un rapport par
    frame.
    """
    for regle in config.scaled_rules(facteur):
        assert regle.cooldown_s > regle.min_duration_s, regle.event_type.value


@pytest.mark.parametrize("facteur", [0.25, 2.0])
def test_scaling_multiplies_both_thresholds(facteur: float) -> None:
    """Les deux seuils sont multipliés, dans le même rapport qu'à l'origine."""
    for origine, ajustee in zip(config.EVENT_RULES, config.scaled_rules(facteur)):
        assert ajustee.min_duration_s == pytest.approx(origine.min_duration_s * facteur)
        assert ajustee.cooldown_s == pytest.approx(origine.cooldown_s * facteur)


def test_scaling_leaves_the_original_rules_intact() -> None:
    """`EVENT_RULES` est la référence : la modifier fausserait toute la session."""
    avant = [(r.min_duration_s, r.cooldown_s) for r in config.EVENT_RULES]

    config.scaled_rules(3.0)

    assert [(r.min_duration_s, r.cooldown_s) for r in config.EVENT_RULES] == avant


@pytest.mark.parametrize("facteur", [0.0, -1.0])
def test_a_non_positive_factor_is_refused(facteur: float) -> None:
    """Un facteur nul rendrait tout déclenchable en permanence."""
    with pytest.raises(ValueError, match="Facteur"):
        config.scaled_rules(facteur)


def test_the_interface_delegates_the_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    """`app.scaled_rules()` n'est qu'un adaptateur : il ne recalcule rien."""
    recu: list[float] = []
    monkeypatch.setattr(config, "scaled_rules", lambda f: recu.append(f) or ())

    app.scaled_rules({"min_duration_factor": 2.5})

    assert recu == [2.5]


# ---------------------------------------------------------------------------
# Vraisemblance de la cadence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("annonce", [0.0, 0.04, 1000.0, -5.0, None])
def test_an_implausible_frame_rate_falls_back_to_the_default(annonce) -> None:
    """Une cadence absurde rendrait toutes les durées fausses.

    Cas réels : une webcam qui n'expose pas la propriété rend 0 — division par
    zéro ; un conteneur mal muxé annonce 0.04 fps ; certains flux RTSP annoncent
    1000 fps.
    """
    assert config.VIDEO.credible_fps(annonce) == config.VIDEO.default_fps


@pytest.mark.parametrize("annonce", [1.0, 25.0, 29.97, 60.0, 240.0])
def test_a_plausible_frame_rate_is_kept(annonce: float) -> None:
    """Une cadence crédible est retenue telle quelle, bornes incluses."""
    assert config.VIDEO.credible_fps(annonce) == annonce


# ---------------------------------------------------------------------------
# Rien n'est revenu en dur dans l'interface
# ---------------------------------------------------------------------------


def test_the_interface_holds_no_frame_rate_bounds() -> None:
    """Les bornes de vraisemblance ne sont plus écrites dans `app.py`."""
    source = APP.read_text(encoding="utf-8")

    assert "240.0" not in source, "Une borne de cadence est codée en dur dans app.py."
    assert "CAP_PROP_FPS" not in source


def test_the_annotation_hardcodes_no_colour() -> None:
    """`annotate()` ne contient aucun triplet BGR littéral.

    La couleur d'alerte vient de la zone, la couleur neutre de `config.UI` :
    changer la charte ne doit demander qu'une seule édition.
    """
    for noeud in ast.walk(_fonction("annotate")):
        if isinstance(noeud, ast.Tuple) and len(noeud.elts) == 3:
            valeurs = [e for e in noeud.elts if isinstance(e, ast.Constant)]
            assert len(valeurs) < 3, f"Couleur codée en dur : {ast.unparse(noeud)}"


def test_the_annotation_asks_the_zone_for_its_alert_colour() -> None:
    """La couleur d'infraction est demandée au `ZoneManager`."""
    appels = [
        ast.unparse(n.func) for n in ast.walk(_fonction("annotate")) if isinstance(n, ast.Call)
    ]

    assert "zone_manager.alert_color_for" in appels


def test_the_interface_does_not_reorder_severities() -> None:
    """Aucun classement de gravités reconstitué dans l'interface."""
    for noeud in ast.walk(_fonction("render_status_strip")):
        if isinstance(noeud, ast.Tuple):
            membres = [
                ast.unparse(e) for e in noeud.elts if ast.unparse(e).startswith("config.Severity")
            ]
            assert len(membres) < 2, "Un second ordre de gravité vit dans app.py."
