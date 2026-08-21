"""L'interface produit le rapport de session, pas seulement des rapports d'incident.

`timeline.py` et `session_report.py` étaient écrits, testés — et jamais importés
par `app.py`. Un module fini que l'exécutable n'appelle pas n'existe pas pour
l'utilisateur : le README annonçait un rapport à deux niveaux que personne ne
pouvait obtenir.

Ces tests vérifient le **branchement** : que la chronologie est alimentée
pendant l'analyse, qu'elle survit à la boucle, et que le rapport s'affiche même
sans incident — une surveillance calme est un résultat, et la chronologie le
documente.
"""

from __future__ import annotations

import ast
import typing
from datetime import datetime
from pathlib import Path

import pytest

import app
import config
from sentinel.session_report import SessionReportGenerator
from sentinel.timeline import SessionContext, Timeline

APP = Path(__file__).resolve().parents[1] / "app.py"
ARBRE = ast.parse(APP.read_text(encoding="utf-8"))


def _fonction(nom: str) -> ast.FunctionDef:
    """Nœud AST d'une fonction de premier niveau d'`app.py`."""
    for noeud in ARBRE.body:
        if isinstance(noeud, ast.FunctionDef) and noeud.name == nom:
            return noeud
    raise AssertionError(f"Fonction « {nom} » introuvable dans app.py.")


def _appels(noeud: ast.AST) -> list[str]:
    """Noms qualifiés de tous les appels contenus dans un nœud."""
    return [
        ast.unparse(enfant.func) for enfant in ast.walk(noeud) if isinstance(enfant, ast.Call)
    ]


class _SourceFactice:
    """Source minimale : seuls sa nature et son libellé intéressent le contexte."""

    def __init__(self, label: str, *, is_live: bool) -> None:
        self.label = label
        self.is_live = is_live


# ---------------------------------------------------------------------------
# Le pipeline porte la chronologie
# ---------------------------------------------------------------------------


def test_the_pipeline_names_its_components() -> None:
    """`pipeline[4]` ne dit pas ce qu'il désigne, et l'ordre finit par changer."""
    assert app.Pipeline._fields == (
        "detector",
        "tracker",
        "zones",
        "events",
        "generator",
        "timeline",
    )


def test_the_pipeline_carries_a_timeline() -> None:
    """La chronologie fait partie de l'état d'une analyse, comme le tracker.

    Elle est donc reconstruite à chaque session : la conserver d'un rerun à
    l'autre ferait démarrer une nouvelle vidéo avec les faits de la précédente.
    """
    # `from __future__ import annotations` transforme les annotations en
    # chaînes : `get_type_hints` les résout en vraies classes.
    assert typing.get_type_hints(app.Pipeline)["timeline"] is Timeline


def test_the_generator_can_produce_session_reports() -> None:
    """Le générateur du pipeline sait faire les deux sortes de rapport.

    `SessionReportGenerator` hérite de `ReportGenerator` : les rapports
    d'incident isolés restent inchangés, la structure de session s'ajoute.
    """
    assert typing.get_type_hints(app.Pipeline)["generator"] is SessionReportGenerator


# ---------------------------------------------------------------------------
# La chronologie est alimentée pendant l'analyse
# ---------------------------------------------------------------------------


def test_the_loop_feeds_the_timeline() -> None:
    """`process_video` appelle `timeline.observe()` à chaque frame traitée."""
    assert "timeline.observe" in _appels(_fonction("process_video"))


def test_the_timeline_observes_retained_objects_not_visible_ones() -> None:
    """`observe()` reçoit `tracker.active()`, pas la liste de la frame courante.

    La rétention du tracker absorbe déjà les occlusions courtes. Lui passer les
    objets vus sur cette seule frame produirait une fausse sortie suivie d'une
    fausse entrée chaque fois qu'un objet passe derrière un poteau — et la
    chronologie deviendrait illisible.
    """
    for noeud in ast.walk(_fonction("process_video")):
        if isinstance(noeud, ast.Call) and ast.unparse(noeud.func) == "timeline.observe":
            arguments = [ast.unparse(a) for a in noeud.args]
            assert "tracker.active()" in arguments, arguments
            return
    raise AssertionError("Aucun appel à timeline.observe() trouvé.")


def test_the_timeline_survives_the_analysis_loop() -> None:
    """La chronologie est rangée dans la session, pas jetée avec le pipeline.

    C'est ce qui permet de rééditer le rapport à chaque rerun — un filtre
    déplacé, une case cochée — sans relancer une analyse de plusieurs minutes.
    """
    # `ast.unparse` normalise les guillemets : on cherche donc les fragments
    # sans présumer de leur forme littérale dans le fichier.
    corps = ast.unparse(_fonction("main"))

    assert "st.session_state['timeline'] = pipeline.timeline" in corps
    assert "st.session_state['context'] = session_context" in corps


def test_the_session_report_is_rendered_from_main() -> None:
    """Le rapport de session est effectivement affiché."""
    assert "render_session_report" in _appels(_fonction("main"))


def test_the_session_report_does_not_require_an_incident() -> None:
    """Une surveillance calme mérite un rapport : la chronologie la documente.

    Le rapport s'affiche « si une analyse a eu lieu », pas « s'il y a des
    incidents ». Le vérifier structurellement évite qu'un `if events:` ne
    l'englobe à la faveur d'un remaniement.
    """
    corps = _fonction("main")
    for noeud in ast.walk(corps):
        if isinstance(noeud, ast.If) and "render_session_report" in _appels(noeud):
            condition = ast.unparse(noeud.test)
            assert "events" not in condition, (
                f"Le rapport de session est conditionné aux incidents : {condition}"
            )
            return
    raise AssertionError("Appel à render_session_report() introuvable dans main().")


# ---------------------------------------------------------------------------
# Contexte de session
# ---------------------------------------------------------------------------


def test_the_context_never_carries_a_stream_password() -> None:
    """Un rapport se transmet : les identifiants RTSP n'ont rien à y faire."""
    source = _SourceFactice("rtsp://***@10.0.0.1/stream1", is_live=True)

    contexte = app.session_context(
        source, {"weights": "", "confidence": 0.4, "classes": ["person"]}
    )

    assert "secret" not in contexte.source_label
    assert contexte.source_label == "rtsp://***@10.0.0.1/stream1"


def test_the_context_records_the_settings_used() -> None:
    """Un rapport qui tait ses réglages n'est pas vérifiable.

    Le même quai filmé à 25 % puis à 75 % de seuil de confiance ne produit pas
    les mêmes incidents ; l'en-tête doit permettre de refaire l'analyse.
    """
    source = _SourceFactice("quai.mp4", is_live=False)

    contexte = app.session_context(
        source,
        {
            "weights": "",
            "confidence": 0.45,
            "classes": ["person", "backpack"],
            "min_duration_factor": 2.0,
        },
    )
    lignes = "\n".join(contexte.settings_lines())

    assert "45%" in lignes
    assert "person, backpack" in lignes
    assert "x 2" in lignes


def test_a_live_context_says_the_period_is_still_open() -> None:
    """« Période analysée » ne se lit pas pareil sur un direct : elle n'est pas close."""
    source = _SourceFactice("webcam n°0", is_live=True)

    contexte = app.session_context(source, {"weights": "", "confidence": 0.4, "classes": []})

    assert contexte.is_live
    assert "flux en direct" in "\n".join(contexte.settings_lines())


def test_an_unknown_model_does_not_break_the_header() -> None:
    """Un chemin de poids hors catalogue ne doit pas faire échouer le rapport."""
    source = _SourceFactice("quai.mp4", is_live=False)

    contexte = app.session_context(
        source, {"weights": "/ailleurs/mon-modele.pt", "confidence": 0.4, "classes": []}
    )

    assert contexte.model_label == "-"


# ---------------------------------------------------------------------------
# Le rapport reste consultable en cours de session
# ---------------------------------------------------------------------------


def test_a_report_can_be_edited_while_the_timeline_keeps_growing() -> None:
    """Exporter un état ne fige pas la surveillance.

    C'est la condition du mode continu : un rapport édité à un instant T, puis
    un autre plus tard, chacun reflétant l'état à son instant d'édition.
    """
    timeline = Timeline()
    generateur = SessionReportGenerator()
    contexte = SessionContext(source_label="webcam n°0", is_live=True)

    timeline.observe(5.0, [], [], datetime(2026, 8, 21, 9, 0, 0))
    premier = generateur.generate_session_report([], timeline, contexte)

    timeline.observe(95.0, [], [], datetime(2026, 8, 21, 9, 1, 30))
    second = generateur.generate_session_report([], timeline, contexte)

    assert "surveillance en cours" in premier
    assert len(second) >= len(premier)


def test_the_cache_key_follows_the_state_of_the_session() -> None:
    """Les exports sont mémoïsés sur ce qui les fait changer, et sur rien d'autre.

    Streamlit exclut de la clé de cache tout argument préfixé d'un `_`. Le
    nombre de faits doit donc **ne pas** l'être : sinon un rapport édité en début
    d'analyse serait resservi tel quel une heure plus tard.
    """
    import inspect

    for fonction in (app._session_pdf, app._timeline_csv):
        parametres = list(inspect.signature(fonction.__wrapped__).parameters)
        assert "fact_count" in parametres, (
            f"{fonction.__wrapped__.__name__} : le nombre de faits est exclu du cache."
        )


def test_slice_width_is_announced_from_the_configuration() -> None:
    """La granularité affichée à l'écran vient de `config.TIMELINE`, pas d'un littéral."""
    corps = ast.unparse(_fonction("render_session_report"))

    assert "config.TIMELINE.slice_seconds" in corps
