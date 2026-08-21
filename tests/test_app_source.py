"""L'interface consomme `VideoSource`, jamais une capture OpenCV brute.

Pourquoi ce fichier existe
---------------------------
`app.py` a longtemps refait à la main ce que `sentinel/source.py` fait
proprement : `cv2.VideoCapture(source)` suivi d'une boucle `read()`. Le résultat
marchait sur un fichier et était **faux en direct** — ni reconnexion après
coupure, ni rattrapage des images accumulées, et surtout un temps métier calculé
en `frame_index / fps` alors qu'en direct des images sont volontairement sautées.
Un objet présent depuis 30 secondes réelles paraissait n'en avoir que 12, et
aucune règle de durée n'était fiable.

La correction est facile à défaire par inadvertance : réintroduire une capture
directe dans `app.py` ne casse **aucun** test fonctionnel, puisque l'analyse d'un
fichier continue de marcher. C'est précisément le genre de régression silencieuse
qui justifie un test structurel.
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

import pytest

import app
import config
from sentinel.source import SourceKind, VideoSource

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
    noms: list[str] = []
    for enfant in ast.walk(noeud):
        if isinstance(enfant, ast.Call):
            noms.append(ast.unparse(enfant.func))
    return noms


# ---------------------------------------------------------------------------
# L'interface n'ouvre plus rien elle-même
# ---------------------------------------------------------------------------


def test_the_interface_never_opens_a_capture_itself() -> None:
    """Aucun `cv2.VideoCapture` dans `app.py`.

    L'ouverture, la cadence et la reconnexion sont la responsabilité de
    `source.py`. Une capture ouverte ici serait un second chemin de lecture, plus
    faible, et invisible aux 29 tests de `test_source.py`.
    """
    assert "cv2.VideoCapture" not in _appels(ARBRE), (
        "app.py ouvre une capture OpenCV en direct : passer par VideoSource."
    )


def test_the_interface_reads_no_capture_property() -> None:
    """Aucune lecture de `CAP_PROP_*` : cadence et longueur viennent de la source."""
    source = APP.read_text(encoding="utf-8")
    assert "CAP_PROP" not in source, (
        "app.py interroge une propriété de capture ; VideoSource.fps / "
        ".total_frames / .progress() exposent déjà ce qu'il faut."
    )


def test_the_analysis_loop_iterates_over_video_source_frames() -> None:
    """`process_video` itère bien sur `source.frames(...)`."""
    appels = _appels(_fonction("process_video"))
    assert "source.frames" in appels, "La boucle n'itère pas sur VideoSource.frames()."


def test_the_source_is_released_by_a_context_manager() -> None:
    """La source est consommée dans un `with`.

    Sans libération, un fichier reste verrouillé sous Windows et une webcam reste
    allumée jusqu'à l'arrêt du processus. Le gestionnaire de contexte la garantit
    même si le pipeline lève au milieu d'une frame.
    """
    corps = _fonction("process_video")
    assert any(isinstance(n, ast.With) for n in ast.walk(corps) if _porte_sur_source(n)), (
        "La source n'est pas ouverte dans un bloc `with`."
    )


def _porte_sur_source(noeud: ast.AST) -> bool:
    """True si le nœud est un `with` dont l'un des sujets s'appelle `source`."""
    if not isinstance(noeud, ast.With):
        return False
    return any(
        isinstance(item.context_expr, ast.Name) and item.context_expr.id == "source"
        for item in noeud.items
    )


def test_the_loop_imposes_the_video_time_carried_by_the_frame() -> None:
    """`Tracker.update()` reçoit `video_time=`, jamais seulement `frame_index`.

    C'est le cœur de la correction : en direct, le compteur d'images ne mesure
    plus le temps écoulé. Laisser le tracker recalculer `index / fps`
    sous-estimerait toutes les durées, donc toutes les règles.
    """
    for noeud in ast.walk(_fonction("process_video")):
        if isinstance(noeud, ast.Call) and ast.unparse(noeud.func) == "tracker.update":
            passes = {mot.arg for mot in noeud.keywords}
            assert "video_time" in passes, "Le temps vidéo de la source n'est pas imposé."
            assert "wall_time" in passes, "L'horodatage réel de la frame n'est pas transmis."
            return
    raise AssertionError("Aucun appel à tracker.update() trouvé.")


def test_a_lost_stream_does_not_discard_the_analysis() -> None:
    """`SourceDisconnectedError` est rattrapée, pas laissée remonter.

    Une coupure réseau après dix minutes de surveillance ne doit pas effacer les
    incidents relevés : l'analyse produite reste valide, seule la suite manque.
    """
    noms = {
        ast.unparse(gestionnaire.type)
        for noeud in ast.walk(_fonction("process_video"))
        if isinstance(noeud, ast.Try)
        for gestionnaire in noeud.handlers
        if gestionnaire.type is not None
    }
    assert "SourceDisconnectedError" in noms


# ---------------------------------------------------------------------------
# `open_source` : la nature du flux est déduite, pas codée dans l'interface
# ---------------------------------------------------------------------------


def test_a_file_path_produces_a_finite_source() -> None:
    """Un chemin de fichier donne une source finie, donc rejouable."""
    source = app.open_source("data/videos/quai.mp4", {})

    assert isinstance(source, VideoSource)
    assert source.kind is SourceKind.FILE
    assert not source.is_live


def test_a_webcam_index_produces_a_live_source() -> None:
    """Un index de webcam donne une source infinie."""
    source = app.open_source(0, {})

    assert source.kind is SourceKind.WEBCAM
    assert source.is_live


def test_a_stream_url_produces_a_live_source() -> None:
    """Une URL RTSP est reconnue comme un direct, sans réglage supplémentaire.

    L'interface ne propose pas encore le flux réseau, mais rien dans `app.py` ne
    l'empêche : la nature est déduite de la désignation par `source.py`.
    """
    source = app.open_source("rtsp://192.168.1.10:554/stream1", {})

    assert source.kind is SourceKind.STREAM
    assert source.is_live


def test_opening_is_deferred_to_the_context_manager() -> None:
    """`open_source()` ne connecte rien : elle décrit, elle n'ouvre pas.

    Une URL injoignable ne doit pas geler le rendu de la page pendant le délai
    de connexion ; l'ouverture appartient au `with` de la boucle d'analyse.
    """
    source = app.open_source("rtsp://adresse-inexistante.invalid/flux", {})

    assert not source.is_open


# ---------------------------------------------------------------------------
# Ligne d'état
# ---------------------------------------------------------------------------


class _SourceFactice:
    """Source minimale : seule sa nature intéresse la ligne d'état."""

    def __init__(self, *, is_live: bool) -> None:
        self.is_live = is_live


class _TrackerFactice:
    """Tracker minimal, sans détecteur ni modèle."""

    video_time = 42.0

    def active(self) -> list[object]:
        return [object(), object()]

    def counts(self) -> dict[str, int]:
        return {"person": 2}


@pytest.mark.parametrize(
    "en_direct, attendu",
    [(False, "temps vidéo"), (True, "temps réel")],
)
def test_the_status_line_names_the_clock_in_use(en_direct: bool, attendu: str) -> None:
    """L'opérateur doit savoir quelle horloge il lit avant de recopier une durée."""
    ligne = app._status_line(_SourceFactice(is_live=en_direct), _TrackerFactice(), 0, 0)

    assert attendu in ligne


def test_dropped_frames_are_reported_only_when_there_are_any() -> None:
    """Le rattrapage du direct se voit ; un fichier n'affiche pas un compteur à zéro."""
    tracker, source = _TrackerFactice(), _SourceFactice(is_live=True)

    assert "écartée" not in app._status_line(source, tracker, 0, 0)
    assert "12 image(s) écartée(s)" in app._status_line(source, tracker, 12, 0)


def test_the_status_line_reports_the_incident_count_it_is_given() -> None:
    """Le compte vient de l'appelant, pas d'une lecture de `st.session_state`."""
    ligne = app._status_line(_SourceFactice(is_live=False), _TrackerFactice(), 0, 7)

    assert "incidents : 7" in ligne
