"""Tests du rapport de session à deux niveaux.

Le rapport est la seule sortie que lira un humain non technique. Trois
propriétés comptent plus que les autres et sont testées explicitement :

* la partie 1 est **triée par priorité**, de façon totale et reproductible ;
* la partie 2 ne dit **que ce qui change** ;
* la mention « BROUILLON » est **toujours** présente.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import config
from sentinel.events import Event, PriorityScore
from sentinel.exceptions import ReportGenerationError
from sentinel.session_report import (
    MENTION_BROUILLON,
    SessionReportGenerator,
    sort_by_priority,
)
from sentinel.timeline import SessionContext, Timeline

T0 = datetime(2026, 8, 20, 23, 0, 0)


def _event(
    track_id: int,
    score: float,
    level: config.PriorityLevel,
    *,
    video_time: float = 10.0,
    event_type: config.EventType = config.EventType.INTRUSION,
) -> Event:
    """Incident porteur d'un score déjà calculé par le moteur."""
    return Event(
        event_id=f"SR-{track_id:04d}",
        event_type=event_type,
        severity=config.Severity.HIGH,
        timestamp=T0,
        video_time=video_time,
        track_id=track_id,
        class_name="person",
        zone_name="Champ de la caméra",
        duration_s=45.0,
        confidence=0.87,
        bbox=(1.0, 2.0, 3.0, 4.0),
        priority=PriorityScore(score, level, (f"{event_type.value} ({score:.0f})",)),
    )


@pytest.fixture()
def generator(tmp_path) -> SessionReportGenerator:
    """Générateur écrivant dans un dossier temporaire."""
    return SessionReportGenerator(config.ReportConfig(directory=tmp_path))


@pytest.fixture()
def timeline() -> Timeline:
    """Chronologie contenant deux tranches de faits."""
    from sentinel.detection import Detection
    from sentinel.tracker import TrackedObject

    def objet(track_id: int) -> TrackedObject:
        detection = Detection(0, "person", 0.9, (10.0, 10.0, 50.0, 210.0), track_id)
        return TrackedObject(track_id, "person", 0.0, 0.0, T0, T0, detection)

    chronologie = Timeline()
    chronologie.observe(5.0, [objet(1)], [], T0)
    chronologie.observe(95.0, [objet(1), objet(2)], [_event(2, 46.0, config.PriorityLevel.MEDIUM)], T0)
    return chronologie


@pytest.fixture()
def events() -> list[Event]:
    """Trois incidents de priorités différentes, volontairement mal ordonnés."""
    return [
        _event(2, 46.0, config.PriorityLevel.MEDIUM, video_time=20.0),
        _event(1, 150.0, config.PriorityLevel.CRITICAL, video_time=95.0,
               event_type=config.EventType.ABANDONED_OBJECT),
        _event(3, 72.0, config.PriorityLevel.HIGH, video_time=60.0,
               event_type=config.EventType.LOITERING),
    ]


# ---------------------------------------------------------------------------
# Tri
# ---------------------------------------------------------------------------


def test_incidents_are_sorted_by_decreasing_priority(events) -> None:
    """Le plus suspect en haut : c'est la raison d'être du score."""
    assert [e.priority.value for e in sort_by_priority(events)] == [150.0, 72.0, 46.0]


def test_sorting_is_total_and_reproducible() -> None:
    """À score égal, l'ordre doit rester le même d'une exécution à l'autre.

    Sans règle de départage, deux exports du même rapport pourraient présenter
    les incidents dans un ordre différent — et le document cesserait d'être
    vérifiable.
    """
    egaux = [
        _event(1, 50.0, config.PriorityLevel.MEDIUM, video_time=10.0),
        _event(2, 50.0, config.PriorityLevel.MEDIUM, video_time=90.0),
        _event(3, 50.0, config.PriorityLevel.MEDIUM, video_time=50.0),
    ]

    premier = [e.event_id for e in sort_by_priority(egaux)]
    second = [e.event_id for e in sort_by_priority(list(reversed(egaux)))]

    assert premier == second
    # À score égal, le plus récent d'abord.
    assert premier[0] == "SR-0002"


def test_sorting_does_not_modify_the_input(events) -> None:
    """Trier pour afficher ne doit pas réordonner l'historique de la session."""
    avant = [e.event_id for e in events]

    sort_by_priority(events)

    assert [e.event_id for e in events] == avant


def test_events_without_a_score_end_up_last() -> None:
    """Un incident sans score ne doit pas remonter en tête par accident."""
    sans_score = Event(
        event_id="SR-9999",
        event_type=config.EventType.INTRUSION,
        severity=config.Severity.LOW,
        timestamp=T0,
        video_time=1.0,
        track_id=9,
        class_name="person",
        zone_name=None,
        duration_s=1.0,
        confidence=0.5,
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    avec_score = _event(1, 10.0, config.PriorityLevel.LOW)

    assert sort_by_priority([sans_score, avec_score])[0] is avec_score


# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------


def test_header_states_period_source_and_settings(generator, events, timeline) -> None:
    """L'en-tête doit permettre de savoir ce qui a été analysé, et comment."""
    contexte = SessionContext(
        source_label="quai.mp4",
        model_label="Rapide (nano)",
        confidence=0.35,
        watched_classes=("person",),
    )

    lignes = "\n".join(
        generator.header_lines(events, timeline, contexte, generated_at=T0)
    )

    assert "RAPPORT DE SESSION" in lignes
    assert "quai.mp4" in lignes
    assert "Rapide (nano)" in lignes
    assert "35%" in lignes
    assert "20/08/2026" in lignes
    assert "Incidents         : 3" in lignes


def test_header_counts_incidents_by_priority(generator, events, timeline) -> None:
    """Le décompte par niveau donne la mesure de la session en un coup d'œil."""
    lignes = "\n".join(generator.header_lines(events, timeline, generated_at=T0))

    assert "critique : 1" in lignes
    assert "eleve : 1" in lignes
    assert "moyen : 1" in lignes


def test_live_sessions_are_marked_as_ongoing(generator, events, timeline) -> None:
    """Une surveillance en cours n'a pas de période close : il faut le dire."""
    contexte = SessionContext(source_label="rtsp://***@camera", is_live=True)

    lignes = "\n".join(generator.header_lines(events, timeline, contexte, generated_at=T0))

    assert "surveillance en cours" in lignes


# ---------------------------------------------------------------------------
# Structure du rapport
# ---------------------------------------------------------------------------


def test_report_has_exactly_two_parts(generator, events, timeline) -> None:
    """Deux ordres de lecture : par urgence, puis par déroulé."""
    rapport = generator.generate_session_report(events, timeline, generated_at=T0)

    assert "PARTIE 1 - INCIDENTS PRIORITAIRES" in rapport
    assert "PARTIE 2 - CHRONOLOGIE" in rapport
    assert rapport.index("PARTIE 1") < rapport.index("PARTIE 2")


def test_part_one_lists_incidents_by_priority(generator, events, timeline) -> None:
    """Le rang dans la liste est le rang de priorité."""
    rapport = generator.generate_session_report(events, timeline, generated_at=T0)

    assert rapport.index("1. [CRITIQUE]") < rapport.index("2. [ELEVE]") or (
        rapport.index("1. [CRITIQUE]") < rapport.index("2. [ÉLEVÉ]")
    )
    assert rapport.index("SR-0001") < rapport.index("SR-0002")


def test_each_incident_carries_its_score_justification(generator, events, timeline) -> None:
    """Un classement sans justification serait un jugement magique."""
    rapport = generator.generate_session_report(events, timeline, generated_at=T0)

    assert "Score         :" in rapport
    assert "Critique (150 pts)" in rapport


def test_part_two_omits_unchanged_slices(generator, events, timeline) -> None:
    """La chronologie ne répète pas un état inchangé."""
    rapport = generator.generate_session_report(events, timeline, generated_at=T0)

    assert "[00:00 - 00:30]" in rapport
    assert "[01:30 - 02:00]" in rapport
    assert "[00:30 - 01:00]" not in rapport, "Une tranche muette a été listée."


def test_the_draft_warning_is_always_present(generator, timeline) -> None:
    """La mention est non négociable : le rapport n'est pas une preuve."""
    avec = generator.generate_session_report(
        [_event(1, 10.0, config.PriorityLevel.LOW)], timeline, generated_at=T0
    )
    sans = generator.generate_session_report([], Timeline(), generated_at=T0)

    assert MENTION_BROUILLON in avec
    assert MENTION_BROUILLON in sans
    assert "n'identifie personne" in sans


def test_an_empty_session_still_produces_a_valid_report(generator) -> None:
    """Une nuit sans incident est une information, pas une erreur."""
    rapport = generator.generate_session_report([], Timeline(), generated_at=T0)

    assert "Aucun incident détecté" in rapport
    assert "Aucun mouvement enregistré" in rapport


def test_report_is_deterministic(generator, events, timeline) -> None:
    """Deux éditions du même état doivent donner exactement le même document."""
    premier = generator.generate_session_report(events, timeline, generated_at=T0)
    second = generator.generate_session_report(events, timeline, generated_at=T0)

    assert premier == second


# ---------------------------------------------------------------------------
# Mode continu
# ---------------------------------------------------------------------------


def test_report_can_be_generated_while_the_session_runs(generator, timeline) -> None:
    """En direct, on exporte un état à un instant donné sans arrêter le flux."""
    incidents = [_event(1, 46.0, config.PriorityLevel.MEDIUM)]

    intermediaire = generator.generate_session_report(incidents, timeline, generated_at=T0)

    incidents.append(_event(2, 150.0, config.PriorityLevel.CRITICAL, video_time=200.0))
    timeline.observe(200.0, [], [incidents[-1]], T0)
    final = generator.generate_session_report(incidents, timeline, generated_at=T0)

    assert "Incidents         : 1" in intermediaire
    assert "Incidents         : 2" in final
    assert "CRITIQUE" not in intermediaire
    assert "CRITIQUE" in final


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_session_pdf_is_written(generator, events, timeline, tmp_path) -> None:
    """Le PDF doit exister, être non vide et porter l'en-tête du format."""
    chemin = generator.export_session_pdf(events, timeline, generated_at=T0)

    assert chemin.is_file()
    assert chemin.parent == tmp_path
    assert chemin.read_bytes().startswith(b"%PDF")


def test_session_pdf_survives_an_empty_session(generator, tmp_path) -> None:
    """Aucun incident ne doit pas empêcher de produire le document de session."""
    chemin = generator.export_session_pdf([], Timeline(), generated_at=T0)

    assert chemin.read_bytes().startswith(b"%PDF")


def test_timeline_csv_has_one_row_per_fact(generator, timeline) -> None:
    """Le CSV de chronologie se prête à la reconstitution d'un déroulé."""
    chemin = generator.export_timeline_csv(timeline)
    contenu = chemin.read_text(encoding="utf-8-sig").splitlines()

    assert contenu[0] == "tranche,debut_s,temps_video_s,nature,fait"
    assert len(contenu) - 1 == len(timeline)


def test_timeline_csv_starts_with_a_bom_for_excel(generator, timeline) -> None:
    """Sans BOM, Excel casse les accents à l'ouverture d'un CSV UTF-8."""
    assert generator.export_timeline_csv(timeline).read_bytes().startswith(b"\xef\xbb\xbf")


def test_exporting_an_empty_timeline_is_an_explicit_error(generator) -> None:
    """Exporter une chronologie vide est une erreur d'usage, pas un fichier vide."""
    with pytest.raises(ReportGenerationError, match="Aucun fait"):
        generator.export_timeline_csv(Timeline())


def test_incident_csv_now_carries_the_priority(generator, events) -> None:
    """L'export existant s'enrichit du score, sans changer de structure."""
    chemin = generator.export_csv(events)
    entete = chemin.read_text(encoding="utf-8-sig").splitlines()[0]

    assert "priorite" in entete
    assert "score" in entete
    assert "justification" in entete


# ---------------------------------------------------------------------------
# Séparation des responsabilités
# ---------------------------------------------------------------------------


def test_the_report_never_computes_a_score() -> None:
    """`report.py` trie et affiche ; il ne juge pas.

    Le score est une appréciation métier, calculée par `EventEngine`. Si ce
    module se mettait à en produire, la même donnée serait calculée à deux
    endroits — et divergerait tôt ou tard.
    """
    from pathlib import Path as _Path

    source = (_Path(__file__).resolve().parents[1] / "sentinel" / "session_report.py").read_text(
        encoding="utf-8"
    )

    assert "PriorityScore(" not in source
    assert "points_for" not in source
    assert "level_for" not in source
