"""Tests de `sentinel.timeline` — la chronologie ne doit dire que ce qui change.

Une chronologie qui répète « 3 personnes présentes » toutes les 30 secondes
pendant une heure ne se lit pas. Ces tests vérifient donc autant ce qui est
**absent** du résultat que ce qui y figure.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import config
from sentinel.detection import Detection
from sentinel.events import Event, PriorityScore
from sentinel.timeline import SessionContext, Timeline, format_duration
from sentinel.tracker import TrackedObject

T0 = datetime(2026, 8, 20, 23, 0, 0)


def _obj(track_id: int, class_name: str = "person") -> TrackedObject:
    """Objet suivi minimal, identifié."""
    detection = Detection(0, class_name, 0.9, (10.0, 10.0, 50.0, 210.0), track_id)
    return TrackedObject(track_id, class_name, 0.0, 0.0, T0, T0, detection)


def _event(
    track_id: int,
    video_time: float,
    *,
    level: config.PriorityLevel = config.PriorityLevel.MEDIUM,
    event_type: config.EventType = config.EventType.INTRUSION,
) -> Event:
    """Incident porteur d'un score, comme en produit le moteur."""
    return Event(
        event_id=f"SR-{track_id:04d}",
        event_type=event_type,
        severity=config.Severity.HIGH,
        timestamp=T0,
        video_time=video_time,
        track_id=track_id,
        class_name="person",
        zone_name="Champ de la caméra",
        duration_s=30.0,
        confidence=0.87,
        bbox=(1.0, 2.0, 3.0, 4.0),
        priority=PriorityScore(50.0, level, ("intrusion (30)",)),
    )


def _labels(timeline: Timeline) -> list[str]:
    """Tous les libellés de faits, dans l'ordre des tranches."""
    return [fait.label for tranche in timeline.slices() for fait in tranche.facts]


# ---------------------------------------------------------------------------
# Mise en forme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [(0, "00:00"), (45, "00:45"), (90, "01:30"), (3725, "1:02:05"), (-5, "00:00")],
)
def test_durations_are_readable(seconds, expected) -> None:
    """Un rapport se lit en minutes, pas en secondes cumulées."""
    assert format_duration(seconds) == expected


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def test_entries_and_exits_are_recorded_once() -> None:
    """Seules les transitions comptent : ni l'arrivée ni le départ ne se répètent."""
    timeline = Timeline()

    for instant in (1.0, 2.0, 3.0):
        timeline.observe(instant, [_obj(1)])
    for instant in (4.0, 5.0):
        timeline.observe(instant, [])

    labels = _labels(timeline)
    assert sum("entre dans le champ" in label for label in labels) == 1
    assert sum("quitte le champ" in label for label in labels) == 1


def test_unchanged_state_produces_no_fact() -> None:
    """Cent frames identiques ne doivent produire aucune ligne supplémentaire."""
    timeline = Timeline()
    timeline.observe(0.0, [_obj(1)])

    avant = len(timeline)
    for index in range(100):
        timeline.observe(1.0 + index * 0.1, [_obj(1)])

    assert len(timeline) == avant


def test_a_slice_without_transition_is_omitted() -> None:
    """Les tranches muettes n'apparaissent pas : elles allongeraient la lecture."""
    timeline = Timeline()
    timeline.observe(5.0, [_obj(1)])  # tranche 0
    timeline.observe(95.0, [_obj(1), _obj(2)])  # tranche 3

    etiquettes = [tranche.label for tranche in timeline.slices()]

    assert etiquettes == ["00:00 - 00:30", "01:30 - 02:00"]


def test_all_slices_can_be_requested() -> None:
    """Le rapport masque les tranches vides, mais elles restent consultables."""
    timeline = Timeline()
    timeline.observe(5.0, [_obj(1)])
    timeline.observe(95.0, [_obj(1)])

    assert len(timeline.slices(only_notable=False)) == 4


# ---------------------------------------------------------------------------
# Échantillonnage en temps vidéo
# ---------------------------------------------------------------------------


def test_slicing_follows_video_time_not_processing_time() -> None:
    """Même vidéo, machine lente ou rapide : la même chronologie.

    C'est le principe des deux horloges appliqué au rapport — condition pour
    qu'un document d'analyse soit vérifiable.
    """
    timeline = Timeline()

    timeline.observe(29.9, [_obj(1)])
    timeline.observe(30.1, [_obj(1), _obj(2)])

    tranches = timeline.slices()
    assert tranches[0].label == "00:00 - 00:30"
    assert tranches[1].label == "00:30 - 01:00"


def test_slice_width_is_configurable() -> None:
    """La granularité est un réglage, pas une constante enfouie dans le code."""
    timeline = Timeline(config.TimelineConfig(slice_seconds=10.0))

    timeline.observe(5.0, [_obj(1)])
    timeline.observe(15.0, [_obj(1), _obj(2)])

    assert [t.label for t in timeline.slices()] == ["00:00 - 00:10", "00:10 - 00:20"]


# ---------------------------------------------------------------------------
# Incidents et priorité
# ---------------------------------------------------------------------------


def test_incidents_appear_in_their_slice() -> None:
    """Un incident est situé dans le temps, pas seulement listé."""
    timeline = Timeline()

    timeline.observe(45.0, [_obj(1)], [_event(1, 45.0)])

    tranche = timeline.slices()[0]
    assert tranche.label == "00:30 - 01:00"
    assert any("intrusion" in fait.label for fait in tranche.facts)


def test_incidents_are_listed_before_movement() -> None:
    """Un lecteur cherche les incidents, pas les allées et venues."""
    timeline = Timeline()

    timeline.observe(5.0, [_obj(1), _obj(2)], [_event(1, 5.0)])

    natures = [fait.kind for fait in timeline.slices()[0].facts]
    assert natures[0] == "incident"


def test_becoming_high_priority_is_signalled_once() -> None:
    """Le passage en priorité élevée est une transition, pas un état à répéter."""
    timeline = Timeline()

    for instant in (10.0, 40.0, 70.0):
        timeline.observe(
            instant, [_obj(1)], [_event(1, instant, level=config.PriorityLevel.CRITICAL)]
        )

    assert sum("passe en priorité" in label for label in _labels(timeline)) == 1


def test_low_priority_incidents_do_not_raise_a_priority_fact() -> None:
    """Le seuil de signalement est configurable, et il est respecté."""
    timeline = Timeline()

    timeline.observe(10.0, [_obj(1)], [_event(1, 10.0, level=config.PriorityLevel.LOW)])

    assert not any("passe en priorité" in label for label in _labels(timeline))


def test_leaving_the_field_clears_the_priority_flag() -> None:
    """Un objet qui revient après être sorti est de nouveau signalé."""
    timeline = Timeline()
    critique = config.PriorityLevel.CRITICAL

    timeline.observe(10.0, [_obj(1)], [_event(1, 10.0, level=critique)])
    timeline.observe(20.0, [])
    timeline.observe(30.0, [_obj(1)], [_event(1, 30.0, level=critique)])

    assert sum("passe en priorité" in label for label in _labels(timeline)) == 2


def test_a_saturated_slice_is_summarised() -> None:
    """Une tranche qui déborde n'informe plus : le surplus est compté, pas listé."""
    timeline = Timeline(config.TimelineConfig(slice_seconds=30.0, max_facts_per_slice=3))

    timeline.observe(5.0, [_obj(index) for index in range(10)])

    tranche = timeline.slices()[0]
    assert len(tranche.facts) == 3
    assert tranche.omitted == 7


# ---------------------------------------------------------------------------
# Cycle de vie
# ---------------------------------------------------------------------------


def test_timeline_is_readable_while_the_session_continues() -> None:
    """Une surveillance en direct doit pouvoir être exportée sans être interrompue."""
    timeline = Timeline()

    timeline.observe(5.0, [_obj(1)], [_event(1, 5.0)], T0)
    premier = timeline.slices()

    timeline.observe(95.0, [_obj(1), _obj(2)])
    second = timeline.slices()

    assert len(premier) == 1
    assert len(second) == 2, "La chronologie doit continuer de s'enrichir."


def test_reset_clears_everything_between_sessions() -> None:
    """Deux vidéos successives ne doivent pas mélanger leurs chronologies."""
    timeline = Timeline()
    timeline.observe(5.0, [_obj(1)], [_event(1, 5.0)], T0)

    timeline.reset()

    assert len(timeline) == 0
    assert timeline.slices() == []
    assert timeline.duration_s == 0.0
    assert timeline.started_at is None


def test_wall_clock_bounds_are_kept_for_the_report_header() -> None:
    """La période analysée se date à l'heure réelle, pas en temps vidéo."""
    timeline = Timeline()
    fin = datetime(2026, 8, 20, 23, 5, 0)

    timeline.observe(0.0, [_obj(1)], [], T0)
    timeline.observe(300.0, [_obj(1)], [], fin)

    assert timeline.started_at == T0
    assert timeline.ended_at == fin
    assert timeline.duration_s == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Contexte de session
# ---------------------------------------------------------------------------


def test_session_context_summarises_the_active_settings() -> None:
    """L'en-tête du rapport doit dire avec quels réglages l'analyse a tourné."""
    contexte = SessionContext(
        source_label="rtsp://***@camera/stream",
        is_live=True,
        model_label="Équilibré (small)",
        confidence=0.45,
        watched_classes=("person", "backpack"),
        duration_factor=2.0,
    )

    lignes = "\n".join(contexte.settings_lines())

    assert "flux en direct" in lignes
    assert "Équilibré (small)" in lignes
    assert "45%" in lignes
    assert "person, backpack" in lignes
    assert "x 2" in lignes


def test_neutral_duration_factor_is_not_mentioned() -> None:
    """Ne pas encombrer l'en-tête d'un réglage laissé à sa valeur par défaut."""
    contexte = SessionContext(duration_factor=1.0)

    assert not any("Durées ajustées" in ligne for ligne in contexte.settings_lines())
