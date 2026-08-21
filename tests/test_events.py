"""Tests du moteur de règles — sans modèle, sans vidéo, sans écriture disque.

Toutes les conditions métier (durée, immobilité, propriétaire, horaires) sont
alimentées par des `TrackedObject` fabriqués à la main. On peut ainsi tester
« un sac immobile depuis 30 secondes sans personne à moins de 150 px » sans
tourner une seule vidéo.

Les frames sont passées à `None` : `_capture_evidence` refuse alors d'écrire et
retourne `None`, ce qui vérifie au passage qu'une capture ratée **n'annule pas**
l'incident.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import config
from sentinel.detection import Detection
from sentinel.events import Event, EventEngine
from sentinel.tracker import TrackedObject

OPEN_HOURS = datetime(2026, 8, 13, 10, 0, 0)  # jeudi 10 h : site ouvert
NIGHT = datetime(2026, 8, 13, 23, 30, 0)  # jeudi 23 h 30 : fermé
WEEKEND = datetime(2026, 8, 15, 10, 0, 0)  # samedi 10 h : fermé


# ---------------------------------------------------------------------------
# Doublures
# ---------------------------------------------------------------------------


class _FakeZones:
    """ZoneManager réduit à ce que le moteur lui demande."""

    def __init__(self, restricted: list[str]) -> None:
        self._restricted = restricted

    def restricted_zone_names(self) -> list[str]:
        return list(self._restricted)


class _FakeTracker:
    """Tracker réduit à `nearest()`, seul service utilisé par les règles."""

    def __init__(self, neighbour: TrackedObject | None = None) -> None:
        self._neighbour = neighbour

    def nearest(self, target, *, class_names, radius_px):
        return self._neighbour


def _obj(
    *,
    track_id: int = 1,
    class_name: str = "person",
    zones: dict[str, float] | None = None,
    age: float = 10.0,
    positions: list[tuple[float, float]] | None = None,
    box: tuple[float, float, float, float] = (100.0, 100.0, 140.0, 200.0),
) -> TrackedObject:
    """Objet suivi prêt à l'emploi.

    Args:
        zones: `{nom_de_zone: instant_d_entrée}`.
        age: Durée de présence simulée.
        positions: Historique de positions (pour l'immobilité).
        box: Boîte englobante — sa hauteur sert d'échelle aux seuils relatifs.
    """
    detection = Detection(0, class_name, 0.9, box, track_id)
    obj = TrackedObject(
        track_id=track_id,
        class_name=class_name,
        first_seen=0.0,
        last_seen=age,
        first_seen_wall=OPEN_HOURS,
        last_seen_wall=OPEN_HOURS,
        detection=detection,
    )
    if zones:
        obj.zones = set(zones)
        obj.zone_entry_time = dict(zones)
    for index, point in enumerate(positions or [(120.0, 200.0)]):
        obj._push_history(index * (age / max(1, len(positions or [1]))), point)
    return obj


def _engine(restricted: list[str] | None = None, rules=None) -> EventEngine:
    return EventEngine(_FakeZones(restricted or ["Quai"]), rules=rules)


def _rule(event_type: config.EventType, **overrides) -> config.EventRule:
    """Règle isolée, pour tester un seul prédicat à la fois."""
    base = next(r for r in config.EVENT_RULES if r.event_type is event_type)
    return config.EventRule(**{**base.__dict__, **overrides})


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


def test_to_dict_is_serialisable() -> None:
    """Enum, Path et datetime doivent devenir des types simples pour le CSV."""
    event = Event(
        event_id="SR-20260813-0001",
        event_type=config.EventType.INTRUSION,
        severity=config.Severity.HIGH,
        timestamp=OPEN_HOURS,
        video_time=12.345,
        track_id=7,
        class_name="person",
        zone_name="Quai",
        duration_s=4.2,
        confidence=0.876,
        bbox=(1.0, 2.0, 3.0, 4.0),
        details={"deplacement_px": 3.0},
    )

    row = event.to_dict()

    assert row["type"] == "intrusion_zone_restreinte"
    assert row["gravite"] == "elevee"
    assert row["horodatage"] == "2026-08-13T10:00:00"
    assert row["preuve"] == ""
    assert row["detail_deplacement_px"] == 3.0
    assert all(isinstance(value, (str, int, float)) for value in row.values())


# ---------------------------------------------------------------------------
# Intrusion et présence prolongée
# ---------------------------------------------------------------------------


def test_intrusion_requires_the_minimum_duration() -> None:
    """Traverser une zone restreinte ne déclenche rien ; y rester déclenche."""
    engine = _engine(["Quai"])
    rule = _rule(config.EventType.INTRUSION, min_duration_s=3.0)

    crossing = _obj(zones={"Quai": 9.0})  # entré il y a 1 s
    assert engine._check_intrusion(crossing, rule, video_time=10.0) is None

    staying = _obj(zones={"Quai": 5.0})  # entré il y a 5 s
    assert engine._check_intrusion(staying, rule, video_time=10.0) == "Quai"


def test_intrusion_ignores_unrestricted_zones() -> None:
    """La règle d'intrusion ne porte que sur les zones marquées restricted."""
    engine = _engine(restricted=["Quai"])
    rule = _rule(config.EventType.INTRUSION, min_duration_s=3.0)
    obj = _obj(zones={"Hall": 0.0})

    assert engine._check_intrusion(obj, rule, video_time=10.0) is None


def test_loitering_applies_to_any_zone() -> None:
    """Rôder devant une entrée non restreinte reste signalable."""
    engine = _engine(restricted=["Quai"])
    rule = _rule(config.EventType.LOITERING, min_duration_s=60.0)
    obj = _obj(zones={"Hall": 0.0})

    assert engine._check_loitering(obj, rule, video_time=61.0) == "Hall"
    assert engine._check_loitering(obj, rule, video_time=59.0) is None


def test_zone_choice_is_deterministic_across_runs() -> None:
    """Deux analyses de la même vidéo doivent nommer la même zone.

    Sans tri, l'ordre d'itération d'un `set` déciderait, et le rapport ne serait
    pas reproductible.
    """
    engine = _engine(restricted=["Alpha", "Beta"])
    rule = _rule(config.EventType.INTRUSION, min_duration_s=1.0)
    obj = _obj(zones={"Beta": 0.0, "Alpha": 0.0})

    assert engine._check_intrusion(obj, rule, video_time=10.0) == "Alpha"


# ---------------------------------------------------------------------------
# Objet abandonné
# ---------------------------------------------------------------------------


def test_abandoned_object_needs_immobility_and_no_owner() -> None:
    """Les trois conditions se cumulent : âge, immobilité, absence de propriétaire."""
    engine = _engine()
    rule = _rule(config.EventType.ABANDONED_OBJECT, min_duration_s=30.0, max_movement_px=25.0)
    still = [(120.0, 200.0)] * 40

    bag = _obj(class_name="backpack", age=40.0, positions=still)
    assert engine._check_abandoned_object(bag, rule, _FakeTracker(None), 40.0) is True


def test_nearby_person_prevents_the_abandoned_alert() -> None:
    """Un sac au pied de son propriétaire n'est pas abandonné."""
    engine = _engine()
    rule = _rule(config.EventType.ABANDONED_OBJECT, min_duration_s=30.0, max_movement_px=25.0)
    bag = _obj(class_name="backpack", age=40.0, positions=[(120.0, 200.0)] * 40)

    owner = _obj(track_id=2)
    assert engine._check_abandoned_object(bag, rule, _FakeTracker(owner), 40.0) is False


def test_movement_threshold_follows_the_apparent_size() -> None:
    """Le seuil relatif s'adapte à la profondeur ; le seuil en pixels, non.

    Un sac au premier plan (200 px de haut) et le même sac au fond du champ
    (25 px) doivent obéir au **même critère physique** : « a-t-il bougé de plus
    d'un tiers de sa propre taille ? ». Un seuil fixe de 25 px déclarerait le
    premier immobile et le second en pleine traversée.
    """
    engine = _engine()
    rule = _rule(config.EventType.ABANDONED_OBJECT, max_movement_ratio=0.35, max_movement_px=25.0)

    proche = _obj(class_name="backpack", box=(100.0, 0.0, 140.0, 200.0))
    lointain = _obj(class_name="backpack", box=(100.0, 0.0, 140.0, 25.0))

    assert engine._movement_threshold(proche, rule) == pytest.approx(70.0)
    assert engine._movement_threshold(lointain, rule) == pytest.approx(8.75)


def test_pixel_threshold_is_the_fallback() -> None:
    """Sans ratio défini, on retombe sur le seuil en pixels de la configuration."""
    engine = _engine()
    rule = _rule(config.EventType.ABANDONED_OBJECT, max_movement_ratio=None, max_movement_px=25.0)

    assert engine._movement_threshold(_obj(class_name="backpack"), rule) == 25.0


def test_owner_radius_follows_the_apparent_size() -> None:
    """Même correction de perspective sur la recherche du propriétaire."""
    engine = _engine()
    rule = _rule(config.EventType.ABANDONED_OBJECT, owner_radius_ratio=3.0, owner_radius_px=150.0)

    proche = _obj(class_name="backpack", box=(100.0, 0.0, 140.0, 200.0))
    lointain = _obj(class_name="backpack", box=(100.0, 0.0, 140.0, 20.0))

    assert engine._owner_radius(proche, rule) == pytest.approx(600.0)
    assert engine._owner_radius(lointain, rule) == pytest.approx(60.0)


def test_moving_object_is_never_abandoned() -> None:
    """Un sac porté se déplace : la condition d'immobilité l'écarte."""
    engine = _engine()
    rule = _rule(config.EventType.ABANDONED_OBJECT, min_duration_s=30.0, max_movement_px=25.0)
    carried = [(120.0 + 10.0 * i, 200.0) for i in range(40)]
    bag = _obj(class_name="backpack", age=40.0, positions=carried)

    assert engine._check_abandoned_object(bag, rule, _FakeTracker(None), 40.0) is False


# ---------------------------------------------------------------------------
# Hors horaires
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moment, expected",
    [(OPEN_HOURS, False), (NIGHT, True), (WEEKEND, True)],
)
def test_after_hours_uses_wall_clock_not_video_time(moment, expected) -> None:
    """C'est l'heure réelle qui qualifie « hors horaires », pas le temps vidéo."""
    assert _engine().is_after_hours(moment) is expected


# ---------------------------------------------------------------------------
# Anti-rebond et cycle complet
# ---------------------------------------------------------------------------


def test_cooldown_blocks_the_flood_of_duplicate_events() -> None:
    """Une intrusion continue ne doit produire qu'un incident par délai de garde."""
    rule = _rule(config.EventType.INTRUSION, min_duration_s=3.0, cooldown_s=60.0)
    engine = _engine(["Quai"], rules=[rule])
    obj = _obj(zones={"Quai": 0.0})
    tracker = _FakeTracker()

    # 100 frames consécutives en infraction, à 0,1 s d'intervalle.
    raised = []
    for step in range(100):
        video_time = 10.0 + step * 0.1
        raised += engine.evaluate([obj], tracker, None, video_time, OPEN_HOURS)

    assert len(raised) == 1, "Le délai de garde n'a pas filtré les doublons."
    assert raised[0].zone_name == "Quai"
    assert raised[0].evidence_path is None  # frame absente : incident conservé


def test_event_ids_are_sequential_and_dated() -> None:
    """Format attendu : SR-AAAAMMJJ-NNNN, croissant dans la journée."""
    rule = _rule(config.EventType.INTRUSION, min_duration_s=1.0, cooldown_s=0.0)
    engine = _engine(["Quai"], rules=[rule])
    tracker = _FakeTracker()

    first = engine.evaluate([_obj(track_id=1, zones={"Quai": 0.0})], tracker, None, 5.0, OPEN_HOURS)
    second = engine.evaluate([_obj(track_id=2, zones={"Quai": 0.0})], tracker, None, 5.0, OPEN_HOURS)

    assert first[0].event_id == "SR-20260813-0001"
    assert second[0].event_id == "SR-20260813-0002"


def test_rule_ignores_classes_it_does_not_target() -> None:
    """Une règle « person » ne doit jamais se déclencher sur un camion."""
    rule = _rule(config.EventType.INTRUSION, min_duration_s=1.0)
    engine = _engine(["Quai"], rules=[rule])

    truck = _obj(class_name="truck", zones={"Quai": 0.0})
    assert engine.evaluate([truck], _FakeTracker(), None, 10.0, OPEN_HOURS) == []


def test_history_accumulates_and_clears() -> None:
    """L'historique alimente le tableau de bord et l'export CSV."""
    rule = _rule(config.EventType.INTRUSION, min_duration_s=1.0, cooldown_s=0.0)
    engine = _engine(["Quai"], rules=[rule])

    engine.evaluate([_obj(zones={"Quai": 0.0})], _FakeTracker(), None, 5.0, OPEN_HOURS)
    assert len(engine) == 1
    assert len(engine.history) == 1

    engine.clear()
    assert len(engine) == 0
