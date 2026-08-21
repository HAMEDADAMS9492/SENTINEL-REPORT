"""Tests du score de priorité — arithmétique vérifiable à la main.

Ce module teste une **convention**, pas un algorithme : le barème de
`config.SCORING` est un choix, discutable et ajustable. Ce qui doit être garanti,
c'est que le calcul suit exactement ce barème, que chaque point est justifié, et
que le résultat est reproductible. D'où des attentes chiffrées explicites plutôt
que des comparaisons approximatives : si quelqu'un change une pondération, le
test doit le dire.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path

import pytest

import config
from sentinel.detection import Detection
from sentinel.events import Event, EventEngine, PriorityScore
from sentinel.tracker import TrackedObject
from zone_doubles import FakeZones

JOUR = datetime(2026, 8, 20, 10, 0, 0)  # jeudi 10 h : site ouvert
NUIT = datetime(2026, 8, 20, 23, 30, 0)  # jeudi 23 h 30 : site fermé


class _FakeTracker:
    """Tracker réduit à `nearest()`, jamais sollicité par le scoring."""

    def nearest(self, target, *, class_names, radius_px):
        return None


def _obj(*, class_name: str = "person", raised: dict[str, float] | None = None) -> TrackedObject:
    """Objet suivi, éventuellement porteur de règles déjà déclenchées."""
    detection = Detection(0, class_name, 0.9, (10.0, 10.0, 50.0, 210.0), 1)
    obj = TrackedObject(
        track_id=1,
        class_name=class_name,
        first_seen=0.0,
        last_seen=10.0,
        first_seen_wall=JOUR,
        last_seen_wall=JOUR,
        detection=detection,
    )
    obj.last_event_time = dict(raised or {})
    return obj


def _rule(event_type: config.EventType, **overrides) -> config.EventRule:
    """Règle de la configuration, surchargeable champ par champ."""
    base = next(rule for rule in config.EVENT_RULES if rule.event_type is event_type)
    return config.EventRule(**{**base.__dict__, **overrides})


def _engine(scoring: config.ScoringConfig | None = None) -> EventEngine:
    return EventEngine(FakeZones.restricted("Champ de la caméra"), scoring=scoring)


# ---------------------------------------------------------------------------
# Barème
# ---------------------------------------------------------------------------


def test_base_points_come_from_the_rule_type() -> None:
    """Un objet abandonné pèse plus lourd qu'un rôdage : c'est un choix, publié."""
    bareme = config.SCORING

    assert bareme.points_for(config.EventType.ABANDONED_OBJECT) == 35.0
    assert bareme.points_for(config.EventType.INTRUSION) == 30.0
    assert bareme.points_for(config.EventType.LOITERING) == 20.0
    assert bareme.points_for(config.EventType.AFTER_HOURS) == 15.0


def test_unknown_rule_type_scores_zero_without_crashing() -> None:
    """Ajouter un EventType sans l'inscrire au barème ne doit rien casser."""
    assert config.SCORING.points_for("type_inexistant") == 0.0


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, config.PriorityLevel.LOW),
        (29.9, config.PriorityLevel.LOW),
        (30.0, config.PriorityLevel.MEDIUM),
        (59.9, config.PriorityLevel.MEDIUM),
        (60.0, config.PriorityLevel.HIGH),
        (89.9, config.PriorityLevel.HIGH),
        (90.0, config.PriorityLevel.CRITICAL),
        (500.0, config.PriorityLevel.CRITICAL),
    ],
)
def test_levels_follow_the_published_thresholds(score, expected) -> None:
    """Le score sert au tri, le niveau à la lecture. Les seuils sont dans config."""
    assert config.SCORING.level_for(score) is expected


# ---------------------------------------------------------------------------
# Calcul
# ---------------------------------------------------------------------------


def test_short_intrusion_scores_base_points_only() -> None:
    """Une intrusion de 3 s : 30 points de base, presque rien de durée."""
    score = _engine()._score_event(_obj(), _rule(config.EventType.INTRUSION), 3.0, JOUR)

    # 30 + (3 / 60 x 10) = 30,5
    assert score.value == pytest.approx(30.5)
    assert score.level is config.PriorityLevel.MEDIUM


def test_duration_adds_points_proportionally() -> None:
    """Une intrusion de 5 minutes vaut plus qu'une intrusion de 5 secondes."""
    engine = _engine()
    rule = _rule(config.EventType.INTRUSION)

    courte = engine._score_event(_obj(), rule, 5.0, JOUR)
    longue = engine._score_event(_obj(), rule, 300.0, JOUR)

    assert longue.value > courte.value
    # 30 + min(5 x 10, 30) = 60
    assert longue.value == pytest.approx(60.0)


def test_duration_points_are_capped() -> None:
    """Sans plafond, une présence d'une heure écraserait tous les autres signaux."""
    engine = _engine()
    rule = _rule(config.EventType.INTRUSION)

    dix_minutes = engine._score_event(_obj(), rule, 600.0, JOUR)
    une_heure = engine._score_event(_obj(), rule, 3600.0, JOUR)

    assert dix_minutes.value == une_heure.value == pytest.approx(60.0)


def test_immobility_counts_only_for_rules_that_measure_it() -> None:
    """Attribuer des points d'immobilité à une règle qui ne l'observe pas serait
    inventer un signal — exactement ce que le projet refuse."""
    engine = _engine()

    intrusion = engine._score_event(_obj(), _rule(config.EventType.INTRUSION), 60.0, JOUR)
    abandon = engine._score_event(
        _obj(class_name="backpack"), _rule(config.EventType.ABANDONED_OBJECT), 60.0, JOUR
    )

    assert not any("immobilité" in c for c in intrusion.contributions)
    assert any("immobilité" in c for c in abandon.contributions)


def test_after_hours_multiplies_the_whole_score() -> None:
    """Le même comportement est plus grave la nuit — proportionnellement."""
    engine = _engine()
    rule = _rule(config.EventType.INTRUSION)

    jour = engine._score_event(_obj(), rule, 60.0, JOUR)
    nuit = engine._score_event(_obj(), rule, 60.0, NUIT)

    assert nuit.value == pytest.approx(jour.value * config.SCORING.after_hours_multiplier)
    assert any("hors horaires" in c for c in nuit.contributions)


def test_combining_rules_is_the_strongest_signal() -> None:
    """Une histoire en trois actes pèse plus que trois incidents isolés.

    Rôder, puis pénétrer en zone restreinte, puis abandonner un sac : aucune des
    trois règles ne dit à elle seule ce que leur enchaînement raconte.
    """
    engine = _engine()
    rule = _rule(config.EventType.INTRUSION)

    seul = engine._score_event(_obj(), rule, 60.0, JOUR)
    cumule = engine._score_event(
        _obj(raised={"presence_prolongee": 5.0, "objet_abandonne": 8.0}), rule, 60.0, JOUR
    )

    assert cumule.value == pytest.approx(seul.value + 30.0)
    assert any("2 autre(s)" in c for c in cumule.contributions)


def test_combination_bonus_is_capped() -> None:
    """Le cumul est fort, mais borné : il ne doit pas saturer le barème."""
    beaucoup = {f"regle_{index}": float(index) for index in range(10)}

    score = _engine()._score_event(
        _obj(raised=beaucoup), _rule(config.EventType.INTRUSION), 60.0, JOUR
    )
    contribution = next(c for c in score.contributions if "autre(s)" in c)

    assert f"+{config.SCORING.max_combination_points:.0f}" in contribution


def test_a_critical_case_reaches_the_top_level() -> None:
    """Sac abandonné, longtemps, la nuit, sur un objet déjà signalé."""
    score = _engine()._score_event(
        _obj(class_name="backpack", raised={"presence_prolongee": 1.0}),
        _rule(config.EventType.ABANDONED_OBJECT),
        300.0,
        NUIT,
    )

    # (35 + 30 + 20 + 15) x 1,5 = 150
    assert score.value == pytest.approx(150.0)
    assert score.level is config.PriorityLevel.CRITICAL


# ---------------------------------------------------------------------------
# Traçabilité
# ---------------------------------------------------------------------------


def test_every_score_carries_its_own_justification() -> None:
    """Un score sans justification serait un jugement magique."""
    score = _engine()._score_event(_obj(), _rule(config.EventType.INTRUSION), 45.0, NUIT)

    assert score.contributions, "Aucune justification produite."
    assert "intrusion_zone_restreinte" in score.explanation
    assert "hors horaires" in score.explanation
    assert score.level.value in score.explanation.lower()


def test_the_total_can_be_recomputed_by_hand() -> None:
    """Chaque terme du calcul est publié : l'addition doit retomber sur ses pieds."""
    score = _engine()._score_event(
        _obj(raised={"presence_prolongee": 1.0}),
        _rule(config.EventType.INTRUSION),
        120.0,
        JOUR,
    )

    # 30 (intrusion) + 20 (2 min de présence) + 15 (1 autre règle) = 65
    assert score.value == pytest.approx(65.0)
    assert len(score.contributions) == 3


def test_scoring_is_reproducible() -> None:
    """Deux analyses de la même vidéo doivent produire exactement les mêmes scores."""
    engine = _engine()
    arguments = (
        _obj(raised={"presence_prolongee": 1.0}),
        _rule(config.EventType.INTRUSION),
        77.0,
        NUIT,
    )

    assert engine._score_event(*arguments) == engine._score_event(*arguments)


def test_scale_is_configurable_without_touching_the_code() -> None:
    """Le barème est une convention : la changer ne demande aucune ligne de logique."""
    doux = config.ScoringConfig(
        base_points=((config.EventType.INTRUSION, 5.0),),
        points_per_minute_present=0.0,
        after_hours_multiplier=1.0,
    )

    score = _engine(doux)._score_event(_obj(), _rule(config.EventType.INTRUSION), 300.0, NUIT)

    assert score.value == pytest.approx(5.0)
    assert score.level is config.PriorityLevel.LOW


# ---------------------------------------------------------------------------
# Intégration dans l'incident
# ---------------------------------------------------------------------------


def test_events_carry_their_score_through_the_engine() -> None:
    """Le score voyage avec l'incident, jusqu'au tableau et au CSV."""
    rule = _rule(config.EventType.INTRUSION, min_duration_s=1.0)
    engine = EventEngine(FakeZones.restricted("Champ de la caméra"), rules=[rule])
    obj = _obj()
    obj.zones = {"Champ de la caméra"}
    obj.zone_entry_time = {"Champ de la caméra": 0.0}

    events = engine.evaluate([obj], _FakeTracker(), None, 60.0, JOUR)

    assert len(events) == 1
    incident = events[0]
    assert isinstance(incident.priority, PriorityScore)
    assert incident.priority.value > 0

    ligne = incident.to_dict()
    assert ligne["priorite"] == incident.priority.level.value
    assert ligne["score"] == round(incident.priority.value)
    assert ligne["justification"] == incident.priority.explanation


def test_evidence_capture_preserves_the_score() -> None:
    """L'ajout de la preuve reconstruit l'incident : le score doit survivre.

    C'est le piège que `dataclasses.replace()` élimine — une reconstruction
    champ par champ aurait silencieusement perdu la priorité.
    """
    score = PriorityScore(72.0, config.PriorityLevel.HIGH, ("intrusion (30)",))
    incident = Event(
        event_id="SR-1",
        event_type=config.EventType.INTRUSION,
        severity=config.Severity.HIGH,
        timestamp=JOUR,
        video_time=1.0,
        track_id=1,
        class_name="person",
        zone_name="Champ de la caméra",
        duration_s=5.0,
        confidence=0.9,
        bbox=(1.0, 2.0, 3.0, 4.0),
        priority=score,
    )

    enrichi = dataclass_replace(incident, evidence_path=Path("preuve.jpg"))

    assert enrichi.priority == score
