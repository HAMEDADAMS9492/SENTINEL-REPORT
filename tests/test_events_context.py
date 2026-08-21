"""Le moteur d'événements voit la scène entière, pas un objet isolé.

Ce qui change, et pourquoi
---------------------------
`EventEngine` bouclait `for objet: for règle:` et passait un objet unique à
chaque prédicat. Cette forme interdit **structurellement** toute règle qui
raisonne sur une relation — attroupement, talonnage, franchissement de ligne,
objet déposé par une personne identifiée — puisque le prédicat ne voit jamais le
reste de la scène.

Le moteur détenait pourtant déjà l'ensemble : il reçoit le `Tracker` complet et
possède le `ZoneManager`. Ce qui manquait n'était pas l'accès, mais le contrat.

Ces tests couvrent les deux exigences de la bascule : le contexte donne bien
accès à la scène, et les quatre règles existantes se comportent **exactement**
comme avant — mêmes incidents, dans le même ordre, avec les mêmes identifiants.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import config
from sentinel.detection import Detection
from sentinel.events import EventEngine, FrameContext, RuleOutcome
from sentinel.tracker import TrackedObject
from zone_doubles import FakeZones

OUVERT = datetime(2026, 8, 13, 10, 0, 0)  # jeudi 10 h : site ouvert


class _FakeTracker:
    """Tracker réduit à `nearest()`, seul service utilisé par les règles."""

    def nearest(self, target, *, class_names, radius_px):
        return None


def _obj(
    track_id: int,
    *,
    class_name: str = "person",
    zones: dict[str, float] | None = None,
    age: float = 10.0,
) -> TrackedObject:
    """Objet suivi prêt à l'emploi, éventuellement déjà dans une zone."""
    detection = Detection(0, class_name, 0.9, (100.0, 100.0, 140.0, 200.0), track_id)
    obj = TrackedObject(
        track_id=track_id,
        class_name=class_name,
        first_seen=0.0,
        last_seen=age,
        first_seen_wall=OUVERT,
        last_seen_wall=OUVERT,
        detection=detection,
    )
    for nom, entree in (zones or {}).items():
        obj.zones.add(nom)
        obj.zone_entry_time[nom] = entree
    return obj


def _context(objets, *, zones=None, video_time: float = 10.0) -> FrameContext:
    """Contexte de frame prêt à l'emploi."""
    return FrameContext(
        objects=tuple(objets),
        tracker=_FakeTracker(),
        zones=zones or FakeZones.restricted("Quai"),
        video_time=video_time,
        wall_time=OUVERT,
        frame=None,
    )


# ---------------------------------------------------------------------------
# Le contexte donne accès à la scène
# ---------------------------------------------------------------------------


def test_the_context_exposes_every_object_of_the_frame() -> None:
    """C'est la condition d'existence de toute règle relationnelle."""
    contexte = _context([_obj(1), _obj(2), _obj(3)])

    assert len(contexte.objects) == 3


def test_objects_can_be_filtered_by_class() -> None:
    """« Un sac sans personne à proximité » demande de séparer les classes."""
    contexte = _context([_obj(1), _obj(2, class_name="backpack"), _obj(3)])

    assert [o.track_id for o in contexte.of_class("person")] == [1, 3]
    assert [o.track_id for o in contexte.of_class("backpack")] == [2]


def test_objects_can_be_counted_by_zone() -> None:
    """« Plus de N personnes dans le hall » demande de compter par zone."""
    contexte = _context(
        [
            _obj(1, zones={"Quai": 0.0}),
            _obj(2, zones={"Hall": 0.0}),
            _obj(3, zones={"Quai": 0.0}),
        ]
    )

    assert [o.track_id for o in contexte.in_zone("Quai")] == [1, 3]
    assert contexte.in_zone("Cave") == ()


def test_the_context_lists_the_surveyed_zones() -> None:
    """Une règle qui balaie les zones doit pouvoir les énumérer."""
    contexte = _context([_obj(1)], zones=FakeZones.restricted("Quai", "Hall"))

    assert contexte.zone_names == ("Quai", "Hall")


def test_a_zone_manager_without_names_does_not_break_the_context() -> None:
    """Une doublure minimale reste utilisable : `zone_names` dégrade en vide."""

    class _Minimal:
        def restricted_zone_names(self) -> list[str]:
            return []

    assert _context([_obj(1)], zones=_Minimal()).zone_names == ()


# ---------------------------------------------------------------------------
# Compter n'est pas désigner
# ---------------------------------------------------------------------------


def test_candidates_keeps_only_the_classes_the_rule_targets() -> None:
    """Une règle « person » ne met jamais en cause un sac."""
    regle = config.EventRule(event_type=config.EventType.INTRUSION, classes=("person",))
    contexte = _context([_obj(1), _obj(2, class_name="backpack")])

    assert [o.track_id for o in contexte.candidates(regle)] == [1]


def test_a_rule_without_class_filter_targets_everything() -> None:
    """Un tuple de classes vide signifie « toutes », pas « aucune »."""
    regle = config.EventRule(event_type=config.EventType.INTRUSION, classes=())
    contexte = _context([_obj(1), _obj(2, class_name="backpack")])

    assert len(contexte.candidates(regle)) == 2


def test_candidates_excludes_objects_under_cooldown() -> None:
    """Le délai de garde filtre en amont du prédicat.

    Inutile de recalculer une condition dont on sait déjà que le résultat ne
    pourra pas être publié.
    """
    regle = config.EventRule(event_type=config.EventType.INTRUSION, cooldown_s=60.0)
    recent = _obj(1)
    recent.mark_event(config.EventType.INTRUSION.value, 5.0)

    contexte = _context([recent, _obj(2)], video_time=10.0)

    assert [o.track_id for o in contexte.candidates(regle)] == [2]


def test_counting_ignores_the_cooldown_that_designating_respects() -> None:
    """La distinction qui rendra possible la surdensité.

    Sept personnes constituent un attroupement même si six d'entre elles
    viennent d'être signalées : `in_zone()` compte, `candidates()` désigne.
    """
    regle = config.EventRule(event_type=config.EventType.INTRUSION, cooldown_s=60.0)
    objets = [_obj(i, zones={"Quai": 0.0}) for i in range(1, 8)]
    for obj in objets[:6]:
        obj.mark_event(config.EventType.INTRUSION.value, 5.0)

    contexte = _context(objets, video_time=10.0)

    assert len(contexte.in_zone("Quai")) == 7, "Le comptage doit voir tout le monde."
    assert len(contexte.candidates(regle)) == 1, "La désignation doit filtrer."


def test_candidates_preserves_the_order_of_the_frame() -> None:
    """L'ordre vient du tracker (par `track_id`) : le contexte ne le remanie pas."""
    regle = config.EventRule(event_type=config.EventType.INTRUSION)
    contexte = _context([_obj(7), _obj(2), _obj(5)])

    assert [o.track_id for o in contexte.candidates(regle)] == [7, 2, 5]


# ---------------------------------------------------------------------------
# Les quatre règles existantes sont inchangées
# ---------------------------------------------------------------------------


def test_events_are_emitted_object_by_object_not_rule_by_rule() -> None:
    """L'ordre d'émission reste (objet, règle).

    Les identifiants d'incident sont attribués séquentiellement. Émettre dans
    l'ordre des règles ferait changer `SR-0001` et `SR-0002` de place sur la
    même vidéo, et le rapport cesserait d'être reproductible.
    """
    engine = EventEngine(FakeZones.restricted("Quai"))
    objets = [_obj(1, zones={"Quai": 0.0}), _obj(2, zones={"Quai": 0.0})]

    events = engine.evaluate(objets, _FakeTracker(), None, 10.0, OUVERT)

    assert [e.track_id for e in events] == [1, 2]
    # Les identifiants sont datés puis numérotés : le suffixe doit suivre
    # l'ordre d'émission.
    assert [e.event_id[-4:] for e in events] == ["0001", "0002"]


def test_one_object_triggering_two_rules_keeps_the_rule_order() -> None:
    """À objet égal, l'ordre est celui de `config.EVENT_RULES`.

    Une personne présente depuis 90 s dans une zone restreinte déclenche
    l'intrusion **puis** le rôdage — l'ordre de déclaration des règles.
    """
    engine = EventEngine(FakeZones.restricted("Quai"))
    obj = _obj(1, zones={"Quai": 0.0}, age=90.0)

    events = engine.evaluate([obj], _FakeTracker(), None, 90.0, OUVERT)

    types = [e.event_type for e in events]
    assert types == [config.EventType.INTRUSION, config.EventType.LOITERING]


def test_an_unhandled_rule_type_is_ignored_not_fatal() -> None:
    """Ajouter un `EventType` sans son prédicat ne doit pas arrêter l'analyse.

    Une configuration incomplète est une erreur d'édition ; l'interrompre en
    plein traitement ferait perdre l'analyse déjà produite.
    """

    class _TypeInconnu(str):
        value = "type_sans_predicat"

    regle = config.EventRule(event_type=_TypeInconnu("x"))
    engine = EventEngine(FakeZones.restricted("Quai"), rules=(regle,))

    assert engine.evaluate([_obj(1)], _FakeTracker(), None, 10.0, OUVERT) == []


# ---------------------------------------------------------------------------
# Le point d'extension fonctionne
# ---------------------------------------------------------------------------


def test_a_relational_rule_can_be_plugged_in() -> None:
    """Le test qui justifie la phase : une règle qui compte avant de désigner.

    Cette règle d'attroupement ne se déclenche que si la zone dépasse un seuil
    d'occupation — une condition qu'aucun prédicat mono-objet ne peut exprimer,
    puisqu'il ne voit jamais le reste de la scène.
    """

    def _attroupement(self, rule, context):
        for zone in context.zone_names:
            if len(context.in_zone(zone)) < 3:
                continue
            for obj in context.candidates(rule):
                if zone in obj.zones:
                    yield RuleOutcome(obj, zone, obj.age, {"occupants": len(context.in_zone(zone))})

    regle = config.EventRule(
        event_type=config.EventType.LOITERING, classes=("person",), min_duration_s=0.0
    )
    engine = EventEngine(FakeZones.restricted("Quai"), rules=(regle,))
    engine._PREDICATES = {**EventEngine._PREDICATES, config.EventType.LOITERING: _attroupement}

    deux = [_obj(i, zones={"Quai": 0.0}) for i in (1, 2)]
    trois = [*deux, _obj(3, zones={"Quai": 0.0})]

    assert engine.evaluate(deux, _FakeTracker(), None, 10.0, OUVERT) == []

    leves = engine.evaluate(trois, _FakeTracker(), None, 10.0, OUVERT)
    assert len(leves) == 3
    assert all(e.details["occupants"] == 3 for e in leves)


def test_a_relational_rule_still_respects_the_cooldown() -> None:
    """Une règle collective ne contourne pas l'anti-rebond.

    Le contrôle est refait au point de publication : un prédicat qui désigne un
    sujet tiré de `objects` plutôt que de `candidates()` ne doit pas pouvoir
    republier un incident sous délai de garde.
    """

    def _tout_le_monde(self, rule, context):
        # Volontairement `objects` et non `candidates` : c'est l'erreur que le
        # second contrôle doit rattraper.
        for obj in context.objects:
            yield RuleOutcome(obj, "Quai", obj.age)

    regle = config.EventRule(event_type=config.EventType.INTRUSION, cooldown_s=60.0)
    engine = EventEngine(FakeZones.restricted("Quai"), rules=(regle,))
    engine._PREDICATES = {config.EventType.INTRUSION: _tout_le_monde}

    obj = _obj(1, zones={"Quai": 0.0})

    premier = engine.evaluate([obj], _FakeTracker(), None, 10.0, OUVERT)
    second = engine.evaluate([obj], _FakeTracker(), None, 20.0, OUVERT)

    assert len(premier) == 1
    assert second == [], "Le délai de garde a été contourné."


def test_the_outcome_carries_everything_the_report_needs() -> None:
    """Un constat nomme un sujet : un rapport sans objet n'est pas illustrable."""
    outcome = RuleOutcome(_obj(4), "Quai", 12.0, {"occupants": 5})

    assert outcome.subject.track_id == 4
    assert outcome.zone_name == "Quai"
    assert outcome.duration_s == 12.0
    assert outcome.details == {"occupants": 5}
