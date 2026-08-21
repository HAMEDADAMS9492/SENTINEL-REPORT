"""Occupation des zones et règle de surdensité.

Ce que débloque un simple comptage
------------------------------------
« Sept personnes dans le hall depuis douze secondes » n'est la propriété d'aucun
objet : c'est une propriété de la **scène**. Aucun prédicat mono-objet ne pouvait
l'exprimer, et c'est précisément ce que l'ouverture du moteur (phase 1) rendait
possible.

Un seul mécanisme sert deux usages : la règle de surdensité, et les transitions
de la chronologie — « hall : 2 → 7 personnes », un fait marquant qu'aucune règle
par objet ne saurait formuler.

Pourquoi une classe distincte de `ZoneManager`
-----------------------------------------------
`ZoneManager` est **sans mémoire** : ses réponses ne dépendent que de la frame
courante, ce qui le rend simple à raisonner et à tester. L'occupation est de
nature opposée — une durée ne se constate que dans le temps. Les séparer garde la
géométrie géométrique, et rend cette mémoire testable sans polygone, sans image
et sans OpenCV : aucun test de ce fichier n'appelle `initialize()`.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import config
from sentinel.detection import Detection
from sentinel.events import EventEngine
from sentinel.occupancy import OccupancyChange, ZoneOccupancy, summarise
from sentinel.timeline import Timeline
from sentinel.tracker import TrackedObject
from zone_doubles import FakeZones, zone

OUVERT = datetime(2026, 8, 13, 10, 0, 0)  # jeudi 10 h : site ouvert


class _FakeTracker:
    """Tracker réduit à `nearest()`, seul service utilisé par les règles."""

    def nearest(self, target, *, class_names, radius_px):
        return None


def _obj(track_id: int, *, class_name: str = "person", zones: set[str] | None = None):
    """Objet suivi minimal, déjà placé dans ses zones."""
    detection = Detection(0, class_name, 0.9, (100.0, 100.0, 140.0, 200.0), track_id)
    obj = TrackedObject(track_id, class_name, 0.0, 100.0, OUVERT, OUVERT, detection)
    for nom in zones or ():
        obj.zones.add(nom)
        obj.zone_entry_time[nom] = 0.0
    return obj


def _foule(nombre: int, *, zone_name: str = "Hall", depart: int = 1, classe: str = "person"):
    """`nombre` objets d'une même classe, tous dans la même zone."""
    return [
        _obj(depart + index, class_name=classe, zones={zone_name})
        for index in range(nombre)
    ]


# ---------------------------------------------------------------------------
# Comptage
# ---------------------------------------------------------------------------


def test_the_count_is_per_zone_and_per_class() -> None:
    """Trois personnes et un sac ne font pas quatre personnes."""
    occupation = ZoneOccupancy()

    occupation.update(
        [*_foule(3), _obj(9, class_name="backpack", zones={"Hall"})], video_time=1.0
    )

    assert occupation.count("Hall", "person") == 3
    assert occupation.count("Hall", "backpack") == 1
    assert occupation.count("Hall") == 4, "Sans classe précisée, on compte tout."


def test_an_object_in_two_zones_is_counted_in_both() -> None:
    """Une personne à cheval sur deux zones occupe bien les deux."""
    occupation = ZoneOccupancy()

    occupation.update([_obj(1, zones={"Hall", "Quai"})], video_time=1.0)

    assert occupation.count("Hall", "person") == 1
    assert occupation.count("Quai", "person") == 1


def test_an_unknown_zone_counts_zero() -> None:
    """Interroger une zone vide ou inexistante ne doit pas lever."""
    occupation = ZoneOccupancy()

    assert occupation.count("Cave") == 0
    assert occupation.counts_for("Cave") == {}


def test_identities_are_counted_not_detections() -> None:
    """Le comptage s'appuie sur `track_id`, pas sur le nombre de boîtes.

    Compter des détections donnerait un nombre qui bat au rythme du détecteur :
    une personne perdue puis retrouvée compterait deux fois. C'est la même
    raison qui rend le tracker indispensable au chronométrage.
    """
    occupation = ZoneOccupancy()
    memes = [_obj(1, zones={"Hall"}), _obj(1, zones={"Hall"})]

    occupation.update(memes, video_time=1.0)

    assert occupation.count("Hall", "person") == 2, (
        "Le comptage additionne ce qu'on lui donne ; c'est au tracker de ne "
        "fournir qu'un objet par identité."
    )


# ---------------------------------------------------------------------------
# Durée au-dessus d'un seuil
# ---------------------------------------------------------------------------


def test_the_threshold_clock_starts_when_the_level_is_reached() -> None:
    """« Depuis combien de temps sommes-nous à cinq ? » est la question utile."""
    occupation = ZoneOccupancy()

    occupation.update(_foule(5), video_time=10.0)
    occupation.update(_foule(5), video_time=25.0)

    assert occupation.duration_at_least("Hall", "person", 5, 25.0) == pytest.approx(15.0)


def test_the_clock_survives_a_further_increase() -> None:
    """Passer de 5 à 8 personnes ne remet pas à zéro le chronomètre du seuil 5."""
    occupation = ZoneOccupancy()

    occupation.update(_foule(5), video_time=10.0)
    occupation.update(_foule(8), video_time=20.0)

    assert occupation.duration_at_least("Hall", "person", 5, 30.0) == pytest.approx(20.0)
    assert occupation.duration_at_least("Hall", "person", 8, 30.0) == pytest.approx(10.0)


def test_dropping_below_the_threshold_resets_its_clock() -> None:
    """La foule s'est dispersée : le compteur repart du nouveau franchissement."""
    occupation = ZoneOccupancy()

    occupation.update(_foule(5), video_time=10.0)
    occupation.update(_foule(2), video_time=20.0)
    occupation.update(_foule(5), video_time=30.0)

    assert occupation.duration_at_least("Hall", "person", 5, 40.0) == pytest.approx(10.0)


def test_an_unreached_threshold_has_no_duration() -> None:
    """Zéro seconde, pas une durée négative ni une exception."""
    occupation = ZoneOccupancy()
    occupation.update(_foule(2), video_time=10.0)

    assert occupation.duration_at_least("Hall", "person", 5, 20.0) == 0.0


def test_the_clock_follows_video_time_not_frame_count() -> None:
    """Deux frames espacées de 30 s valent 30 s, quelle que soit la cadence.

    Mémoriser l'instant du franchissement plutôt qu'un compteur de frames rend
    la mesure indépendante de la vitesse d'analyse — le principe des deux
    horloges, appliqué au comptage.
    """
    occupation = ZoneOccupancy()

    occupation.update(_foule(4), video_time=100.0)
    occupation.update(_foule(4), video_time=130.0)

    assert occupation.duration_at_least("Hall", "person", 4, 130.0) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Variations
# ---------------------------------------------------------------------------


def test_a_stable_scene_produces_no_change() -> None:
    """Cent frames identiques ne doivent produire aucune ligne."""
    occupation = ZoneOccupancy()
    occupation.update(_foule(4), video_time=0.0)

    for index in range(100):
        assert occupation.update(_foule(4), video_time=1.0 + index) == []


def test_a_change_says_where_it_comes_from_and_where_it_goes() -> None:
    """« hall : 2 → 7 personnes » : les deux nombres comptent."""
    occupation = ZoneOccupancy()
    occupation.update(_foule(2), video_time=0.0)

    variations = occupation.update(_foule(7), video_time=5.0)

    assert len(variations) == 1
    variation = variations[0]
    assert (variation.before, variation.after) == (2, 7)
    assert variation.is_increase
    assert "Hall : 2 → 7 person" in variation.label


def test_changes_are_ordered_reproducibly() -> None:
    """Deux analyses de la même vidéo doivent lister les zones dans le même ordre.

    Sans tri, l'ordre d'itération d'un dictionnaire déciderait, et la
    chronologie cesserait d'être reproductible.
    """
    occupation = ZoneOccupancy()

    variations = occupation.update(
        [_obj(1, zones={"Zulu"}), _obj(2, zones={"Alpha"}), _obj(3, zones={"Mike"})],
        video_time=1.0,
    )

    assert [v.zone_name for v in variations] == ["Alpha", "Mike", "Zulu"]


def test_an_emptying_zone_is_reported_too() -> None:
    """Une zone qui se vide est une information, pas une absence d'information."""
    occupation = ZoneOccupancy()
    occupation.update(_foule(5), video_time=0.0)

    variations = occupation.update([], video_time=5.0)

    assert variations[0].after == 0
    assert not variations[0].is_increase


def test_small_movements_are_filtered_out_of_the_report() -> None:
    """Une zone qui passe de 0 à 1 n'est pas un fait marquant de rapport."""
    occupation = ZoneOccupancy()

    occupation.update(_foule(1), video_time=0.0)
    occupation.update(_foule(2), video_time=1.0)

    assert occupation.changes(notable_only=True) == []
    assert len(occupation.changes(notable_only=False)) == 2


def test_a_crowd_forming_is_a_notable_fact() -> None:
    """Passer de 2 à 7 personnes mérite d'être raconté."""
    occupation = ZoneOccupancy()
    occupation.update(_foule(2), video_time=0.0)
    occupation.update(_foule(7), video_time=5.0)

    marquants = occupation.changes(notable_only=True)

    assert len(marquants) == 1
    assert marquants[0].after == 7


def test_reset_clears_everything_between_sessions() -> None:
    """Deux vidéos successives ne doivent pas mélanger leurs comptages."""
    occupation = ZoneOccupancy()
    occupation.update(_foule(5), video_time=10.0)

    occupation.reset()

    assert occupation.count("Hall") == 0
    assert occupation.changes(notable_only=False) == []
    assert occupation.occupied_zones == []


def test_the_summary_reads_in_one_line() -> None:
    """Résumé destiné à la ligne d'état de l'interface."""
    occupation = ZoneOccupancy()
    occupation.update([*_foule(3), _obj(9, class_name="backpack", zones={"Quai"})], 1.0)

    resume = summarise(occupation, ["Hall", "Quai", "Cave"])

    assert "Hall : 3 person" in resume
    assert "Quai : 1 backpack" in resume
    assert "Cave" not in resume, "Une zone vide n'a rien à dire."


def test_the_summary_degrades_to_a_dash() -> None:
    """Une scène vide ne doit pas produire une ligne d'état vide."""
    assert summarise(ZoneOccupancy(), ["Hall"]) == "—"


# ---------------------------------------------------------------------------
# La règle de surdensité
# ---------------------------------------------------------------------------


def _engine(min_occupancy: int = 5, min_duration_s: float = 10.0, **zone_kwargs) -> EventEngine:
    """Moteur n'appliquant que la règle de surdensité."""
    regle = config.EventRule(
        event_type=config.EventType.OVERCROWDING,
        classes=("person",),
        min_duration_s=min_duration_s,
        cooldown_s=180.0,
        min_occupancy=min_occupancy,
    )
    zones = FakeZones([zone("Hall", config.ZoneType.TRANSIT, **zone_kwargs)])
    return EventEngine(zones, rules=(regle,))


def _leve(engine: EventEngine, objets, video_time: float):
    """Incidents levés sur une frame."""
    return engine.evaluate(objets, _FakeTracker(), None, video_time, OUVERT)


def test_a_crowd_below_the_threshold_raises_nothing() -> None:
    """Quatre personnes ne font pas un attroupement quand le seuil est à cinq."""
    engine = _engine(min_occupancy=5)

    _leve(engine, _foule(4), 0.0)

    assert _leve(engine, _foule(4), 60.0) == []


def test_a_crowd_that_disperses_quickly_raises_nothing() -> None:
    """Sept personnes qui se croisent devant une porte ne sont pas un attroupement.

    C'est le rôle de `min_duration_s` : la même durée minimale qui distingue une
    intrusion d'un simple passage distingue ici une foule d'un croisement.
    """
    engine = _engine(min_occupancy=5, min_duration_s=10.0)

    _leve(engine, _foule(7), 0.0)

    assert _leve(engine, _foule(7), 5.0) == [], "5 s < 10 s : trop court."


def test_a_lasting_crowd_raises_the_incident() -> None:
    """Le cas nominal : assez de monde, assez longtemps."""
    engine = _engine(min_occupancy=5, min_duration_s=10.0)

    _leve(engine, _foule(7), 0.0)
    leves = _leve(engine, _foule(7), 15.0)

    assert leves, "L'attroupement aurait dû être signalé."
    assert leves[0].event_type is config.EventType.OVERCROWDING
    assert leves[0].zone_name == "Hall"


def test_the_incident_says_how_many_people_were_there() -> None:
    """Un rapport « surdensité » sans le nombre d'occupants n'apprend rien."""
    engine = _engine(min_occupancy=5, min_duration_s=10.0)
    _leve(engine, _foule(7), 0.0)

    incident = _leve(engine, _foule(7), 15.0)[0]

    assert incident.details["occupants"] == 7
    assert incident.details["seuil_occupation"] == 5


def test_a_zone_can_set_its_own_crowd_threshold() -> None:
    """Six personnes dans un hall de gare est normal ; dans un local technique, non."""
    strict = EventEngine(
        FakeZones([zone("Hall", config.ZoneType.TRANSIT, min_occupancy=3)]),
        rules=(
            config.EventRule(
                event_type=config.EventType.OVERCROWDING,
                classes=("person",),
                min_duration_s=10.0,
                cooldown_s=180.0,
                min_occupancy=20,
            ),
        ),
    )

    _leve(strict, _foule(4), 0.0)
    leves = _leve(strict, _foule(4), 15.0)

    assert leves, "Le seuil local de 3 doit l'emporter sur le seuil global de 20."


def test_only_the_targeted_class_is_counted() -> None:
    """Une règle « person » ne compte pas les sacs."""
    engine = _engine(min_occupancy=3, min_duration_s=0.0)
    melange = [*_foule(2), *_foule(5, depart=50, classe="backpack")]

    _leve(engine, melange, 0.0)

    assert _leve(engine, melange, 15.0) == []


def test_counting_ignores_the_cooldown_that_designating_respects() -> None:
    """La dissymétrie assumée de la règle collective.

    Sept personnes forment un attroupement même si six d'entre elles viennent
    d'être signalées : c'est le fait collectif qui est constaté, l'objet désigné
    n'en est que le porteur pour le rapport.
    """
    engine = _engine(min_occupancy=5, min_duration_s=0.0)
    foule = _foule(7)
    # Six des sept personnes viennent d'être signalées : elles sont sous délai
    # de garde et ne peuvent pas porter un nouvel incident.
    for obj in foule[:6]:
        obj.mark_event(config.EventType.OVERCROWDING.value, 0.0)

    leves = _leve(engine, foule, 1.0)

    assert len(leves) == 1, "Un seul porteur reste éligible."
    assert leves[0].details["occupants"] == 7, (
        "Le comptage doit voir les sept, pas seulement le porteur éligible."
    )
    assert leves[0].track_id == 7


def test_a_counting_zone_never_raises_overcrowding() -> None:
    """Compter les entrées d'un magasin ne doit pas produire une alerte de foule."""
    regle = config.EventRule(
        event_type=config.EventType.OVERCROWDING,
        classes=("person",),
        min_duration_s=0.0,
        cooldown_s=180.0,
        min_occupancy=2,
    )
    engine = EventEngine(
        FakeZones([zone("Porte", config.ZoneType.COUNTING)]), rules=(regle,)
    )
    foule = _foule(10, zone_name="Porte")

    _leve(engine, foule, 0.0)

    assert _leve(engine, foule, 30.0) == []


def test_a_rule_without_occupancy_threshold_is_inert() -> None:
    """Un champ `min_occupancy` laissé à `None` désactive proprement la règle."""
    regle = config.EventRule(
        event_type=config.EventType.OVERCROWDING, classes=("person",), min_duration_s=0.0
    )
    engine = EventEngine(FakeZones.restricted("Hall"), rules=(regle,))
    foule = _foule(50)

    assert _leve(engine, foule, 10.0) == []


def test_the_engine_owns_a_counter_without_implementing_it() -> None:
    """Le moteur *utilise* un compteur ; il n'en porte pas la logique.

    C'est ce qui garde « une classe = une responsabilité » : `ZoneOccupancy`
    compte, `EventEngine` applique des règles.
    """
    engine = _engine()

    assert isinstance(engine.occupancy, ZoneOccupancy)


def test_the_engine_updates_the_count_before_evaluating() -> None:
    """Une règle collective a besoin de l'état de la frame courante.

    Compter après aurait fait raisonner la règle sur la frame précédente — un
    décalage d'une frame invisible en test unitaire et redoutable en production.
    """
    engine = _engine()

    _leve(engine, _foule(6), 5.0)

    assert engine.occupancy.count("Hall", "person") == 6


# ---------------------------------------------------------------------------
# La chronologie raconte les variations
# ---------------------------------------------------------------------------


def test_the_timeline_records_a_crowd_forming() -> None:
    """« hall : 2 → 7 personnes » décrit la scène, pas un individu."""
    timeline = Timeline()
    variation = OccupancyChange("Hall", "person", 2, 7, 12.0)

    timeline.observe(12.0, [], [], None, occupancy_changes=[variation])

    libelles = [fait.label for tranche in timeline.slices() for fait in tranche.facts]
    assert any("Hall : 2 → 7 person" in libelle for libelle in libelles)


def test_occupancy_facts_rank_after_incidents_but_before_movement() -> None:
    """Un lecteur cherche les incidents, puis la scène, puis les allées et venues."""
    timeline = Timeline()

    timeline.observe(
        1.0,
        [],
        [],
        None,
        occupancy_changes=[OccupancyChange("Hall", "person", 2, 7, 1.0)],
    )

    natures = [fait.kind for tranche in timeline.slices() for fait in tranche.facts]
    assert natures == ["occupation"]
    assert config.TIMELINE.slice_seconds > 0


def test_an_object_without_a_label_is_ignored_by_the_timeline() -> None:
    """La chronologie ne fait pas confiance à ce qu'on lui passe.

    Elle lit `label` s'il existe et ignore le reste, plutôt que de lever au
    milieu d'une analyse d'une heure.
    """
    timeline = Timeline()

    timeline.observe(1.0, [], [], None, occupancy_changes=[object()])

    assert len(timeline) == 0
