"""Une zone déclare ce qu'elle surveille, et avec quels seuils.

Ce que `ZoneConfig` ne savait pas dire
--------------------------------------
Un polygone plus un booléen `restricted` ne décrit pas un périmètre réel. Un
quai de chargement, un hall d'accueil et une porte de magasin n'appellent pas
les mêmes règles : rôder dans un hall est banal, rôder dans une réserve ne l'est
pas, et compter les passages à une porte ne doit lever **aucun** incident.

Un seul mécanisme de ciblage
-----------------------------
`EventRule.zones` a été supprimé. Deux mécanismes concurrents — une règle qui
déclare ses zones et une zone qui déclare ses règles — auraient exigé une règle
de résolution de conflit, donc une explication de plus dans le README et une
source d'erreur de plus dans la configuration. C'est la zone qui gagne : elle
est ce qu'on édite quand on installe le logiciel sur un site.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import config
from sentinel.detection import Detection
from sentinel.events import EventEngine, FrameContext
from sentinel.exceptions import ZoneConfigurationError
from sentinel.tracker import TrackedObject
from sentinel.zones import ZoneManager
from zone_doubles import FakeZones, zone

OUVERT = datetime(2026, 8, 13, 10, 0, 0)  # jeudi 10 h : site ouvert
NUIT = datetime(2026, 8, 13, 23, 30, 0)  # jeudi 23 h 30 : site fermé


class _FakeTracker:
    """Tracker réduit à `nearest()`, seul service utilisé par les règles."""

    def nearest(self, target, *, class_names, radius_px):
        return None


def _obj(
    track_id: int = 1,
    *,
    class_name: str = "person",
    zones: dict[str, float] | None = None,
    age: float = 100.0,
    positions: list[tuple[float, float]] | None = None,
) -> TrackedObject:
    """Objet suivi prêt à l'emploi, éventuellement immobile."""
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
    for index, point in enumerate(positions or []):
        obj.history.append((float(index), point))
    return obj


def _leve(engine: EventEngine, objets, *, video_time: float, moment=OUVERT):
    """Types d'incidents levés sur une frame."""
    return [e.event_type for e in engine.evaluate(objets, _FakeTracker(), None, video_time, moment)]


# ---------------------------------------------------------------------------
# Le type conditionne les règles applicables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("genre", list(config.ZoneType))
def test_every_zone_type_declares_what_it_accepts(genre: config.ZoneType) -> None:
    """Ajouter un type sans dire ce qu'il déclenche doit échouer tout de suite."""
    assert isinstance(genre.event_types, tuple)
    assert genre.description


def test_a_forbidden_zone_accepts_every_rule() -> None:
    """C'est le type le plus strict : rien n'y est écarté.

    C'est aussi celui de la zone plein cadre livrée par défaut — « surveiller
    tout ce qui s'affiche » se traduit exactement par là.
    """
    interdite = config.ZoneType.FORBIDDEN

    for genre in config.EventType:
        assert genre in interdite.event_types


def test_a_transit_zone_refuses_intrusion_but_accepts_loitering() -> None:
    """On y passe, on n'y reste pas.

    Traverser un hall est normal — donc aucune intrusion. Y stationner une
    minute ne l'est pas — donc le rôdage s'applique.
    """
    transit = config.ZoneType.TRANSIT

    assert config.EventType.INTRUSION not in transit.event_types
    assert config.EventType.LOITERING in transit.event_types


def test_a_scheduled_zone_refuses_intrusion_but_watches_the_night() -> None:
    """Présence normale aux heures d'ouverture, anormale en dehors."""
    horaires = config.ZoneType.SCHEDULED

    assert config.EventType.INTRUSION not in horaires.event_types
    assert config.EventType.AFTER_HOURS in horaires.event_types


def test_a_sensitive_zone_centres_on_the_abandoned_object() -> None:
    """Quai, salle d'attente, hall de gare : le risque est l'objet déposé.

    La présence de personnes y est normale, y compris prolongée — le rôdage
    n'est donc pas déclaré.
    """
    sensible = config.ZoneType.SENSITIVE

    assert config.EventType.ABANDONED_OBJECT in sensible.event_types
    assert config.EventType.LOITERING not in sensible.event_types


def test_a_counting_zone_raises_nothing_at_all() -> None:
    """Compter les entrées d'un magasin ne doit pas produire un rapport par client."""
    assert config.ZoneType.COUNTING.event_types == ()


def test_only_a_forbidden_zone_is_restricted() -> None:
    """`restricted` est **dérivé** du type, pas un second drapeau à contredire."""
    for genre in config.ZoneType:
        attendu = genre is config.ZoneType.FORBIDDEN
        assert zone("Z", genre).restricted is attendu


# ---------------------------------------------------------------------------
# Le type gouverne réellement le moteur
# ---------------------------------------------------------------------------


def test_no_incident_is_raised_in_a_counting_zone() -> None:
    """Le cas qui compte : une zone de comptage reste silencieuse.

    L'objet est là depuis 100 s, immobile, à 23 h 30 — trois règles auraient de
    quoi se déclencher. Aucune ne le fait.
    """
    engine = EventEngine(FakeZones.of_types(Porte=config.ZoneType.COUNTING))
    immobile = _obj(class_name="backpack", zones={"Porte": 0.0}, positions=[(120.0, 200.0)] * 60)

    assert _leve(engine, [immobile], video_time=100.0, moment=NUIT) == []


def test_a_transit_zone_raises_loitering_but_never_intrusion() -> None:
    """Le type filtre à l'exécution, pas seulement sur le papier."""
    engine = EventEngine(FakeZones.of_types(Hall=config.ZoneType.TRANSIT))
    obj = _obj(zones={"Hall": 0.0}, age=100.0)

    types = _leve(engine, [obj], video_time=100.0)

    assert config.EventType.LOITERING in types
    assert config.EventType.INTRUSION not in types


def test_a_forbidden_zone_raises_the_intrusion_a_transit_zone_refused() -> None:
    """Même objet, même durée : seul le type de zone change le résultat."""
    obj = _obj(zones={"Hall": 0.0}, age=100.0)

    interdite = EventEngine(FakeZones.of_types(Hall=config.ZoneType.FORBIDDEN))

    assert config.EventType.INTRUSION in _leve(interdite, [obj], video_time=100.0)


def test_an_object_outside_every_zone_is_still_watched() -> None:
    """Aucune zone occupée n'est une absence d'avis, pas un refus.

    Un objet abandonné hors de tout périmètre nommé reste un objet abandonné.
    Le traiter comme « refusé » ferait disparaître tout incident dès qu'un
    polygone ne couvre pas la totalité de l'image.
    """
    engine = EventEngine(FakeZones.of_types(Quai=config.ZoneType.FORBIDDEN))
    orphelin = _obj(class_name="backpack", zones=None, positions=[(120.0, 200.0)] * 60)

    types = _leve(engine, [orphelin], video_time=100.0)

    assert config.EventType.ABANDONED_OBJECT in types


# ---------------------------------------------------------------------------
# Surcharges de seuils par zone
# ---------------------------------------------------------------------------


def test_a_zone_without_override_returns_the_global_rule_unchanged() -> None:
    """L'identité rend visible l'absence de surcharge, et évite une copie."""
    regle = config.EVENT_RULES[0]

    assert zone("Quai").rule_for(regle) is regle


def test_a_zone_can_be_stricter_than_the_global_configuration() -> None:
    """Une réserve peut exiger 1 s là où le reste du site en demande 3."""
    stricte = zone("Réserve", config.ZoneType.FORBIDDEN, min_duration_s=1.0)
    regle = config.EventRule(
        event_type=config.EventType.INTRUSION, min_duration_s=3.0, cooldown_s=60.0
    )

    assert stricte.rule_for(regle).min_duration_s == 1.0
    assert stricte.rule_for(regle).cooldown_s == 60.0, "Le reste de la règle est intact."


def test_overrides_actually_change_when_an_incident_fires() -> None:
    """Le seuil de la zone décide, pas celui de la configuration globale."""
    regle = config.EventRule(
        event_type=config.EventType.INTRUSION, classes=("person",),
        min_duration_s=30.0, cooldown_s=60.0,
    )
    zones = FakeZones([zone("Réserve", config.ZoneType.FORBIDDEN, min_duration_s=2.0)])
    engine = EventEngine(zones, rules=(regle,))

    obj = _obj(zones={"Réserve": 0.0})

    # 5 s de séjour : sous le seuil global de 30 s, au-dessus du seuil local de 2 s.
    assert _leve(engine, [obj], video_time=5.0) == [config.EventType.INTRUSION]


def test_a_shorter_local_cooldown_is_not_blocked_by_the_global_one() -> None:
    """Le pré-filtre retient le délai le plus court, sinon il écarterait à tort.

    Une zone qui autorise un redéclenchement toutes les 10 s ne doit pas être
    bâillonnée par un délai global de 300 s.
    """
    regle = config.EventRule(
        event_type=config.EventType.INTRUSION, classes=("person",),
        min_duration_s=1.0, cooldown_s=300.0,
    )
    zones = FakeZones([zone("Quai", config.ZoneType.FORBIDDEN, cooldown_s=10.0)])
    engine = EventEngine(zones, rules=(regle,))
    obj = _obj(zones={"Quai": 0.0})

    premier = _leve(engine, [obj], video_time=5.0)
    trop_tot = _leve(engine, [obj], video_time=10.0)
    assez_tard = _leve(engine, [obj], video_time=20.0)

    assert premier == [config.EventType.INTRUSION]
    assert trop_tot == [], "Le délai local de 10 s doit encore courir."
    assert assez_tard == [config.EventType.INTRUSION], "Le délai global a bâillonné la zone."


def test_an_override_that_breaks_the_anti_bounce_is_refused() -> None:
    """Une zone qui exige 120 s mais redéclenche toutes les 60 s crie sans arrêt."""
    incoherente = zone(
        "Quai", config.ZoneType.FORBIDDEN, min_duration_s=120.0, cooldown_s=60.0
    )
    regle = config.EventRule(event_type=config.EventType.INTRUSION)

    with pytest.raises(ValueError, match="cooldown_s"):
        incoherente.rule_for(regle)


def test_an_incoherent_zone_is_refused_at_startup_not_mid_analysis() -> None:
    """Une configuration fautive doit échouer au démarrage.

    La découvrir au milieu de l'analyse d'une vidéo d'une heure coûterait
    l'analyse entière.
    """
    incoherente = config.SurveillanceZone(
        name="Quai",
        polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        zone_type=config.ZoneType.FORBIDDEN,
        min_duration_s=10_000.0,
    )

    with pytest.raises(ZoneConfigurationError, match="cooldown_s"):
        ZoneManager([incoherente])


# ---------------------------------------------------------------------------
# Le gestionnaire répond sur les règles
# ---------------------------------------------------------------------------


def test_the_manager_lists_the_zones_that_accept_a_rule() -> None:
    """C'est le seul mécanisme de ciblage : la règle ne nomme plus de zone."""
    manager = ZoneManager(
        [
            zone("Quai", config.ZoneType.FORBIDDEN),
            zone("Hall", config.ZoneType.TRANSIT),
            zone("Porte", config.ZoneType.COUNTING),
        ]
    )

    assert manager.zones_handling(config.EventType.INTRUSION) == ["Quai"]
    assert manager.zones_handling(config.EventType.LOITERING) == ["Quai", "Hall"]


def test_the_rule_no_longer_declares_its_zones() -> None:
    """`EventRule.zones` a disparu : un seul mécanisme, aucun conflit à arbitrer."""
    assert not hasattr(config.EVENT_RULES[0], "zones")


def test_an_unknown_zone_falls_back_to_the_global_rule() -> None:
    """Un nom de zone inconnu ne doit pas faire échouer une analyse en cours."""
    manager = ZoneManager([zone("Quai")])
    regle = config.EVENT_RULES[0]

    assert manager.rule_for("Inexistante", regle) is regle


def test_the_shortest_cooldown_spans_every_zone() -> None:
    """Le pré-filtre du moteur doit être le plus permissif des délais."""
    # Les délais restent au-dessus du plus long `min_duration_s` du jeu de règles
    # livré (60 s pour le rôdage) : `ZoneManager` valide les surcharges contre
    # **toutes** les règles que la zone accepte, et refuserait un délai plus court.
    manager = ZoneManager(
        [
            zone("Lent", config.ZoneType.FORBIDDEN, cooldown_s=300.0),
            zone("Rapide", config.ZoneType.FORBIDDEN, cooldown_s=65.0),
        ]
    )
    regle = config.EventRule(
        event_type=config.EventType.INTRUSION, min_duration_s=1.0, cooldown_s=120.0
    )

    assert manager.shortest_cooldown(regle) == 65.0


def test_the_default_zone_still_watches_everything() -> None:
    """Zéro régression : la configuration livrée déclenche les quatre règles."""
    manager = ZoneManager()

    for genre in config.EventType:
        assert manager.zones_handling(genre) == ["Champ de la caméra"], genre.value
