"""Objet abandonné : « son porteur est parti », pas « personne n'est à côté ».

Ce que le critère de voisinage ne savait pas dire
--------------------------------------------------
La règle demandait « aucune personne dans le rayon **maintenant** ». Le critère
est fragile dans les deux sens :

* **en foule**, il y a toujours quelqu'un dans le rayon — un sac réellement
  abandonné dans un hall de gare n'est jamais signalé ;
* **dans un lieu désert**, le premier passant qui s'éloigne suffit à déclencher,
  alors que le propriétaire est peut-être à trois mètres.

La logique relationnelle pose la bonne question : *cette personne-là*, celle qui
accompagnait l'objet quand il est apparu, est-elle encore dans le champ ? Le
`track_id` stable suffit à la suivre — aucune dépendance supplémentaire.

Le moment où l'on regarde
---------------------------
L'association se fait pendant les premières secondes de vie de l'objet, pas au
moment où la règle se déclenche. Quand un sac est immobile depuis trente
secondes, la personne qui l'a posé est partie depuis longtemps : il faut avoir
regardé au bon moment.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import config
from sentinel.detection import Detection
from sentinel.events import EventEngine
from sentinel.tracker import TrackedObject
from zone_doubles import FakeZones

OUVERT = datetime(2026, 8, 13, 10, 0, 0)  # jeudi 10 h : site ouvert

IMMOBILE = [(120.0, 200.0)] * 60


class _Tracker:
    """Tracker réduit aux deux services que la règle consomme.

    `nearest()` répond « qui est à côté », `get()` répond « cette piste-là
    existe-t-elle encore ». C'est tout le contrat de la règle relationnelle.
    """

    def __init__(self, voisin=None, presents=()) -> None:
        self._voisin = voisin
        self._presents = {obj.track_id: obj for obj in presents}

    def nearest(self, target, *, class_names, radius_px):
        if self._voisin is None or self._voisin.class_name not in class_names:
            return None
        return self._voisin

    def get(self, track_id: int):
        return self._presents.get(track_id)


def _obj(
    track_id: int = 1,
    *,
    class_name: str = "backpack",
    age: float = 40.0,
    positions: list[tuple[float, float]] | None = None,
) -> TrackedObject:
    """Objet suivi, immobile par défaut."""
    detection = Detection(0, class_name, 0.9, (100.0, 100.0, 140.0, 200.0), track_id)
    obj = TrackedObject(track_id, class_name, 0.0, age, OUVERT, OUVERT, detection)
    for index, point in enumerate(positions if positions is not None else IMMOBILE):
        obj.history.append((float(index), point))
    return obj


def _rule(**kwargs) -> config.EventRule:
    """Règle « objet abandonné » prête à l'emploi."""
    defauts = dict(
        event_type=config.EventType.ABANDONED_OBJECT,
        classes=("backpack",),
        min_duration_s=30.0,
        cooldown_s=300.0,
        max_movement_px=25.0,
        requires_no_owner=True,
        owner_radius_px=150.0,
        owner_binding_s=5.0,
    )
    defauts.update(kwargs)
    return config.EventRule(**defauts)


def _engine() -> EventEngine:
    """Moteur n'appliquant que la règle « objet abandonné »."""
    return EventEngine(FakeZones.restricted("Quai"), rules=(_rule(),))


# ---------------------------------------------------------------------------
# L'association au porteur
# ---------------------------------------------------------------------------


def test_an_object_appearing_alone_has_no_owner() -> None:
    """`None` n'est pas une erreur : un sac déjà posé n'a pas de porteur observable."""
    assert _obj().owner_id is None


def test_the_owner_is_the_most_voted_neighbour() -> None:
    """On vote plutôt que de retenir la première réponse.

    À l'apparition d'un sac, la personne la plus proche sur une frame isolée
    peut être un passant. Le vote laisse la répétition trancher.
    """
    sac = _obj()

    for _ in range(5):
        sac.bind_owner(7)
    sac.bind_owner(3)

    assert sac.owner_id == 7


def test_a_tie_is_broken_reproducibly() -> None:
    """Deux analyses de la même vidéo doivent désigner le même porteur.

    `Counter.most_common` n'ordonne pas les égalités : sans départage explicite,
    le rapport ne serait pas reproductible.
    """
    premier, second = _obj(), _obj()

    for objet, ordre in ((premier, (4, 9)), (second, (9, 4))):
        for identifiant in ordre:
            objet.bind_owner(identifiant)

    assert premier.owner_id == second.owner_id == 4


def test_the_binding_window_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Une personne qui passe devant un sac déjà posé n'en devient pas le porteur.

    C'est le verrou qui distingue « le propriétaire » d'« un passant ».
    """
    engine = _engine()
    regle = _rule(owner_binding_s=5.0)
    passant = _obj(9, class_name="person")

    tardif = _obj(age=30.0)  # bien au-delà de la fenêtre
    engine._bind_owner(tardif, regle, _contexte(tardif, voisin=passant))

    assert tardif.owner_id is None


def test_the_binding_happens_inside_the_window() -> None:
    """Dans la fenêtre, le voisin le plus proche est enregistré."""
    engine = _engine()
    regle = _rule(owner_binding_s=5.0)
    porteur = _obj(9, class_name="person")

    jeune = _obj(age=2.0)
    engine._bind_owner(jeune, regle, _contexte(jeune, voisin=porteur))

    assert jeune.owner_id == 9


def test_only_the_configured_classes_can_own() -> None:
    """Un sac n'appartient pas à une voiture."""
    engine = _engine()
    regle = _rule(owner_binding_s=5.0, owner_classes=("person",))
    voiture = _obj(9, class_name="car")

    jeune = _obj(age=1.0)
    engine._bind_owner(jeune, regle, _contexte(jeune, voisin=voiture))

    assert jeune.owner_id is None


def _contexte(obj: TrackedObject, *, voisin=None, presents=()):
    """Contexte de frame minimal, centré sur un objet."""
    from sentinel.events import FrameContext

    return FrameContext(
        objects=(obj,),
        tracker=_Tracker(voisin, presents),
        zones=FakeZones.restricted("Quai"),
        video_time=obj.age,
        wall_time=OUVERT,
        frame=None,
    )


# ---------------------------------------------------------------------------
# Le déclenchement
# ---------------------------------------------------------------------------


def test_the_incident_fires_when_the_owner_has_left() -> None:
    """Le cas nominal de la refonte : ce porteur-là n'est plus dans le champ."""
    engine = _engine()
    sac = _obj()
    sac.bind_owner(9)

    # Une foule est présente, mais le porteur n'y est pas.
    tracker = _Tracker(voisin=_obj(5, class_name="person"), presents=())

    assert engine._check_abandoned_object(sac, _rule(), tracker, 40.0) is True


def test_a_crowd_no_longer_masks_an_abandoned_bag() -> None:
    """Le défaut principal de l'ancien critère.

    Dans un hall de gare, il y a toujours quelqu'un dans le rayon : un sac
    réellement abandonné n'était jamais signalé.
    """
    engine = _engine()
    sac = _obj()
    sac.bind_owner(9)
    foule = _obj(42, class_name="person")

    assert engine._check_abandoned_object(sac, _rule(), _Tracker(foule), 40.0) is True


def test_nothing_fires_while_the_owner_is_still_around() -> None:
    """Un sac au pied de son propriétaire n'est pas abandonné, même à dix mètres.

    C'est l'autre défaut de l'ancien critère : dans un lieu désert, le
    propriétaire qui s'éloigne de trois mètres suffisait à déclencher.
    """
    engine = _engine()
    sac = _obj()
    sac.bind_owner(9)
    porteur = _obj(9, class_name="person")

    tracker = _Tracker(voisin=None, presents=(porteur,))

    assert engine._check_abandoned_object(sac, _rule(), tracker, 40.0) is False


def test_the_report_names_the_departed_owner() -> None:
    """Un rapport « objet abandonné » doit dire de qui il s'agissait.

    C'est ce qui permet à un opérateur de retrouver la personne dans la vidéo —
    sans que le logiciel ne l'identifie ni ne juge la faute.
    """
    engine = _engine()
    sac = _obj()
    sac.bind_owner(9)

    incidents = engine.evaluate([sac], _Tracker(), None, 40.0, OUVERT)

    assert incidents, "L'incident aurait dû être levé."
    assert incidents[0].details["proprietaire_presume"] == "#9 (parti)"


# ---------------------------------------------------------------------------
# Le repli
# ---------------------------------------------------------------------------


def test_an_ownerless_object_falls_back_to_proximity() -> None:
    """Un sac déjà posé au démarrage n'a pas de porteur observable.

    Le critère de voisinage instantané reste alors le seul disponible, et il
    vaut mieux qu'aucun critère du tout.
    """
    engine = _engine()
    sac = _obj()  # aucun vote

    seul = engine._check_abandoned_object(sac, _rule(), _Tracker(None), 40.0)
    accompagne = engine._check_abandoned_object(
        sac, _rule(), _Tracker(_obj(5, class_name="person")), 40.0
    )

    assert seul is True
    assert accompagne is False


def test_the_report_says_when_no_owner_was_ever_seen() -> None:
    """« Aucun observé » et « parti » ne veulent pas dire la même chose."""
    engine = _engine()

    incidents = engine.evaluate([_obj()], _Tracker(), None, 40.0, OUVERT)

    assert incidents[0].details["proprietaire_presume"] == "aucun observé"


# ---------------------------------------------------------------------------
# Les autres conditions restent
# ---------------------------------------------------------------------------


def test_a_moving_object_is_never_abandoned() -> None:
    """Un sac porté se déplace : la condition d'immobilité l'écarte d'abord."""
    engine = _engine()
    porte = _obj(positions=[(120.0 + 10.0 * i, 200.0) for i in range(40)])
    porte.bind_owner(9)

    assert engine._check_abandoned_object(porte, _rule(), _Tracker(), 40.0) is False


def test_a_young_object_is_never_abandoned() -> None:
    """La durée minimale est la première condition, avant toute relation."""
    engine = _engine()
    jeune = _obj(age=5.0)
    jeune.bind_owner(9)

    assert engine._check_abandoned_object(jeune, _rule(), _Tracker(), 5.0) is False


def test_a_rule_without_owner_requirement_ignores_the_relation() -> None:
    """`requires_no_owner=False` désactive proprement toute la logique."""
    engine = _engine()
    sac = _obj()
    porteur = _obj(9, class_name="person")
    sac.bind_owner(9)

    regle = _rule(requires_no_owner=False)
    tracker = _Tracker(voisin=porteur, presents=(porteur,))

    assert engine._check_abandoned_object(sac, regle, tracker, 40.0) is True


def test_binding_is_skipped_when_the_rule_does_not_ask_for_it() -> None:
    """Ne pas voter inutilement : la règle décide, pas le tracker."""
    engine = _engine()
    jeune = _obj(age=1.0)
    porteur = _obj(9, class_name="person")

    engine._bind_owner(jeune, _rule(requires_no_owner=False), _contexte(jeune, voisin=porteur))

    assert jeune.owner_id is None
