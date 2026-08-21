"""Franchissement de ligne : compter des passages, et dans quel sens.

L'algorithme, en une phrase
----------------------------
Le côté d'un point par rapport à une droite est donné par le **signe du produit
vectoriel**. Un objet a franchi la ligne entre deux frames si ce signe a changé
— et le nouveau signe donne le sens.

Le piège que ces tests verrouillent
-------------------------------------
Un changement de signe ne suffit pas : le produit vectoriel décrit une **droite
infinie**, pas un segment. Sans un second test d'intersection, un objet passant
dix mètres au-delà de l'extrémité de la ligne serait compté comme l'ayant
franchie. C'est l'erreur classique de cette implémentation, et elle est
silencieuse : les chiffres restent plausibles, ils sont seulement faux.

Aucune librairie
-----------------
`supervision.LineZone` rendrait le même service au prix de ~30 Mo, refusés
ailleurs dans ce projet pour la même raison. Quatre soustractions et deux
multiplications ne justifient pas une dépendance.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import config
from sentinel.crossing import LineCounter, LineCrossing, segments_intersect, side_of
from sentinel.detection import Detection
from sentinel.timeline import Timeline
from sentinel.tracker import TrackedObject

T0 = datetime(2026, 8, 21, 10, 0, 0)

# Ligne horizontale à mi-hauteur, du quart gauche au quart droit de l'image.
PORTE = config.CrossingLine(
    name="Porte",
    start=(0.25, 0.50),
    end=(0.75, 0.50),
    positive_label="entrée",
    negative_label="sortie",
    classes=("person",),
)

FRAME = (1000, 1000, 3)  # hauteur, largeur, canaux


def _obj(track_id: int, position: tuple[float, float], *, class_name: str = "person"):
    """Objet suivi dont le point d'appui tombe exactement sur `position`."""
    x, y = position
    detection = Detection(0, class_name, 0.9, (x - 20.0, y - 100.0, x + 20.0, y), track_id)
    return TrackedObject(track_id, class_name, 0.0, 0.0, T0, T0, detection)


def _counter(*lines: config.CrossingLine) -> LineCounter:
    """Compteur déjà initialisé sur une image carrée de 1000 px."""
    compteur = LineCounter(list(lines) or [PORTE])
    compteur.initialize(FRAME)
    return compteur


def _parcours(compteur: LineCounter, positions, *, track_id: int = 1, classe: str = "person"):
    """Fait suivre un chemin à un objet et rend tous les franchissements."""
    constats: list[LineCrossing] = []
    for index, position in enumerate(positions):
        constats += compteur.update([_obj(track_id, position, class_name=classe)], float(index))
    return constats


# ---------------------------------------------------------------------------
# Les primitives géométriques
# ---------------------------------------------------------------------------


def test_the_cross_product_separates_the_two_sides() -> None:
    """Positif d'un côté, négatif de l'autre, nul sur la droite."""
    debut, fin = (0.0, 0.0), (10.0, 0.0)

    assert side_of(debut, fin, (5.0, 5.0)) > 0
    assert side_of(debut, fin, (5.0, -5.0)) < 0
    assert side_of(debut, fin, (5.0, 0.0)) == 0


def test_reversing_the_line_reverses_the_sides() -> None:
    """Inverser `start` et `end` inverse les deux libellés : c'est documenté."""
    point = (5.0, 5.0)

    assert side_of((0.0, 0.0), (10.0, 0.0), point) > 0
    assert side_of((10.0, 0.0), (0.0, 0.0), point) < 0


def test_crossing_segments_are_detected() -> None:
    """Deux segments qui se croisent franchement."""
    assert segments_intersect((5.0, -5.0), (5.0, 5.0), (0.0, 0.0), (10.0, 0.0))


def test_segments_that_only_cross_the_infinite_line_are_rejected() -> None:
    """LE test qui compte : passer au-delà de l'extrémité n'est pas franchir.

    L'objet traverse bien la *droite* portée par la ligne, mais à x = 50 alors
    que le segment s'arrête à x = 10. Sans ce rejet, tout objet croisant le
    prolongement imaginaire de la ligne serait compté.
    """
    assert not segments_intersect((50.0, -5.0), (50.0, 5.0), (0.0, 0.0), (10.0, 0.0))


def test_parallel_segments_never_cross() -> None:
    """Longer une ligne n'est pas la franchir."""
    assert not segments_intersect((0.0, 5.0), (10.0, 5.0), (0.0, 0.0), (10.0, 0.0))


def test_touching_an_endpoint_is_not_a_crossing() -> None:
    """Un frôlement exactement sur la ligne ferait osciller le total.

    Le rejeter est le choix conservateur : un objet qui s'arrête pile sur le
    seuil n'a rien franchi tant qu'il n'est pas passé de l'autre côté.
    """
    assert not segments_intersect((5.0, -5.0), (5.0, 0.0), (0.0, 0.0), (10.0, 0.0))


# ---------------------------------------------------------------------------
# Comptage
# ---------------------------------------------------------------------------


def test_a_straight_crossing_is_counted_once() -> None:
    """Le cas nominal : un aller simple, un passage."""
    compteur = _counter()

    constats = _parcours(compteur, [(500.0, 600.0), (500.0, 400.0)])

    assert len(constats) == 1
    assert constats[0].line_name == "Porte"
    assert constats[0].track_id == 1


def test_the_direction_follows_the_side_it_ends_on() -> None:
    """Aller et retour portent des libellés opposés."""
    monte = _parcours(_counter(), [(500.0, 600.0), (500.0, 400.0)])
    descend = _parcours(_counter(), [(500.0, 400.0), (500.0, 600.0)])

    assert monte[0].direction != descend[0].direction
    assert {monte[0].direction, descend[0].direction} == {"entrée", "sortie"}


def test_the_positive_direction_goes_down_the_image() -> None:
    """La convention, énoncée par un test plutôt que par un commentaire.

    Pour une ligne tracée de gauche à droite, le sens positif va vers le **bas**
    de l'image : les coordonnées image ont leur origine en haut à gauche et leur
    axe y dirigé vers le bas. Quand la convention ne tombe pas dans le bon sens
    sur la scène filmée, on inverse `start` et `end` — plus lisible que
    d'échanger les libellés.
    """
    descend = _parcours(_counter(), [(500.0, 400.0), (500.0, 600.0)])

    assert descend[0].direction == PORTE.positive_label


def test_swapping_the_endpoints_swaps_the_directions() -> None:
    """Le réglage prévu quand la convention ne tombe pas dans le bon sens."""
    inversee = config.CrossingLine(
        name="Porte", start=(0.75, 0.50), end=(0.25, 0.50), classes=("person",)
    )

    droite = _parcours(_counter(PORTE), [(500.0, 400.0), (500.0, 600.0)])
    gauche = _parcours(_counter(inversee), [(500.0, 400.0), (500.0, 600.0)])

    assert droite[0].direction != gauche[0].direction


def test_the_labels_come_from_the_configuration() -> None:
    """« Montée » et « descente » conviennent à un escalier.

    Un rapport qui parle la langue du site se relit sans traduction.
    """
    escalier = config.CrossingLine(
        name="Escalier",
        start=(0.25, 0.50),
        end=(0.75, 0.50),
        positive_label="montée",
        negative_label="descente",
    )

    constats = _parcours(_counter(escalier), [(500.0, 600.0), (500.0, 400.0)])

    assert constats[0].direction == "descente"
    assert "montée" not in constats[0].direction


def test_staying_on_one_side_counts_nothing() -> None:
    """Se promener d'un côté ne doit produire aucune ligne."""
    compteur = _counter()

    constats = _parcours(compteur, [(300.0, 600.0), (700.0, 700.0), (500.0, 900.0)])

    assert constats == []
    assert len(compteur) == 0


def test_passing_beyond_the_line_ends_counts_nothing() -> None:
    """Le même piège, vu du compteur.

    L'objet traverse la hauteur de la ligne, mais à x = 900 alors que la ligne
    s'arrête à x = 750. C'est un passage à côté, pas au travers.
    """
    compteur = _counter()

    constats = _parcours(compteur, [(900.0, 600.0), (900.0, 400.0)])

    assert constats == []


def test_a_round_trip_counts_twice_in_opposite_directions() -> None:
    """Entrer puis sortir laisse un solde nul."""
    compteur = _counter()

    _parcours(compteur, [(500.0, 600.0), (500.0, 400.0), (500.0, 600.0)])

    assert compteur.count("Porte") == 2
    assert compteur.net("Porte") == 0


def test_the_net_balance_is_the_useful_measure() -> None:
    """Le solde répond à « combien de personnes sont à l'intérieur ? »."""
    compteur = _counter()

    # Trois personnes descendent (sens positif), une remonte.
    for track_id in (1, 2, 3):
        _parcours(compteur, [(500.0, 400.0), (500.0, 600.0)], track_id=track_id)
    _parcours(compteur, [(500.0, 600.0), (500.0, 400.0)], track_id=4)

    assert compteur.count("Porte", "entrée") == 3
    assert compteur.count("Porte", "sortie") == 1
    assert compteur.net("Porte") == 2


def test_only_the_configured_classes_are_counted() -> None:
    """Une porte qui compte les personnes ne compte pas les voitures."""
    compteur = _counter()

    constats = _parcours(compteur, [(500.0, 600.0), (500.0, 400.0)], classe="car")

    assert constats == []


def test_an_empty_class_filter_counts_everything() -> None:
    """Un tuple vide signifie « toutes les classes », pas « aucune »."""
    ouverte = config.CrossingLine(name="Portail", start=(0.25, 0.5), end=(0.75, 0.5))

    constats = _parcours(_counter(ouverte), [(500.0, 600.0), (500.0, 400.0)], classe="car")

    assert len(constats) == 1


def test_the_summary_reads_in_one_line() -> None:
    """Destiné à la ligne d'état et au rapport."""
    compteur = _counter()
    _parcours(compteur, [(500.0, 400.0), (500.0, 600.0)])

    resume = compteur.summary("Porte")

    assert "1 entrée" in resume
    assert "0 sortie" in resume
    assert "+1" in resume


def test_an_unknown_line_degrades_instead_of_raising() -> None:
    """Interroger une ligne inexistante ne doit pas interrompre une analyse."""
    compteur = _counter()

    assert compteur.net("Inconnue") == 0
    assert "inconnue" in compteur.summary("Inconnue")


# ---------------------------------------------------------------------------
# Cycle de vie et pièges de suivi
# ---------------------------------------------------------------------------


def test_nothing_happens_before_initialisation() -> None:
    """Les lignes sont normalisées : sans résolution, aucun pixel n'existe."""
    compteur = LineCounter([PORTE])

    assert not compteur.is_initialized
    assert compteur.update([_obj(1, (500.0, 400.0))], 1.0) == []


def test_a_resolution_change_forgets_the_previous_positions() -> None:
    """Les positions mémorisées étaient en pixels de l'ancienne résolution.

    Les garder ferait constater un franchissement imaginaire au premier
    redimensionnement — l'objet paraîtrait avoir bondi d'un côté à l'autre.
    """
    compteur = _counter()
    compteur.update([_obj(1, (500.0, 600.0))], 1.0)

    compteur.initialize((500, 500, 3))
    constats = compteur.update([_obj(1, (250.0, 200.0))], 2.0)

    assert constats == []


def test_a_disappeared_track_releases_its_position() -> None:
    """Un identifiant réattribué ne doit pas hériter d'une position étrangère.

    ByteTrack recycle les identifiants. Sans ce nettoyage, l'objet #1 d'une
    scène hériterait de la dernière position de l'objet #1 de la précédente et
    produirait un franchissement fantôme.
    """
    compteur = _counter()
    compteur.update([_obj(1, (500.0, 900.0))], 1.0)
    compteur.update([], 2.0)  # l'objet #1 disparaît

    constats = compteur.update([_obj(1, (500.0, 100.0))], 3.0)

    assert constats == [], "Un franchissement fantôme a été compté."


def test_an_immobile_object_is_not_re_examined() -> None:
    """Deux positions identiques ne forment pas un segment de déplacement."""
    compteur = _counter()

    constats = _parcours(compteur, [(500.0, 500.0)] * 10)

    assert constats == []


def test_crossings_are_ordered_reproducibly() -> None:
    """Deux analyses de la même vidéo doivent lister les passages dans le même ordre."""
    compteur = _counter()
    compteur.update([_obj(3, (500.0, 600.0)), _obj(1, (500.0, 600.0))], 1.0)

    constats = compteur.update([_obj(3, (500.0, 400.0)), _obj(1, (500.0, 400.0))], 2.0)

    assert [c.track_id for c in constats] == [1, 3]


def test_reset_clears_everything_between_sessions() -> None:
    """Deux vidéos successives ne doivent pas mélanger leurs comptages."""
    compteur = _counter()
    _parcours(compteur, [(500.0, 600.0), (500.0, 400.0)])

    compteur.reset()

    assert len(compteur) == 0
    assert compteur.net("Porte") == 0


def test_no_line_is_configured_by_default() -> None:
    """Le comptage de flux suppose de connaître la scène.

    En déclarer une « au cas où » ferait apparaître des chiffres que personne n'a
    demandés dans tous les rapports.
    """
    assert config.CROSSING_LINES == ()
    assert LineCounter().update([_obj(1, (500.0, 400.0))], 1.0) == []


# ---------------------------------------------------------------------------
# Ce que ça alimente
# ---------------------------------------------------------------------------


def test_a_crossing_becomes_a_timeline_fact() -> None:
    """Un franchissement se raconte : « Porte : person #1 — entrée »."""
    timeline = Timeline()
    passage = LineCrossing("Porte", 1, "person", "entrée", 12.0)

    timeline.observe(12.0, [], [], None, crossings=[passage])

    libelles = [fait.label for tranche in timeline.slices() for fait in tranche.facts]
    assert libelles == ["Porte : person #1 — entrée"]


def test_a_crossing_is_listed_before_the_count_it_explains() -> None:
    """Un franchissement **explique** un changement de comptage : il le précède."""
    from sentinel.occupancy import OccupancyChange

    timeline = Timeline()

    timeline.observe(
        1.0,
        [],
        [],
        None,
        occupancy_changes=[OccupancyChange("Hall", "person", 2, 7, 1.0)],
        crossings=[LineCrossing("Porte", 1, "person", "entrée", 1.0)],
    )

    natures = [fait.kind for tranche in timeline.slices() for fait in tranche.facts]
    assert natures == ["franchissement", "occupation"]


def test_a_counting_zone_can_reference_its_line() -> None:
    """Le lien entre les deux structures est un **nom**, pas une imbrication.

    Une zone de comptage délimite l'espace observé, la ligne y mesure le flux :
    deux objets distincts, reliés par une référence, chacun testable seul.
    """
    zone = config.SurveillanceZone(
        name="Sas d'entrée",
        polygon=((0.2, 0.3), (0.8, 0.3), (0.8, 0.9), (0.2, 0.9)),
        zone_type=config.ZoneType.COUNTING,
        crossing_line="Porte",
    )

    assert zone.crossing_line == "Porte"
    assert zone.zone_type.event_types == (), "Une zone de comptage ne lève rien."
