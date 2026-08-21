"""Tests de `sentinel.zones` — géométrie pure, sans modèle ni vidéo.

Les zones sont l'un des rares endroits du projet où une erreur est **silencieuse**
et coûteuse : un polygone mal converti ne lève rien, il déplace simplement la
frontière de quelques dizaines de pixels, et les incidents se déclenchent au
mauvais endroit. D'où des tests sur des coordonnées calculées à la main.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from sentinel.detection import Detection
from sentinel.exceptions import ZoneConfigurationError
from sentinel.zones import ZoneManager, _ascii_label

FRAME_SHAPE = (1000, 1000, 3)  # hauteur, largeur, canaux : 1 unité normalisée = 1000 px


# ---------------------------------------------------------------------------
# Fabriques
# ---------------------------------------------------------------------------


def _zone(
    name: str = "Quai",
    polygon: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)),
    *,
    restricted: bool = True,
    **surcharges,
) -> config.SurveillanceZone:
    """Zone rectangulaire couvrant par défaut la moitié gauche de l'image.

    `restricted` n'est plus un champ mais une **propriété dérivée du type** :
    la fabrique traduit donc l'intention du test en type de zone. `TRANSIT` est
    le contraire naturel d'`interdite` — on y passe sans que rien ne se
    déclenche.
    """
    return config.SurveillanceZone(
        name=name,
        polygon=polygon,
        zone_type=config.ZoneType.FORBIDDEN if restricted else config.ZoneType.TRANSIT,
        **surcharges,
    )


def _dedans(manager, detection) -> list[str]:
    """Zones où la détection se trouve, d'après les marges signées.

    `zones_for()` rend désormais `{zone: marge}` pour **toutes** les zones — la
    marge étant la distance signée au bord, rapportée à la hauteur apparente de
    l'objet. Ces tests ne portent pas sur la largeur de la bande d'incertitude
    (voir `test_zone_margin.py`) mais sur la géométrie : on ne garde donc que le
    signe.
    """
    return sorted(nom for nom, marge in manager.zones_for(detection).items() if marge >= 0)


def _at(x: float, y: float, *, height: float = 40.0) -> Detection:
    """Détection dont le point d'appui (bas de boîte) tombe exactement en (x, y)."""
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        xyxy=(x - 10.0, y - height, x + 10.0, y),
    )


def _ready_manager(*zones: config.ZoneConfig) -> ZoneManager:
    """Gestionnaire déjà converti pour une image de 1000x1000."""
    manager = ZoneManager(zones or (_zone(),))
    manager.initialize(FRAME_SHAPE)
    return manager


# ---------------------------------------------------------------------------
# Validation de la configuration
# ---------------------------------------------------------------------------


def test_polygon_with_less_than_three_vertices_is_rejected() -> None:
    """Deux points ne délimitent pas une surface."""
    with pytest.raises(ZoneConfigurationError, match="au moins 3"):
        ZoneManager([_zone(polygon=((0.1, 0.1), (0.5, 0.5)))])


def test_coordinates_outside_unit_interval_are_rejected() -> None:
    """Un sommet en pixels (ex. 450) au lieu de fraction trahit une confusion d'unité."""
    with pytest.raises(ZoneConfigurationError, match="normalisées"):
        ZoneManager([_zone(polygon=((0.1, 0.1), (450.0, 0.2), (0.3, 0.9)))])


def test_duplicate_zone_names_are_rejected() -> None:
    """Les noms servent de clés dans les rapports : ils doivent être uniques."""
    with pytest.raises(ZoneConfigurationError, match="uniques"):
        ZoneManager([_zone(name="Quai"), _zone(name="Quai")])


def test_empty_zone_list_is_rejected() -> None:
    """Surveiller zéro zone est certainement une erreur de configuration."""
    with pytest.raises(ZoneConfigurationError, match="Aucune zone"):
        ZoneManager([])


def test_default_configuration_is_valid() -> None:
    """Les zones livrées dans config.py passent leur propre validation."""
    manager = ZoneManager()
    assert manager.names == [zone.name for zone in config.ZONES]


# ---------------------------------------------------------------------------
# Conversion en pixels
# ---------------------------------------------------------------------------


def test_membership_requires_initialization() -> None:
    """Tester une appartenance sans connaître la résolution est une erreur de programmation."""
    manager = ZoneManager([_zone()])

    assert manager.is_initialized is False
    with pytest.raises(RuntimeError, match="initialize"):
        manager.zones_for(_at(100.0, 100.0))


def test_normalized_polygon_is_scaled_to_pixels() -> None:
    """0,5 sur une image de 1000 px doit donner exactement 500 px."""
    manager = _ready_manager(_zone())

    polygon = manager.polygon("Quai")
    assert polygon.tolist() == [[0, 0], [500, 0], [500, 1000], [0, 1000]]


def test_scaling_uses_width_for_x_and_height_for_y() -> None:
    """Sur une image non carrée, inverser les axes décalerait toutes les zones."""
    manager = ZoneManager([_zone(polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)))])
    manager.initialize((480, 1280, 3))  # hauteur 480, largeur 1280

    assert manager.polygon("Quai").tolist() == [[0, 0], [640, 0], [640, 240], [0, 240]]


def test_initialize_is_idempotent_and_rebuilds_on_resize() -> None:
    """Même résolution = aucun recalcul ; résolution différente = reconstruction."""
    manager = _ready_manager(_zone())
    first = manager.polygon("Quai")

    manager.initialize(FRAME_SHAPE)
    assert manager.polygon("Quai") is first  # même objet : rien n'a été recalculé

    manager.initialize((500, 500, 3))
    assert manager.polygon("Quai").tolist() == [[0, 0], [250, 0], [250, 500], [0, 500]]
    assert manager.frame_size == (500, 500)


@pytest.mark.parametrize("shape", [(0, 640, 3), (480, 0, 3), (480,)])
def test_invalid_frame_shape_is_rejected(shape) -> None:
    """Une frame de dimension nulle ou tronquée doit lever un message clair."""
    manager = ZoneManager([_zone()])
    with pytest.raises(ZoneConfigurationError):
        manager.initialize(shape)


# ---------------------------------------------------------------------------
# Appartenance
# ---------------------------------------------------------------------------


def test_point_inside_and_outside() -> None:
    """Cas nominal : moitié gauche dedans, moitié droite dehors."""
    manager = _ready_manager(_zone())

    assert _dedans(manager, _at(250.0, 500.0)) == ["Quai"]
    assert _dedans(manager, _at(750.0, 500.0)) == []


def test_point_on_the_border_counts_as_inside() -> None:
    """Choix conservateur pour un système d'alerte : le bord appartient à la zone."""
    manager = _ready_manager(_zone())

    assert _dedans(manager, _at(500.0, 500.0)) == ["Quai"]


def test_anchor_is_the_feet_not_the_box_centre() -> None:
    """Le test porte sur le point d'appui au sol, pas sur le centre de la boîte.

    Boîte haute de 400 px dont le bas touche y=500 : son centre est à y=300,
    **hors** de la zone qui commence à y=400, alors que ses pieds sont dedans.
    Tester le centre manquerait l'intrusion — c'est le cas typique d'une caméra
    en plongée.
    """
    zone = _zone(polygon=((0.0, 0.4), (1.0, 0.4), (1.0, 1.0), (0.0, 1.0)))
    manager = _ready_manager(zone)
    detection = _at(500.0, 500.0, height=400.0)

    assert detection.center[1] == 300.0  # hors zone
    assert detection.anchor[1] == 500.0  # dans la zone
    assert _dedans(manager, detection) == ["Quai"]


def test_anchor_mode_is_configurable(monkeypatch) -> None:
    """Basculer sur 'center' change le point testé, sans toucher au code métier."""
    zone = _zone(polygon=((0.0, 0.4), (1.0, 0.4), (1.0, 1.0), (0.0, 1.0)))
    manager = _ready_manager(zone)
    detection = _at(500.0, 500.0, height=400.0)

    monkeypatch.setattr(config, "GEOMETRY", config.GeometryConfig(anchor="center"))
    assert _dedans(manager, detection) == []


def test_unknown_anchor_falls_back_instead_of_crashing(monkeypatch) -> None:
    """Une faute de frappe dans la configuration ne doit pas arrêter l'analyse."""
    manager = _ready_manager(_zone())
    monkeypatch.setattr(config, "GEOMETRY", config.GeometryConfig(anchor="milieu"))

    assert _dedans(manager, _at(250.0, 500.0)) == ["Quai"]


def test_overlapping_zones_are_all_reported() -> None:
    """Un objet peut occuper deux zones à la fois ; chacune tiendra son chronomètre."""
    left = _zone(name="Quai", polygon=((0.0, 0.0), (0.6, 0.0), (0.6, 1.0), (0.0, 1.0)))
    right = _zone(name="Hall", polygon=((0.4, 0.0), (1.0, 0.0), (1.0, 1.0), (0.4, 1.0)))
    manager = _ready_manager(left, right)

    assert _dedans(manager, _at(500.0, 500.0)) == ["Hall", "Quai"]


def test_zones_for_all_omits_objects_outside_every_zone() -> None:
    """Le dictionnaire ne contient que les objets localisés ; les autres s'obtiennent par .get()."""
    manager = _ready_manager(_zone())
    detections = [_at(250.0, 500.0), _at(900.0, 500.0), _at(100.0, 100.0)]

    located = manager.zones_for_all(detections)

    assert sorted(located) == [0, 2], "Seuls les objets localisés figurent au dictionnaire."
    assert all(marges["Quai"] >= 0 for marges in located.values())
    assert located.get(1, []) == []


def test_restricted_zone_names_filters_on_the_flag() -> None:
    """Seules les zones marquées restricted alimentent la règle d'intrusion."""
    manager = _ready_manager(
        _zone(name="Quai", restricted=True),
        _zone(name="Hall", polygon=((0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0)), restricted=False),
    )

    assert manager.restricted_zone_names() == ["Quai"]
    assert manager.names == ["Quai", "Hall"]
    assert len(manager) == 2


# ---------------------------------------------------------------------------
# Dessin
# ---------------------------------------------------------------------------


def test_draw_returns_a_new_image_and_leaves_the_original_intact() -> None:
    """La frame d'origine sert de preuve : elle ne doit jamais être annotée en place."""
    manager = _ready_manager(_zone())
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)

    annotated = manager.draw(frame)

    assert annotated.shape == frame.shape
    assert annotated is not frame
    assert frame.sum() == 0, "La frame d'origine a été modifiée."
    assert annotated.sum() > 0, "Aucune zone n'a été dessinée."


def test_full_frame_zone_is_not_painted_over_the_video() -> None:
    """Une zone couvrant tout le champ ne doit pas teinter l'image entière.

    C'est le mode de surveillance globale : le périmètre existe pour chronométrer
    les présences, mais le tracer masquerait précisément ce que l'opérateur
    regarde.
    """
    manager = _ready_manager(
        config.SurveillanceZone(
            name="Champ de la caméra",
            polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            draw=False,
        )
    )
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)

    annotated = manager.draw(frame)

    assert annotated.sum() == 0, "La zone plein cadre a été peinte sur la vidéo."
    assert _dedans(manager, _at(500.0, 500.0)) == ["Champ de la caméra"]


def test_default_configuration_watches_the_whole_frame() -> None:
    """Par défaut, tout ce qui est visible est surveillé.

    Un objet situé dans n'importe quel coin de l'image doit être couvert par une
    zone restreinte, sinon les règles d'intrusion ne se déclencheraient que sur
    une portion du champ.
    """
    manager = ZoneManager()
    manager.initialize(FRAME_SHAPE)
    restricted = set(manager.restricted_zone_names())

    for x, y in ((5.0, 5.0), (995.0, 5.0), (500.0, 500.0), (995.0, 995.0)):
        assert restricted & set(_dedans(manager, _at(x, y))), f"angle mort en ({x}, {y})"


def test_sub_zones_remain_possible() -> None:
    """Le découpage en zones reste pleinement fonctionnel si on le configure."""
    manager = _ready_manager(
        _zone(name="Quai", polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))),
        _zone(
            name="Hall",
            polygon=((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)),
            restricted=False,
        ),
    )

    assert _dedans(manager, _at(250.0, 500.0)) == ["Quai"]
    assert _dedans(manager, _at(750.0, 500.0)) == ["Hall"]
    assert manager.restricted_zone_names() == ["Quai"]


def test_draw_requires_initialization() -> None:
    """Dessiner sans résolution connue est la même erreur que tester sans initialiser."""
    manager = ZoneManager([_zone()])
    with pytest.raises(RuntimeError, match="initialize"):
        manager.draw(np.zeros(FRAME_SHAPE, dtype=np.uint8))


def test_zone_labels_are_transliterated_for_opencv() -> None:
    """Les polices Hershey d'OpenCV ne savent pas dessiner les accents."""
    assert _ascii_label("Zone réservée n°2") == "Zone reservee n2"
    assert _ascii_label("Quai de chargement") == "Quai de chargement"
