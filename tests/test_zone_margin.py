"""Frontière à marge signée : une bande d'incertitude autour du bord.

Le défaut visé
---------------
Une boîte de détection tremble de quelques pixels d'une frame à l'autre. Quand
son point d'appui longe la frontière d'une zone, l'appartenance oscille — et
chaque oscillation **remet le chronomètre d'intrusion à zéro**. Une intrusion de
cinquante secondes le long d'une clôture n'est alors jamais signalée.

Deux hystérésis, deux défauts différents
-----------------------------------------
La bande **spatiale** (ce fichier) absorbe l'imprécision de la boîte : tant que
l'objet est à moins de `margin_ratio × hauteur` du bord, son état ne change pas.
L'hystérésis **temporelle** (compteurs de frames, `test_tracker.py`) absorbe les
détections erratiques : une incursion d'une frame ne vaut pas entrée. La première
ne remplace pas la seconde — un objet peut franchir nettement la bande sur une
frame isolée par une erreur de détection, et seul le compteur l'écarte.

Pourquoi la marge est relative
-------------------------------
Même argument qu'au § 3.5 du README : vingt pixels valent un pas de côté au
premier plan et trois mètres au fond du champ. Rapporter la marge à la hauteur
apparente lui donne le même sens à toutes les profondeurs, sans calibration.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

import config
from sentinel.detection import Detection
from sentinel.tracker import TrackedObject
from sentinel.zones import ZoneManager, _covers_frame

T0 = datetime(2026, 8, 21, 10, 0, 0)

# Moitié gauche de l'image : la frontière verticale tombe à x = 0.5.
MOITIE_GAUCHE = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))


def _manager(*zones: config.SurveillanceZone, taille: tuple[int, int] = (1000, 1000)) -> ZoneManager:
    """Gestionnaire déjà initialisé sur une image carrée de 1000 px."""
    manager = ZoneManager(list(zones))
    manager.initialize((taille[1], taille[0], 3))
    return manager


def _zone(name: str = "Quai", polygon=MOITIE_GAUCHE, **kwargs) -> config.SurveillanceZone:
    """Zone de test, interdite par défaut."""
    return config.SurveillanceZone(name=name, polygon=polygon, **kwargs)


def _at(x: float, y: float, *, height: float = 100.0) -> Detection:
    """Détection dont le point d'appui tombe exactement en (x, y)."""
    return Detection(0, "person", 0.9, (x - 20.0, y - height, x + 20.0, y), 1)


def _obj(track_id: int = 1) -> TrackedObject:
    """Objet suivi vierge."""
    detection = _at(100.0, 500.0)
    return TrackedObject(track_id, "person", 0.0, 0.0, T0, T0, detection)


# ---------------------------------------------------------------------------
# La marge est une distance signée, relative à la taille
# ---------------------------------------------------------------------------


def test_the_margin_is_positive_inside_and_negative_outside() -> None:
    """Le signe dit le côté, la valeur dit de combien."""
    manager = _manager(_zone())

    dedans = manager.zones_for(_at(400.0, 500.0))["Quai"]
    dehors = manager.zones_for(_at(600.0, 500.0))["Quai"]

    assert dedans > 0
    assert dehors < 0


def test_every_zone_is_reported_even_the_ones_left_behind() -> None:
    """La sortie a besoin de savoir de combien l'objet est dehors.

    Ne rendre que les zones occupées priverait l'hystérésis de l'information
    dont elle a besoin pour décider d'une sortie.
    """
    manager = _manager(_zone("Quai"), _zone("Hall", ((0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0))))

    marges = manager.zones_for(_at(100.0, 500.0))

    assert set(marges) == {"Quai", "Hall"}


def test_the_same_pixel_gap_reads_differently_at_two_depths() -> None:
    """Le cœur de la phase : 30 px n'ont pas le même sens au fond et au premier plan.

    Deux personnes sont exactement à 30 px à l'intérieur du bord. Celle du
    premier plan (200 px de haut) en est à 0,15 hauteur — un pas. Celle du fond
    (40 px de haut) en est à 0,75 hauteur — bien à l'intérieur. Un seuil en
    pixels fixes les aurait traitées identiquement.
    """
    manager = _manager(_zone())

    proche = manager.zones_for(_at(470.0, 500.0, height=200.0))["Quai"]
    lointain = manager.zones_for(_at(470.0, 500.0, height=40.0))["Quai"]

    assert proche == pytest.approx(0.15)
    assert lointain == pytest.approx(0.75)


def test_a_degenerate_box_does_not_divide_by_zero() -> None:
    """Une boîte de hauteur nulle ne doit pas faire exploser l'analyse.

    Cela arrive : une détection tronquée au bord du cadre, ou un modèle qui rend
    une boîte plate. Le plancher à 1 px rend la marge très grande plutôt
    qu'infinie — c'est-à-dire « franchement dedans », le choix conservateur pour
    un système d'alerte.
    """
    manager = _manager(_zone())
    plate = Detection(0, "person", 0.9, (400.0, 500.0, 440.0, 500.0), 1)

    marge = manager.zones_for(plate)["Quai"]

    assert math.isfinite(marge)
    assert marge > 0


# ---------------------------------------------------------------------------
# Le bord de l'image n'est pas une frontière
# ---------------------------------------------------------------------------


def test_a_full_frame_zone_gives_a_clear_cut_verdict() -> None:
    """Aucune marge n'est exigée là où le cadre fait office de bord.

    Un objet ne peut pas sortir latéralement de l'image : il disparaît. Exiger
    une marge autour du cadre créerait un anneau aveugle — une personne dont les
    pieds touchent le bas du champ resterait éternellement « en cours d'entrée ».
    """
    manager = _manager(_zone("Champ", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))))

    au_bord = manager.zones_for(_at(500.0, 999.0, height=200.0))["Champ"]

    assert au_bord == math.inf


def test_the_default_configuration_has_no_blind_ring() -> None:
    """Zéro régression sur la configuration livrée.

    La zone plein cadre par défaut doit rester franche sur tout son pourtour.
    """
    manager = _manager(*config.ZONES)

    for x, y in ((1.0, 1.0), (999.0, 1.0), (1.0, 999.0), (999.0, 999.0), (500.0, 999.0)):
        marges = manager.zones_for(_at(x, y, height=300.0))
        assert marges["Champ de la caméra"] == math.inf, f"angle mort en ({x}, {y})"


@pytest.mark.parametrize(
    "polygone, attendu",
    [
        (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), True),
        (((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)), True),  # tracé inversé
        (((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)), False),
        (((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)), False),
        (((0.0, 0.0), (1.0, 0.0)), False),  # dégénéré
    ],
)
def test_full_frame_detection_handles_both_drawing_directions(polygone, attendu) -> None:
    """L'aire est prise en valeur absolue : le sens du tracé n'a pas à compter."""
    assert _covers_frame(polygone) is attendu


# ---------------------------------------------------------------------------
# La bande gèle l'appartenance
# ---------------------------------------------------------------------------


def test_hovering_on_the_border_never_starts_the_clock() -> None:
    """Le cas qui motive la phase : une boîte qui tremble sur la frontière.

    L'objet oscille de part et d'autre du bord sans jamais franchir la bande.
    Sans elle, chaque oscillation validerait puis annulerait une entrée.
    """
    obj = _obj()
    seuil = config.GEOMETRY.margin_ratio

    for index in range(40):
        marge = (seuil / 2.0) * (1 if index % 2 else -1)
        obj.update_zones({"Quai": marge}, float(index))

    assert obj.zones == set()
    assert obj.dwell_time("Quai", 40.0) == 0.0


def test_crossing_the_band_does_enter_the_zone() -> None:
    """La bande retarde l'entrée, elle ne l'empêche pas."""
    obj = _obj()
    franc = config.GEOMETRY.margin_ratio + 0.5

    for index in range(config.GEOMETRY.min_overlap_frames):
        obj.update_zones({"Quai": franc}, float(index))

    assert obj.zones == {"Quai"}


def test_leaving_requires_crossing_the_band_the_other_way() -> None:
    """Sortir demande d'être franchement dehors, pas seulement sur le bord."""
    obj = _obj()
    franc = config.GEOMETRY.margin_ratio + 0.5
    for index in range(config.GEOMETRY.min_overlap_frames):
        obj.update_zones({"Quai": franc}, float(index))
    assert obj.zones == {"Quai"}

    # Vingt frames dans la bande : l'objet reste dedans, le chronomètre court.
    for index in range(20):
        obj.update_zones({"Quai": 0.0}, 10.0 + index)

    assert obj.zones == {"Quai"}
    assert obj.dwell_time("Quai", 30.0) > 0.0


def test_a_clear_exit_still_ends_the_stay() -> None:
    """La bande ne doit pas rendre une zone impossible à quitter."""
    obj = _obj()
    franc = config.GEOMETRY.margin_ratio + 0.5
    for index in range(config.GEOMETRY.min_overlap_frames):
        obj.update_zones({"Quai": franc}, float(index))

    for index in range(config.GEOMETRY.exit_tolerance_frames):
        obj.update_zones({"Quai": -franc}, 10.0 + index)

    assert obj.zones == set()


def test_the_band_freezes_a_pending_entry_instead_of_losing_it() -> None:
    """Revenir dans la bande gèle le décompte d'entrée au lieu de l'annuler.

    L'objet a déjà passé des frames à l'intérieur : les décompter parce qu'il
    frôle le bord reviendrait à punir la précision de la mesure.
    """
    obj = _obj()
    franc = config.GEOMETRY.margin_ratio + 0.5

    obj.update_zones({"Quai": franc}, 0.0)  # 1 frame franche
    obj.update_zones({"Quai": 0.0}, 1.0)  # dans la bande : gel
    obj.update_zones({"Quai": franc}, 2.0)  # 2e frame franche

    assert obj.zones == {"Quai"}, "Le décompte d'entrée a été perdu dans la bande."


def test_a_clear_exit_does_cancel_a_pending_entry() -> None:
    """Une sortie franche, elle, annule bien l'entrée en cours de validation."""
    obj = _obj()
    franc = config.GEOMETRY.margin_ratio + 0.5

    obj.update_zones({"Quai": franc}, 0.0)
    obj.update_zones({"Quai": -franc}, 1.0)
    obj.update_zones({"Quai": franc}, 2.0)

    assert obj.zones == set(), "L'entrée aurait dû repartir de zéro."


# ---------------------------------------------------------------------------
# Compatibilité : une liste de noms reste acceptée
# ---------------------------------------------------------------------------


def test_a_plain_list_of_names_means_definitely_inside() -> None:
    """Un appelant qui ne mesure pas de marge affirme une appartenance franche.

    C'est le cas des tests du tracker, et de tout code qui raisonne en
    appartenance plutôt qu'en distance.
    """
    obj = _obj()

    for index in range(config.GEOMETRY.min_overlap_frames):
        obj.update_zones(["Quai"], float(index))

    assert obj.zones == {"Quai"}


def test_an_omitted_zone_counts_as_definitely_outside() -> None:
    """Une zone absente de la mesure est réputée quittée, pas gelée.

    Sinon un appelant qui ne transmet que les zones occupées ne pourrait plus
    jamais faire sortir un objet.
    """
    obj = _obj()
    for index in range(config.GEOMETRY.min_overlap_frames):
        obj.update_zones(["Quai"], float(index))

    for index in range(config.GEOMETRY.exit_tolerance_frames):
        obj.update_zones([], 10.0 + index)

    assert obj.zones == set()


def test_a_zero_margin_restores_the_previous_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    """`margin_ratio = 0` supprime la bande — utile pour comparer avant/après."""
    monkeypatch.setattr(
        config, "GEOMETRY", config.GeometryConfig(margin_ratio=0.0), raising=True
    )
    obj = _obj()

    for index in range(config.GEOMETRY.min_overlap_frames):
        obj.update_zones({"Quai": 0.0}, float(index))

    assert obj.zones == {"Quai"}, "Sans bande, être sur le bord suffit."
