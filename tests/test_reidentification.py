"""Ré-association des pistes perdues : le doute vaut refus.

Le compromis, énoncé une fois
-------------------------------
Au-delà de `TRACKING.max_age_s`, une piste est purgée et l'objet qui réapparaît
reçoit un nouvel identifiant : son chronomètre repart à zéro. Une personne qui
rôde depuis cinquante secondes et passe cinq secondes derrière un camion redevient
« vue à l'instant ».

Mais **le mode de panne du remède est pire que la maladie**. Un chronomètre remis
à zéro fait manquer un incident — faux négatif, visible et corrigeable. Une
ré-association erronée **fusionne deux personnes en une seule piste**, et le
rapport affirme qu'une personne est restée quarante minutes là où deux se sont
succédé. Un document présenté comme opposable énonce alors un fait faux.

Ce que ces tests verrouillent
-------------------------------
La majorité d'entre eux vérifient un **refus**, pas une réussite. C'est
volontaire : dans ce module, le comportement par défaut à préserver est de ne
rien faire. La fonctionnalité est d'ailleurs désactivée dans la configuration
livrée — l'activer est un choix d'exploitation, pas un comportement subi.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from sentinel.detection import Detection
from sentinel.reidentification import (
    LostTrack,
    ReidentificationBuffer,
    colour_signature,
)

REGLAGES = config.ReidConfig(enabled=True)


def _image(couleur: tuple[int, int, int] = (60, 200, 120)) -> np.ndarray:
    """Image unie, avec un peu de texture pour que l'histogramme ne soit pas plat."""
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image[:, :] = couleur
    bruit = np.indices((600, 800)).sum(axis=0) % 17
    image[:, :, 0] = np.clip(image[:, :, 0].astype(int) + bruit, 0, 255).astype(np.uint8)
    return image


def _detection(
    x: float = 300.0, y: float = 400.0, *, height: float = 120.0, track_id: int = 99
) -> Detection:
    """Détection dont le point d'appui tombe en (x, y)."""
    return Detection(0, "person", 0.9, (x - 25.0, y - height, x + 25.0, y), track_id)


# Sentinelle distincte de `None` : `signature=None` est un cas de test à part
# entière (« la capture d'apparence a échoué »), pas une demande de valeur par
# défaut.
_DEFAUT = object()


def _perdue(
    *,
    position=(300.0, 400.0),
    velocity=(0.0, 0.0),
    scale_px: float = 120.0,
    signature=_DEFAUT,
    lost_at: float = 10.0,
    class_name: str = "person",
) -> LostTrack:
    """Piste perdue prête à l'emploi."""
    if signature is _DEFAUT:
        signature = colour_signature(
            _image(), (275.0, 280.0, 325.0, 400.0), REGLAGES
        )
    return LostTrack(
        track_id=7,
        class_name=class_name,
        position=position,
        velocity=velocity,
        scale_px=scale_px,
        signature=signature,
        lost_at=lost_at,
    )


def _tampon(*pistes: LostTrack, **surcharges) -> ReidentificationBuffer:
    """Tampon actif contenant les pistes fournies."""
    tampon = ReidentificationBuffer(config.ReidConfig(enabled=True, **surcharges))
    for piste in pistes:
        tampon.remember(piste)
    return tampon


# ---------------------------------------------------------------------------
# La position par défaut
# ---------------------------------------------------------------------------


def test_the_feature_is_disabled_in_the_shipped_configuration() -> None:
    """Entre manquer un incident et en fabriquer un, on choisit le premier.

    L'activation est un choix d'exploitation, pris en connaissance du compromis.
    """
    assert config.REID.enabled is False


def test_a_disabled_buffer_remembers_nothing() -> None:
    """Désactivée, la fonctionnalité ne doit rien coûter, pas même en mémoire."""
    tampon = ReidentificationBuffer(config.ReidConfig(enabled=False))

    tampon.remember(_perdue())

    assert len(tampon) == 0


def test_a_disabled_buffer_never_matches() -> None:
    """Le chemin de décision est court-circuité, pas seulement vidé."""
    tampon = ReidentificationBuffer(config.ReidConfig(enabled=False))
    tampon._lost.append(_perdue())  # forcé, pour tester le garde

    assert tampon.match(_detection(), 14.0, _image()) is None


def test_the_default_tracker_behaviour_is_unchanged() -> None:
    """Zéro régression : le tracker livré n'invente aucune identité."""
    from sentinel.tracker import Tracker

    class _Detecteur:
        def track(self, frame):
            return []

    tracker = Tracker(_Detecteur())

    assert tracker._reid.settings.enabled is False


# ---------------------------------------------------------------------------
# Le doute vaut refus
# ---------------------------------------------------------------------------


def test_no_frame_means_no_reassociation() -> None:
    """Sans image, la condition d'apparence est **invérifiable**.

    Et une condition non vérifiée n'est pas une condition remplie. Il ne
    resterait que la géométrie, que deux personnes côte à côte satisfont toutes
    les deux.
    """
    tampon = _tampon(_perdue())

    assert tampon.match(_detection(), 14.0, None) is None
    assert tampon.refusals == 1


def test_a_patch_too_small_to_describe_is_refused() -> None:
    """Un histogramme calculé sur douze pixels est du bruit."""
    tampon = _tampon(_perdue())
    minuscule = Detection(0, "person", 0.9, (300.0, 400.0, 303.0, 404.0), 99)

    assert tampon.match(minuscule, 14.0, _image()) is None


def test_two_equally_plausible_candidates_are_refused() -> None:
    """LA situation où une erreur fusionnerait deux personnes.

    Deux pistes perdues au même endroit, de même taille, de même apparence : rien
    ne permet de les départager. On refuse plutôt que de tirer au sort.
    """
    image = _image()
    signature = colour_signature(image, (275.0, 280.0, 325.0, 400.0), REGLAGES)
    jumelle_a = _perdue(signature=signature)
    jumelle_b = _perdue(signature=signature)
    jumelle_b.track_id = 8

    tampon = _tampon(jumelle_a, jumelle_b)

    assert tampon.match(_detection(), 14.0, image) is None
    assert tampon.refusals == 1


def test_a_different_class_is_refused() -> None:
    """Une personne ne prolonge pas la piste d'une valise."""
    tampon = _tampon(_perdue(class_name="suitcase"))

    assert tampon.match(_detection(), 14.0, _image()) is None


def test_a_gap_that_is_too_short_is_refused() -> None:
    """En dessous de `min_gap_s`, la rétention normale du tracker suffit.

    Doubler un mécanisme qui fonctionne n'ajoute que des occasions de se tromper.
    """
    tampon = _tampon(_perdue(lost_at=10.0))

    assert tampon.match(_detection(), 11.0, _image()) is None


def test_a_gap_that_is_too_long_is_refused() -> None:
    """Au-delà de `max_gap_s`, la prédiction n'a plus de sens.

    L'objet a eu le temps de faire demi-tour, et l'éclairage a pu changer.
    """
    tampon = _tampon(_perdue(lost_at=10.0))

    assert tampon.match(_detection(), 60.0, _image()) is None


def test_an_object_reappearing_far_away_is_refused() -> None:
    """Condition 1 : la position prédite. L'objet devrait être là où il allait."""
    tampon = _tampon(_perdue(position=(100.0, 400.0)))

    assert tampon.match(_detection(x=700.0), 14.0, _image()) is None


def test_a_sudden_change_of_apparent_size_is_refused() -> None:
    """Condition 2 : un objet ne change pas brutalement de profondeur.

    Écarte le passant du premier plan confondu avec la silhouette du fond.
    """
    tampon = _tampon(_perdue(scale_px=120.0))

    assert tampon.match(_detection(height=400.0), 14.0, _image()) is None


def test_a_different_colour_is_refused() -> None:
    """Condition 3 : un manteau rouge ne redevient pas bleu.

    C'est la seule condition qui porte sur l'apparence, donc la seule qui
    distingue deux personnes également placées et également grandes.
    """
    rouge = colour_signature(_image((30, 30, 220)), (275.0, 280.0, 325.0, 400.0), REGLAGES)
    tampon = _tampon(_perdue(signature=rouge))

    bleue = _image((220, 40, 30))

    assert tampon.match(_detection(), 14.0, bleue) is None


def test_a_track_lost_without_a_signature_is_refused() -> None:
    """Une piste dont l'apparence n'a pas pu être capturée reste irrécupérable.

    C'est délibéré : lui appliquer les deux seules conditions géométriques
    rouvrirait exactement le risque que la troisième existe pour fermer.
    """
    tampon = _tampon(_perdue(signature=None))

    assert tampon.match(_detection(), 14.0, _image()) is None


def test_an_empty_buffer_refuses_quietly() -> None:
    """Aucune piste en attente : rien à faire, et surtout rien à inventer."""
    tampon = ReidentificationBuffer(REGLAGES)

    assert tampon.match(_detection(), 14.0, _image()) is None


# ---------------------------------------------------------------------------
# Le cas nominal
# ---------------------------------------------------------------------------


def test_the_same_person_reappearing_nearby_is_recovered() -> None:
    """Le cas visé : un poteau, un camion, cinq secondes d'occlusion."""
    image = _image()
    signature = colour_signature(image, (275.0, 280.0, 325.0, 400.0), REGLAGES)
    tampon = _tampon(_perdue(signature=signature, lost_at=10.0))

    retrouvee = tampon.match(_detection(x=310.0), 15.0, image)

    assert retrouvee is not None
    assert retrouvee.track_id == 7
    assert tampon.matches == 1


def test_a_recovered_track_leaves_the_buffer() -> None:
    """Une piste ne peut prolonger qu'un seul objet."""
    image = _image()
    signature = colour_signature(image, (275.0, 280.0, 325.0, 400.0), REGLAGES)
    tampon = _tampon(_perdue(signature=signature))

    tampon.match(_detection(), 14.0, image)

    assert len(tampon) == 0


def test_the_prediction_follows_the_movement() -> None:
    """Un objet en marche est cherché là où il allait, pas là où il s'est arrêté."""
    piste = _perdue(position=(100.0, 400.0), velocity=(40.0, 0.0), lost_at=10.0)

    assert piste.predicted_position(15.0) == (300.0, 400.0)


def test_a_moving_object_is_recovered_at_its_predicted_position() -> None:
    """Sans prédiction, une personne qui marche serait déclarée trop loin."""
    image = _image()
    signature = colour_signature(image, (275.0, 280.0, 325.0, 400.0), REGLAGES)
    tampon = _tampon(
        _perdue(position=(100.0, 400.0), velocity=(40.0, 0.0), signature=signature)
    )

    assert tampon.match(_detection(x=300.0), 15.0, image) is not None


# ---------------------------------------------------------------------------
# Cycle de vie du tampon
# ---------------------------------------------------------------------------


def test_stale_tracks_are_forgotten() -> None:
    """Un tampon qui grossit indéfiniment finirait par ralentir l'analyse."""
    tampon = _tampon(_perdue(lost_at=10.0))

    oubliees = tampon.expire(60.0)

    assert oubliees == 1
    assert len(tampon) == 0


def test_reset_clears_everything_between_sessions() -> None:
    """Deux vidéos successives ne doivent pas se prêter d'identités."""
    tampon = _tampon(_perdue())
    tampon.match(_detection(), 14.0, None)  # incrémente les refus

    tampon.reset()

    assert len(tampon) == 0
    assert tampon.refusals == 0
    assert tampon.matches == 0


# ---------------------------------------------------------------------------
# La signature couleur
# ---------------------------------------------------------------------------


def test_no_image_gives_no_signature() -> None:
    """`None` est la réponse honnête, et vaut refus en aval."""
    assert colour_signature(None, (0.0, 0.0, 10.0, 10.0), REGLAGES) is None


def test_an_out_of_frame_box_gives_no_signature() -> None:
    """Une boîte hors cadre ne décrit rien."""
    assert colour_signature(_image(), (900.0, 700.0, 950.0, 750.0), REGLAGES) is None


def test_the_same_colour_scores_higher_than_a_different_one() -> None:
    """Le principe même de la troisième condition."""
    import cv2

    boite = (275.0, 280.0, 325.0, 400.0)
    verte = colour_signature(_image((60, 200, 120)), boite, REGLAGES)
    verte_bis = colour_signature(_image((60, 200, 120)), boite, REGLAGES)
    rouge = colour_signature(_image((30, 30, 220)), boite, REGLAGES)

    identique = cv2.compareHist(verte, verte_bis, cv2.HISTCMP_CORREL)
    differente = cv2.compareHist(verte, rouge, cv2.HISTCMP_CORREL)

    assert identique > differente


# ---------------------------------------------------------------------------
# La mémoire survit à l'occlusion
# ---------------------------------------------------------------------------


def test_a_resurrected_object_keeps_its_clock() -> None:
    """C'est tout l'intérêt : sans restauration, le chronomètre repartirait à zéro.

    Une personne qui rôde depuis cinquante secondes et passe cinq secondes
    derrière un camion doit rester une personne qui rôde depuis cinquante-cinq
    secondes.
    """
    from datetime import datetime

    from sentinel.tracker import Tracker

    class _Detecteur:
        def track(self, frame):
            return []

    tracker = Tracker(_Detecteur())
    tracker._video_time = 55.0
    piste = _perdue()
    piste.memory = {
        "first_seen": 0.0,
        "first_seen_wall": datetime(2026, 8, 21, 10, 0, 0),
        "hits": 120,
        "history": [(1.0, (300.0, 400.0))],
        "zones": {"Hall"},
        "zone_entry_time": {"Hall": 2.0},
        "last_event_time": {},
        "class_votes": {"person": 120},
        "owner_votes": {},
    }

    ressuscite = tracker._resurrect(piste, _detection(), datetime(2026, 8, 21, 10, 1, 0))

    assert ressuscite.age == pytest.approx(55.0)
    assert ressuscite.dwell_time("Hall", 55.0) == pytest.approx(53.0)
    assert ressuscite.hits == 120


def test_velocity_is_zero_without_enough_history() -> None:
    """Une vitesse inventée ferait prédire une position fausse.

    La condition de position deviendrait alors un tirage au sort — exactement ce
    que ce module refuse.
    """
    from datetime import datetime

    from sentinel.tracker import Tracker, TrackedObject

    detection = _detection()
    obj = TrackedObject(
        1, "person", 0.0, 0.0, datetime.now(), datetime.now(), detection
    )

    assert Tracker._velocity(obj) == (0.0, 0.0)
