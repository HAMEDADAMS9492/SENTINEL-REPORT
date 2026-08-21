"""Tests de `sentinel.tracker` — SANS modèle, sans vidéo, sans GPU.

`Tracker` ne dépend du détecteur que par une seule méthode (`track(frame)`), ce
qui permet de le remplacer par une dizaine de lignes et de **scénariser** des
séquences de frames impossibles à reproduire avec une vraie vidéo : occlusion de
2,5 secondes exactement, sac immobile au pixel près, personne qui sort et rentre
dans une zone. C'est tout l'intérêt d'avoir gardé `tracker.py` indépendant
d'Ultralytics et de Streamlit.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import config
from sentinel.detection import Detection
from sentinel.tracker import TrackedObject, Tracker

FPS = 10.0  # 0,1 s par frame : les durées se lisent directement dans les tests


# ---------------------------------------------------------------------------
# Doublures et fabriques
# ---------------------------------------------------------------------------


class _FakeDetector:
    """Détecteur scénarisé : renvoie la liste de détections prévue par frame."""

    def __init__(self, script: list[list[Detection]]) -> None:
        self._script = script
        self._index = 0
        self.reset_calls = 0

    def track(self, frame, *, persist: bool = True) -> list[Detection]:
        if self._index >= len(self._script):
            return []
        detections = self._script[self._index]
        self._index += 1
        return detections

    def reset(self) -> None:
        self.reset_calls += 1


def _detection(
    track_id: int | None,
    *,
    x: float = 100.0,
    y: float = 200.0,
    class_name: str = "person",
) -> Detection:
    """Détection carrée de 40 px dont le point d'appui vaut exactement (x, y)."""
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        xyxy=(x - 20.0, y - 40.0, x + 20.0, y),
        track_id=track_id,
    )


def _run(script: list[list[Detection]], *, fps: float = FPS) -> Tracker:
    """Rejoue un scénario complet et retourne le tracker dans son état final."""
    tracker = Tracker(_FakeDetector(script))
    for index in range(len(script)):
        tracker.update(None, frame_index=index, fps=fps)
    return tracker


def _tracked_object(points: list[tuple[float, float]], *, step_s: float = 0.1) -> TrackedObject:
    """Construit un objet suivi dont l'historique parcourt `points`."""
    first = _detection(1, x=points[0][0], y=points[0][1])
    obj = TrackedObject(
        track_id=1,
        class_name="backpack",
        first_seen=0.0,
        last_seen=0.0,
        first_seen_wall=datetime(2026, 1, 1, 12, 0, 0),
        last_seen_wall=datetime(2026, 1, 1, 12, 0, 0),
        detection=first,
    )
    obj._push_history(0.0, points[0])
    for index, (x, y) in enumerate(points[1:], start=1):
        obj.update(_detection(1, x=x, y=y), index * step_s, datetime(2026, 1, 1, 12, 0, 0))
    return obj


# ---------------------------------------------------------------------------
# Confirmation et cycle de vie
# ---------------------------------------------------------------------------


def test_object_is_confirmed_only_after_min_hits() -> None:
    """Une détection fugace ne doit jamais atteindre les règles d'événements."""
    tracker = Tracker(_FakeDetector([[_detection(1)]] * 5))

    returned = [tracker.update(None, frame_index=i, fps=FPS) for i in range(5)]

    # min_hits = 3 : les deux premières frames ne retournent rien...
    assert [len(objects) for objects in returned[: config.TRACKING.min_hits - 1]] == [0, 0]
    # ...mais l'objet existe déjà en mémoire, il n'est simplement pas confirmé.
    assert len(tracker) == 1
    assert tracker.get(1).is_confirmed is True
    assert tracker.get(1).hits == 5


def test_untracked_detections_are_kept_aside() -> None:
    """Une détection sans track_id est affichable mais n'entre pas en mémoire."""
    tracker = Tracker(_FakeDetector([[_detection(None), _detection(4, x=300.0)]]))

    tracker.update(None, frame_index=0, fps=FPS)

    assert len(tracker) == 1
    assert tracker.get(4) is not None
    assert len(tracker.untracked) == 1
    assert tracker.untracked[0].track_id is None


def test_video_time_is_derived_from_frame_index() -> None:
    """Le temps métier vient du numéro de frame, jamais de l'horloge réelle."""
    tracker = _run([[_detection(1)]] * 26)

    assert tracker.video_time == pytest.approx(25 / FPS)


def test_absent_fps_falls_back_to_configuration() -> None:
    """Une webcam qui n'expose pas son FPS ne doit pas provoquer de division par zéro."""
    tracker = Tracker(_FakeDetector([[_detection(1)]]))

    tracker.update(None, frame_index=10, fps=0.0)

    assert tracker.video_time == pytest.approx(10 / config.VIDEO.default_fps)


# ---------------------------------------------------------------------------
# Purge et occlusions
# ---------------------------------------------------------------------------


def test_short_occlusion_does_not_reset_the_object() -> None:
    """Passer 2 s derrière un poteau ne doit pas remettre le chronomètre à zéro."""
    seen = [[_detection(1)]] * 5
    occluded = [[]] * 20  # 2,0 s d'absence, sous max_age_s = 3,0 s
    tracker = _run(seen + occluded + [[_detection(1)]])

    obj = tracker.get(1)
    assert obj is not None
    assert obj.first_seen == 0.0  # même objet, chronomètre intact
    assert obj.misses == 0


def test_long_absence_purges_the_object() -> None:
    """Au-delà de max_age_s, l'objet est oublié : un retour créerait un nouvel ID."""
    absence_frames = int(config.TRACKING.max_age_s * FPS) + 5
    tracker = _run([[_detection(1)]] * 5 + [[]] * absence_frames)

    assert tracker.get(1) is None
    assert len(tracker) == 0


# ---------------------------------------------------------------------------
# Chronomètres de zone
# ---------------------------------------------------------------------------


def test_dwell_time_starts_at_entry_and_is_not_reset_each_frame() -> None:
    """Le chronomètre démarre à l'entrée ; y rester ne le réécrit pas."""
    obj = _tracked_object([(100.0, 200.0)])

    obj.update_zones(["Quai"], video_time=10.0)
    obj.update_zones(["Quai"], video_time=11.0)
    obj.update_zones(["Quai"], video_time=13.0)

    assert obj.dwell_time("Quai", video_time=13.0) == pytest.approx(3.0)


def _enter(obj: TrackedObject, zones: list[str], start: float, *, step: float = 0.1) -> None:
    """Fait entrer l'objet dans des zones en respectant l'hystérésis d'entrée."""
    for index in range(config.GEOMETRY.min_overlap_frames):
        obj.update_zones(zones, video_time=start + index * step)


def _leave(obj: TrackedObject, start: float, *, step: float = 0.1) -> None:
    """Fait sortir l'objet de toutes ses zones en respectant l'hystérésis de sortie."""
    for index in range(config.GEOMETRY.exit_tolerance_frames):
        obj.update_zones([], video_time=start + index * step)


def test_leaving_a_zone_clears_its_timer() -> None:
    """Sortir puis revenir redémarre le compte — une intrusion n'est pas cumulative."""
    obj = _tracked_object([(100.0, 200.0)])

    _enter(obj, ["Quai"], 10.0)
    _leave(obj, 14.0)
    assert obj.dwell_time("Quai", video_time=14.0) == 0.0

    _enter(obj, ["Quai"], 20.0)
    assert obj.dwell_time("Quai", video_time=21.0) == pytest.approx(1.0)


def test_object_can_occupy_several_zones_at_once() -> None:
    """Deux zones qui se chevauchent tiennent chacune leur propre chronomètre."""
    obj = _tracked_object([(100.0, 200.0)])

    _enter(obj, ["Quai"], 5.0)
    _enter(obj, ["Quai", "Hall"], 8.0)

    assert obj.dwell_time("Quai", video_time=10.0) == pytest.approx(5.0)
    assert obj.dwell_time("Hall", video_time=10.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Hystérésis des frontières de zone
# ---------------------------------------------------------------------------


def test_single_frame_incursion_does_not_count_as_an_entry() -> None:
    """Une frame isolée dans la zone est du bruit, pas une intrusion."""
    obj = _tracked_object([(100.0, 200.0)])

    obj.update_zones(["Quai"], video_time=5.0)

    assert obj.zones == set()
    assert obj.dwell_time("Quai", video_time=6.0) == 0.0


def test_entry_timer_starts_at_first_contact_not_at_confirmation() -> None:
    """Les frames d'observation ont été passées dans la zone : elles comptent.

    Faire partir le chronomètre à la confirmation amputerait chaque durée de
    `min_overlap_frames` frames, soit un biais systématique sur tous les rapports.
    """
    obj = _tracked_object([(100.0, 200.0)])

    obj.update_zones(["Quai"], video_time=10.0)  # premier contact
    obj.update_zones(["Quai"], video_time=10.5)  # confirmation

    assert obj.zone_entry_time["Quai"] == pytest.approx(10.0)


def test_flicker_on_the_border_does_not_reset_the_timer() -> None:
    """Le cas qui casse tout : une boîte qui tremble sur la frontière.

    Sans tolérance de sortie, chaque frame perdue remettrait le chronomètre à
    zéro et une intrusion prolongée ne franchirait jamais son seuil.
    """
    obj = _tracked_object([(100.0, 200.0)])
    _enter(obj, ["Quai"], 0.0)

    # 30 s de présence entrecoupées d'une frame manquée toutes les cinq frames.
    for step in range(300):
        moment = 1.0 + step * 0.1
        obj.update_zones([] if step % 5 == 4 else ["Quai"], video_time=moment)

    assert "Quai" in obj.zones
    assert obj.dwell_time("Quai", video_time=31.0) > 30.0


def test_a_real_exit_is_eventually_confirmed() -> None:
    """La tolérance ne doit pas empêcher de constater une sortie franche."""
    obj = _tracked_object([(100.0, 200.0)])
    _enter(obj, ["Quai"], 0.0)

    _leave(obj, 5.0)

    assert obj.zones == set()
    assert obj.dwell_time("Quai", video_time=10.0) == 0.0


# ---------------------------------------------------------------------------
# Stabilité de la classe
# ---------------------------------------------------------------------------


def test_class_is_decided_by_majority_vote() -> None:
    """La première frame ne décide pas de la classe pour toute la vidéo.

    C'est souvent la moins fiable : l'objet y est le plus petit ou le plus
    partiellement visible.
    """
    obj = _tracked_object([(100.0, 200.0)])
    obj.class_votes.clear()
    obj.class_votes["truck"] = 1

    for _ in range(5):
        obj.update(_detection(1, class_name="bus"), 1.0, datetime(2026, 1, 1, 12, 0))

    assert obj.class_name == "bus"


def test_class_vote_ties_are_broken_deterministically() -> None:
    """À égalité, l'ordre alphabétique tranche : deux analyses, même rapport."""
    obj = _tracked_object([(100.0, 200.0)])
    obj.class_votes.clear()
    obj.class_votes.update({"truck": 2, "bus": 1})

    obj.update(_detection(1, class_name="bus"), 1.0, datetime(2026, 1, 1, 12, 0))

    assert obj.class_name == "bus"  # 2 partout -> 'bus' avant 'truck'


def test_dwell_time_is_zero_outside_the_zone() -> None:
    """Interroger une zone où l'objet n'est pas ne lève pas, retourne 0."""
    obj = _tracked_object([(100.0, 200.0)])

    assert obj.dwell_time("Zone inconnue", video_time=99.0) == 0.0


# ---------------------------------------------------------------------------
# Immobilité (règle « objet abandonné »)
# ---------------------------------------------------------------------------


def test_displacement_ignores_detection_jitter() -> None:
    """Une boîte qui tremble de ±2 px reste « immobile » malgré 40 frames.

    C'est précisément ce qu'une distance **cumulée** mesurerait à tort : 40
    frames à 4 px d'aller-retour donneraient 160 px de « déplacement ».
    """
    jitter = [(100.0 + (2.0 if i % 2 else -2.0), 200.0) for i in range(40)]
    obj = _tracked_object(jitter)

    assert obj.displacement(window_s=4.0) <= 4.0


def test_displacement_detects_real_movement() -> None:
    """Un objet qui s'éloigne vraiment dépasse le seuil."""
    walk = [(100.0 + 10.0 * i, 200.0) for i in range(20)]
    obj = _tracked_object(walk)

    assert obj.displacement(window_s=2.0) > 100.0


def test_is_stationary_requires_covering_the_whole_window() -> None:
    """Un sac vu depuis 1 s n'est pas « abandonné depuis 30 s »."""
    obj = _tracked_object([(100.0, 200.0)] * 10)  # 0,9 s d'historique

    assert obj.is_stationary(max_movement_px=25.0, window_s=5.0) is False
    assert obj.is_stationary(max_movement_px=25.0, window_s=0.5) is True


def test_is_stationary_is_false_for_a_moving_object() -> None:
    """Un objet qui se déplace n'est pas immobile, même observé longtemps."""
    obj = _tracked_object([(100.0 + 5.0 * i, 200.0) for i in range(100)])

    assert obj.is_stationary(max_movement_px=25.0, window_s=5.0) is False


def test_history_window_is_bounded() -> None:
    """L'historique ne croît pas indéfiniment : c'est une fenêtre glissante."""
    total_s = config.TRACKING.history_seconds * 2
    obj = _tracked_object([(100.0, 200.0)] * int(total_s * FPS), step_s=1 / FPS)

    span = obj.history[-1][0] - obj.history[0][0]
    assert span <= config.TRACKING.history_seconds


# ---------------------------------------------------------------------------
# Anti-rebond
# ---------------------------------------------------------------------------


def test_cooldown_blocks_immediate_repeat() -> None:
    """Sans délai de garde, une intrusion de 2 min produirait 3 000 rapports."""
    obj = _tracked_object([(100.0, 200.0)])

    assert obj.can_raise("intrusion", video_time=10.0, cooldown_s=60.0) is True
    obj.mark_event("intrusion", video_time=10.0)

    assert obj.can_raise("intrusion", video_time=11.0, cooldown_s=60.0) is False
    assert obj.can_raise("intrusion", video_time=69.9, cooldown_s=60.0) is False
    assert obj.can_raise("intrusion", video_time=70.0, cooldown_s=60.0) is True


def test_cooldown_is_tracked_per_event_type() -> None:
    """Une intrusion signalée ne doit pas masquer une présence prolongée."""
    obj = _tracked_object([(100.0, 200.0)])

    obj.mark_event("intrusion", video_time=10.0)

    assert obj.can_raise("intrusion", video_time=12.0, cooldown_s=60.0) is False
    assert obj.can_raise("presence_prolongee", video_time=12.0, cooldown_s=60.0) is True


# ---------------------------------------------------------------------------
# Requêtes sur la population suivie
# ---------------------------------------------------------------------------


def test_counts_are_per_distinct_object_not_per_frame() -> None:
    """Une personne présente 30 s est comptée une fois, pas 750 fois."""
    frame = [
        _detection(1, x=100.0),
        _detection(2, x=300.0),
        _detection(3, x=500.0, class_name="backpack"),
    ]
    tracker = _run([frame] * 10)

    assert tracker.counts() == {"person": 2, "backpack": 1}
    assert [obj.track_id for obj in tracker.by_class("person")] == [1, 2]


def test_nearest_finds_the_owner_within_radius() -> None:
    """Le sac trouve son propriétaire présumé ; hors rayon, personne."""
    frame = [
        _detection(1, x=100.0, y=200.0, class_name="backpack"),
        _detection(2, x=180.0, y=200.0),  # 80 px du sac
        _detection(3, x=900.0, y=200.0),  # très loin
    ]
    tracker = _run([frame] * 5)
    bag = tracker.get(1)

    assert tracker.nearest(bag, class_names=["person"], radius_px=150.0).track_id == 2
    assert tracker.nearest(bag, class_names=["person"], radius_px=50.0) is None


def test_nearest_ignores_unconfirmed_neighbours() -> None:
    """Un faux positif d'une frame ne doit pas « sauver » un sac abandonné."""
    tracker = Tracker(
        _FakeDetector(
            [
                [_detection(1, x=100.0, class_name="backpack")] * 1,
                [_detection(1, x=100.0, class_name="backpack")],
                [
                    _detection(1, x=100.0, class_name="backpack"),
                    _detection(9, x=120.0),  # apparaît à l'instant
                ],
            ]
        )
    )
    for index in range(3):
        tracker.update(None, frame_index=index, fps=FPS)

    bag = tracker.get(1)
    assert tracker.nearest(bag, class_names=["person"], radius_px=150.0) is None


def test_reset_clears_memory_and_detector_state() -> None:
    """Entre deux vidéos, ni les objets ni les identifiants ne doivent survivre."""
    detector = _FakeDetector([[_detection(1)]] * 5)
    tracker = Tracker(detector)
    for index in range(5):
        tracker.update(None, frame_index=index, fps=FPS)

    tracker.reset()

    assert len(tracker) == 0
    assert tracker.video_time == 0.0
    assert tracker.untracked == []
    assert detector.reset_calls == 1
