"""Tests de `sentinel.detector` — SANS charger YOLO ni toucher au GPU.

C'est le bénéfice concret de la frontière d'anti-corruption : `_parse_results()`
est la seule méthode qui manipule des tenseurs Ultralytics, on peut donc la
tester avec un faux objet « results » de quelques lignes. Les tests tournent en
millisecondes, sans réseau et sans poids de modèle.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.detection import Detection, count_by_class
from sentinel.detector import Detector


# ---------------------------------------------------------------------------
# Faux objets Ultralytics
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Imite le minimum de l'API tenseur utilisé : `.cpu().numpy()`."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeBoxes:
    """Imite `ultralytics.engine.results.Boxes`."""

    def __init__(self, xyxy, conf, cls, ids=None) -> None:
        self.xyxy = _FakeTensor(np.array(xyxy, dtype=np.float32))
        self.conf = _FakeTensor(np.array(conf, dtype=np.float32))
        self.cls = _FakeTensor(np.array(cls, dtype=np.float32))
        self.id = None if ids is None else _FakeTensor(np.array(ids, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.xyxy.numpy())


class _FakeResult:
    """Imite `ultralytics.engine.results.Results`."""

    def __init__(self, boxes: _FakeBoxes, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


def _detector_without_model(names: dict[int, str] | None = None) -> Detector:
    """Construit un `Detector` sans exécuter `__init__` (donc sans charger YOLO).

    `object.__new__` court-circuite le constructeur ; on injecte à la main les
    seuls attributs dont `_parse_results` et `_resolve_class_ids` ont besoin.
    """
    detector = object.__new__(Detector)
    detector._names = names or {0: "person", 24: "backpack", 2: "car"}
    detector.weights = "test.pt"
    return detector


# ---------------------------------------------------------------------------
# _parse_results
# ---------------------------------------------------------------------------


def test_parse_results_without_tracking() -> None:
    """En mode predict(), `boxes.id` vaut None : les track_id doivent être None."""
    detector = _detector_without_model()
    result = _FakeResult(
        _FakeBoxes(xyxy=[[10, 20, 50, 120]], conf=[0.91], cls=[0]),
        names={0: "person"},
    )

    detections = detector._parse_results([result])

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == "person"
    assert detection.confidence == pytest.approx(0.91)
    assert detection.xyxy == (10.0, 20.0, 50.0, 120.0)
    assert detection.track_id is None
    assert detection.is_tracked is False


def test_parse_results_with_tracking() -> None:
    """En mode track(), chaque détection porte son identifiant ByteTrack."""
    detector = _detector_without_model()
    result = _FakeResult(
        _FakeBoxes(
            xyxy=[[0, 0, 10, 10], [20, 20, 40, 60]],
            conf=[0.8, 0.6],
            cls=[0, 24],
            ids=[7, 12],
        ),
        names={0: "person", 24: "backpack"},
    )

    detections = detector._parse_results([result])

    assert [d.track_id for d in detections] == [7, 12]
    assert [d.class_name for d in detections] == ["person", "backpack"]
    assert all(d.is_tracked for d in detections)


def test_parse_results_handles_empty_frame() -> None:
    """Une frame sans objet ne doit pas lever, mais retourner une liste vide."""
    detector = _detector_without_model()
    empty = _FakeResult(_FakeBoxes(xyxy=[], conf=[], cls=[]), names={})

    assert detector._parse_results([empty]) == []
    assert detector._parse_results([]) == []


def test_parse_results_unknown_class_id_falls_back_to_index() -> None:
    """Un index absent du dictionnaire de noms ne doit pas faire planter."""
    detector = _detector_without_model()
    result = _FakeResult(_FakeBoxes(xyxy=[[0, 0, 5, 5]], conf=[0.5], cls=[99]), names={})

    assert detector._parse_results([result])[0].class_name == "99"


# ---------------------------------------------------------------------------
# Filtrage par classes
# ---------------------------------------------------------------------------


def test_resolve_class_ids_maps_names_to_indices() -> None:
    """Les noms de classes sont traduits en index compris par YOLO."""
    detector = _detector_without_model()
    assert detector._resolve_class_ids(["person", "car"]) == [0, 2]


def test_resolve_class_ids_ignores_unknown_names() -> None:
    """Un nom inconnu est ignoré (journalisé), pas levé en exception."""
    detector = _detector_without_model()
    assert detector._resolve_class_ids(["person", "licorne"]) == [0]


def test_resolve_class_ids_returns_none_for_all_classes() -> None:
    """Séquence vide ou entièrement invalide = None = toutes les classes."""
    detector = _detector_without_model()
    assert detector._resolve_class_ids([]) is None
    assert detector._resolve_class_ids(["licorne", "dragon"]) is None


# ---------------------------------------------------------------------------
# Validation des frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.array([]),
        np.zeros((480, 640), dtype=np.uint8),  # 2 dimensions au lieu de 3
    ],
)
def test_validate_frame_rejects_invalid_input(frame) -> None:
    """Une frame absente, vide ou mal formée est rejetée explicitement."""
    with pytest.raises(ValueError):
        Detector._validate_frame(frame)


def test_validate_frame_accepts_valid_frame() -> None:
    """Une frame BGR correcte passe la validation."""
    Detector._validate_frame(np.zeros((480, 640, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Géométrie de Detection
# ---------------------------------------------------------------------------


def test_detection_anchor_is_bottom_center() -> None:
    """Le point d'appui est le milieu du bord inférieur, pas le centre."""
    detection = Detection(0, "person", 0.9, (100.0, 50.0, 200.0, 250.0))

    assert detection.center == (150.0, 150.0)
    assert detection.anchor == (150.0, 250.0)
    assert detection.width == 100.0
    assert detection.height == 200.0
    assert detection.area == 20000.0


def test_count_by_class_aggregates_names() -> None:
    """Le comptage par classe agrège correctement les détections."""
    detections = [
        Detection(0, "person", 0.9, (0, 0, 1, 1)),
        Detection(0, "person", 0.8, (2, 2, 3, 3)),
        Detection(2, "car", 0.7, (4, 4, 5, 5)),
    ]

    assert count_by_class(detections) == {"person": 2, "car": 1}
