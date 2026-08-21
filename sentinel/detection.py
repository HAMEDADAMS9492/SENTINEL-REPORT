"""Structure de données commune : une détection sur une frame.

`Detection` est le **contrat** entre le détecteur et tout le reste du pipeline.
Elle est isolée dans son propre module pour que `detector.py`, `tracker.py`,
`zones.py` et `events.py` puissent l'importer sans créer de cycle d'import.

Choix de conception : dataclass `frozen` + `slots`.
- `frozen=True` : une détection est un fait observé à un instant t, elle ne doit
  jamais être modifiée après coup (un rapport doit être reproductible).
- `slots=True` : pas de `__dict__` par instance ; sur une vidéo de 10 minutes à
  25 fps avec 10 objets par frame, cela représente 150 000 objets — l'économie
  mémoire est réelle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """Un objet détecté sur une frame.

    Attributes:
        class_id: Index de la classe dans le modèle (index COCO pour YOLOv8).
        class_name: Nom lisible de la classe (« person », « backpack »...).
        confidence: Score de confiance du modèle, dans [0, 1].
        xyxy: Boîte englobante en pixels absolus `(x1, y1, x2, y2)`, coin
            supérieur gauche et coin inférieur droit.
        track_id: Identifiant persistant attribué par le tracker. `None` quand la
            détection provient de `Detector.detect()` (mode sans suivi) ou quand
            le tracker n'a pas encore confirmé la piste.
    """

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None

    # -- Géométrie dérivée ---------------------------------------------------

    @property
    def center(self) -> tuple[float, float]:
        """Centre géométrique de la boîte, en pixels."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def anchor(self) -> tuple[float, float]:
        """Point d'appui au sol : milieu du bord inférieur de la boîte.

        C'est ce point — et non le centre — qu'on teste contre les polygones de
        zone : pour une caméra en plongée, le centre de la boîte d'une personne
        debout se situe à mi-hauteur du corps et peut tomber hors de la zone
        alors que ses pieds y sont clairement.
        """
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        """Largeur de la boîte en pixels."""
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        """Hauteur de la boîte en pixels."""
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        """Aire de la boîte en pixels carrés."""
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def is_tracked(self) -> bool:
        """True si la détection porte un identifiant de suivi."""
        return self.track_id is not None

    # -- Utilitaires ---------------------------------------------------------

    def as_int_box(self) -> tuple[int, int, int, int]:
        """Boîte arrondie en entiers, pour les fonctions de dessin OpenCV."""
        x1, y1, x2, y2 = self.xyxy
        return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    def distance_to(self, other: "Detection") -> float:
        """Distance euclidienne entre les centres de deux détections, en pixels.

        Utilisée par la règle « objet abandonné » pour chercher un propriétaire
        présumé à proximité de l'objet.
        """
        (ax, ay), (bx, by) = self.center, other.center
        return math.hypot(ax - bx, ay - by)

    def label(self, *, show_confidence: bool = True, show_track_id: bool = True) -> str:
        """Étiquette lisible pour l'annotation vidéo.

        Args:
            show_confidence: Inclure le score de confiance.
            show_track_id: Inclure l'identifiant de suivi s'il existe.

        Returns:
            Une chaîne du type ``"#12 person 0.87"``.
        """
        parts: list[str] = []
        if show_track_id and self.track_id is not None:
            parts.append(f"#{self.track_id}")
        parts.append(self.class_name)
        if show_confidence:
            parts.append(f"{self.confidence:.2f}")
        return " ".join(parts)


def to_supervision(detections: Sequence[Detection]):
    """Convertit une liste de `Detection` en `supervision.Detections`.

    Point de passage unique vers la librairie `supervision` (Roboflow), utilisée
    pour les tests d'appartenance aux polygones (`zones.py`) et les annotateurs
    (`app.py`). Concentrer la conversion ici évite de disperser la dépendance
    dans tout le code : si `supervision` change d'API, une seule fonction est à
    corriger.

    Args:
        detections: Détections à convertir.

    Returns:
        Un objet `supervision.Detections` (vide si la liste d'entrée est vide).

    Raises:
        ImportError: Si `supervision` n'est pas installé.
    """
    import supervision as sv  # import local : dépendance optionnelle au détecteur

    if not detections:
        return sv.Detections.empty()

    xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
    confidence = np.array([d.confidence for d in detections], dtype=np.float32)
    class_id = np.array([d.class_id for d in detections], dtype=int)

    tracker_id = None
    if all(d.track_id is not None for d in detections):
        tracker_id = np.array([d.track_id for d in detections], dtype=int)

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
        data={"class_name": np.array([d.class_name for d in detections])},
    )


def count_by_class(detections: Iterable[Detection]) -> dict[str, int]:
    """Compte les détections par nom de classe.

    Reprend la fonctionnalité de comptage du prototype d'origine, mais sur des
    objets **suivis** : passer `tracker.active()` plutôt qu'une liste de
    détections brutes donne un comptage sans doublon inter-frames.

    Args:
        detections: Détections à agréger.

    Returns:
        Un dictionnaire `{nom_de_classe: nombre}`.
    """
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
    return counts
