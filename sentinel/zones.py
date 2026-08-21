"""Gestion des zones polygonales — ÉTAPE 4.

Responsabilité unique : répondre à la question « cet objet est-il dans cette
zone ? ». Le module ne connaît ni les événements, ni les rapports.

Choix de la librairie
---------------------
Le squelette prévoyait `supervision.PolygonZone` (Roboflow). Mesure faite, cette
dépendance tire ~30 Mo de paquets — dont PyAV, un décodeur vidéo complet — pour
un service que `cv2.pointPolygonTest` rend déjà : c'est **le même algorithme du
lancer de rayon**, dans une API stable depuis dix ans, et OpenCV est de toute
façon une dépendance obligatoire du projet (lecture vidéo, dessin).

L'argument de la vectorisation ne tient pas à cette échelle : avec au plus
`config.MODEL.max_detections` objets et deux ou trois zones, on compte quelques
dizaines d'appels par frame, soit quelques microsecondes — sans commune mesure
avec les ~140 ms d'inférence YOLO qui les précèdent.

La dépendance a donc été retirée du projet, y compris la fonction de
conversion `to_supervision()` qui la gardait en vie sans appelant : un
argumentaire qui refuse une librairie tout en l'installant ne tient pas.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Sequence

import cv2
import numpy as np

import config
from sentinel.detection import Detection
from sentinel.exceptions import ZoneConfigurationError

logger = logging.getLogger(__name__)


def _ascii_label(text: str) -> str:
    """Translittère un texte pour l'affichage OpenCV.

    Les polices Hershey de `cv2.putText` ne couvrent pas les caractères
    accentués : « Zone réservée » s'y afficherait « Zone r?serv?e ». On
    translittère donc en ASCII **au moment du dessin uniquement** — les noms de
    zones restent intacts dans la configuration, les rapports et les événements.

    Args:
        text: Texte d'origine, éventuellement accentué.

    Returns:
        Le texte sans diacritiques.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return decomposed.encode("ascii", "ignore").decode("ascii")


class ZoneManager:
    """Convertit les zones de la configuration en zones pixel et teste l'appartenance.

    Les polygones de `config.ZONES` sont exprimés en coordonnées normalisées
    (0-1). Ils ne peuvent donc être convertis en pixels qu'une fois la résolution
    de la vidéo connue, d'où la méthode `initialize()` séparée du constructeur.
    """

    def __init__(self, zones: Sequence[config.ZoneConfig] | None = None) -> None:
        """Prépare le gestionnaire à partir de la configuration.

        Args:
            zones: Zones à surveiller. `None` = `config.ZONES`.

        Raises:
            ZoneConfigurationError: Si une zone est invalide.
        """
        selected = config.ZONES if zones is None else tuple(zones)
        self._validate(selected)

        self._zones: tuple[config.ZoneConfig, ...] = selected
        self._by_name: dict[str, config.ZoneConfig] = {zone.name: zone for zone in selected}

        # Résolution inconnue tant qu'aucune frame n'a été vue.
        self._frame_size: tuple[int, int] | None = None
        self._polygons: dict[str, np.ndarray] = {}

    def _validate(self, zones: Sequence[config.ZoneConfig]) -> None:
        """Vérifie la cohérence des zones configurées.

        La validation a lieu **à la construction**, donc au démarrage de
        l'application : une faute de frappe dans `config.ZONES` doit produire un
        message clair tout de suite, et non une zone silencieusement inopérante
        découverte au milieu de l'analyse d'une vidéo.

        Args:
            zones: Zones à valider.

        Raises:
            ZoneConfigurationError: Si un polygone a moins de 3 sommets, si une
                coordonnée sort de [0, 1], ou si deux zones portent le même nom.
        """
        if not zones:
            raise ZoneConfigurationError(
                "Aucune zone configurée : renseignez au moins une entrée dans config.ZONES."
            )

        seen: set[str] = set()
        for zone in zones:
            if zone.name in seen:
                raise ZoneConfigurationError(
                    f"Deux zones portent le nom '{zone.name}'. Les noms servent de "
                    "clés dans les rapports et doivent être uniques."
                )
            seen.add(zone.name)

            if len(zone.polygon) < 3:
                raise ZoneConfigurationError(
                    f"Zone '{zone.name}' : {len(zone.polygon)} sommet(s), il en faut "
                    "au moins 3 pour délimiter une surface."
                )

            for index, point in enumerate(zone.polygon):
                if len(point) != 2:
                    raise ZoneConfigurationError(
                        f"Zone '{zone.name}', sommet {index} : {point!r} n'est pas "
                        "un couple (x, y)."
                    )
                x, y = point
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ZoneConfigurationError(
                        f"Zone '{zone.name}', sommet {index} = ({x}, {y}) : les "
                        "coordonnées sont normalisées et doivent tenir dans [0, 1]."
                    )

    def initialize(self, frame_shape: tuple[int, ...]) -> None:
        """Convertit les polygones normalisés en pixels pour une résolution donnée.

        À appeler une fois, avec la première frame du flux. Idempotent : rappeler
        avec la même résolution ne change rien ; avec une résolution différente,
        les zones sont reconstruites.

        Args:
            frame_shape: Forme de la frame `(hauteur, largeur, canaux)`.

        Raises:
            ZoneConfigurationError: Si la forme fournie est inexploitable.
        """
        if len(frame_shape) < 2:
            raise ZoneConfigurationError(
                f"Forme de frame inattendue : {frame_shape}, attendu (hauteur, largeur, ...)."
            )

        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height <= 0 or width <= 0:
            raise ZoneConfigurationError(
                f"Dimensions de frame invalides : {width}x{height} pixels."
            )

        if self._frame_size == (height, width):
            return  # déjà calculé pour cette résolution

        self._frame_size = (height, width)
        self._polygons = {
            zone.name: np.array(
                [(round(x * width), round(y * height)) for x, y in zone.polygon],
                dtype=np.int32,
            )
            for zone in self._zones
        }

        logger.info(
            "Zones converties pour une image %dx%d : %s",
            width,
            height,
            ", ".join(self._polygons),
        )

    def _anchor_point(self, detection: Detection) -> tuple[float, float]:
        """Point de la boîte testé contre les polygones.

        Args:
            detection: Détection à localiser.

        Returns:
            Le point d'appui au sol ou le centre, selon `config.GEOMETRY.anchor`.
        """
        anchor = config.GEOMETRY.anchor
        if anchor == "bottom_center":
            return detection.anchor
        if anchor == "center":
            return detection.center

        logger.warning(
            "Ancre inconnue dans la configuration : '%s'. Repli sur 'bottom_center'.",
            anchor,
        )
        return detection.anchor

    def zones_for(self, detection: Detection) -> list[str]:
        """Retourne les noms des zones contenant une détection.

        Le point testé est l'ancre configurée dans `config.GEOMETRY.anchor`
        (`bottom_center` par défaut : le point d'appui au sol).

        Args:
            detection: Détection à localiser.

        Returns:
            Les noms de zones contenant l'objet (liste vide si aucune).

        Raises:
            RuntimeError: Si `initialize()` n'a pas été appelé.
        """
        if not self.is_initialized:
            raise RuntimeError(
                "ZoneManager.initialize(frame.shape) doit être appelé avec la "
                "première frame avant tout test d'appartenance."
            )

        point = self._anchor_point(detection)
        # `measureDist=False` : on ne veut que le signe. Le retour vaut +1
        # (dedans), 0 (exactement sur le bord) ou -1 (dehors) ; un objet sur la
        # frontière est compté dedans, choix conservateur pour un système
        # d'alerte.
        return [
            name
            for name, polygon in self._polygons.items()
            if cv2.pointPolygonTest(polygon, point, False) >= 0
        ]

    def zones_for_all(self, detections: Sequence[Detection]) -> dict[int, list[str]]:
        """Localise toutes les détections d'une frame en un appel.

        Args:
            detections: Détections de la frame.

        Returns:
            Un dictionnaire `{index_dans_la_liste: [noms_de_zones]}`. Les
            détections situées hors de toute zone sont **absentes** du
            dictionnaire : l'appelant utilise `.get(i, [])`.

        Raises:
            RuntimeError: Si `initialize()` n'a pas été appelé.
        """
        located: dict[int, list[str]] = {}
        for index, detection in enumerate(detections):
            names = self.zones_for(detection)
            if names:
                located[index] = names
        return located

    def restricted_zone_names(self) -> list[str]:
        """Noms des zones marquées `restricted=True` dans la configuration."""
        return [zone.name for zone in self._zones if zone.restricted]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Dessine les zones sur une copie de la frame.

        Args:
            frame: Image BGR d'origine (non modifiée).

        Returns:
            Une nouvelle image avec les polygones remplis et étiquetés.

        Raises:
            RuntimeError: Si `initialize()` n'a pas été appelé.
        """
        if not self.is_initialized:
            raise RuntimeError(
                "ZoneManager.initialize(frame.shape) doit être appelé avant draw()."
            )

        annotated = frame.copy()

        # Une zone couvrant tout le champ (`draw=False`) est exclue du tracé :
        # la remplir teinterait l'image entière et masquerait précisément ce que
        # l'opérateur doit voir.
        drawable = {
            name: polygon
            for name, polygon in self._polygons.items()
            if self._by_name[name].draw
        }
        if not drawable:
            return annotated

        # Le remplissage est peint sur un calque séparé puis fondu : dessiner
        # directement sur l'image masquerait ce qui se trouve dans la zone.
        overlay = annotated.copy()
        for name, polygon in drawable.items():
            cv2.fillPoly(overlay, [polygon], self._by_name[name].color)

        opacity = config.UI.zone_opacity
        cv2.addWeighted(overlay, opacity, annotated, 1.0 - opacity, 0.0, dst=annotated)

        for name, polygon in drawable.items():
            zone = self._by_name[name]
            cv2.polylines(
                annotated,
                [polygon],
                isClosed=True,
                color=zone.color,
                thickness=config.UI.box_thickness,
            )

            label = _ascii_label(name if not zone.restricted else f"{name} [RESTREINTE]")
            anchor_x, anchor_y = polygon[polygon[:, 1].argmin()]
            cv2.putText(
                annotated,
                label,
                (int(anchor_x), max(15, int(anchor_y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                zone.color,
                config.UI.box_thickness,
                cv2.LINE_AA,
            )

        return annotated

    @property
    def is_initialized(self) -> bool:
        """True si les zones ont été converties en pixels."""
        return self._frame_size is not None

    @property
    def names(self) -> list[str]:
        """Noms de toutes les zones configurées."""
        return [zone.name for zone in self._zones]

    @property
    def frame_size(self) -> tuple[int, int] | None:
        """Résolution `(hauteur, largeur)` pour laquelle les zones sont calculées."""
        return self._frame_size

    def polygon(self, name: str) -> np.ndarray | None:
        """Polygone en pixels d'une zone, ou `None` si elle est inconnue.

        Args:
            name: Nom de la zone.

        Returns:
            Un tableau `(N, 2)` d'entiers, ou `None`.
        """
        return self._polygons.get(name)

    def __len__(self) -> int:
        """Nombre de zones configurées."""
        return len(self._zones)

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        state = f"{self._frame_size[1]}x{self._frame_size[0]}" if self._frame_size else "non initialisé"
        return f"ZoneManager(zones={len(self._zones)}, resolution={state})"
