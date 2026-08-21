"""Gestion des zones polygonales — ÉTAPE 4.

Responsabilité unique : répondre aux questions que le moteur de règles pose sur
la géométrie — « cet objet est-il dans cette zone ? » et « cette zone
accepte-t-elle ce type d'incident ? ». Le module ne lève aucun incident et ne
connaît pas les rapports.

Le second point mérite d'être justifié : faire répondre `zones.py` sur les
**types** d'incidents pourrait passer pour une fuite de logique métier. Ce n'en
est pas une. La zone ne décide pas qu'un incident a lieu, elle décrit ce qu'elle
surveille — c'est une propriété du périmètre, au même titre que son polygone. Le
moteur reste seul à évaluer les conditions et à produire des `Event`.

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
import math
import unicodedata
from typing import Iterable, Sequence

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


def _covers_frame(polygon: Sequence[Sequence[float]], tolerance: float = 0.995) -> bool:
    """Indique si un polygone normalisé couvre (presque) tout le champ.

    Pourquoi cette question se pose
    --------------------------------
    La bande d'incertitude de la phase 3 suppose qu'une frontière **sépare deux
    espaces réels** : dedans et dehors. Le bord de l'image ne sépare rien — un
    objet ne peut pas en sortir latéralement, il disparaît. Appliquer une marge à
    une zone qui épouse le cadre créerait un anneau aveugle tout autour de
    l'image : une personne dont les pieds touchent le bas du cadre resterait
    éternellement « en cours d'entrée ».

    L'aire est calculée par la formule du lacet (Gauss), en valeur absolue pour
    accepter les deux sens de tracé.

    Args:
        polygon: Sommets normalisés dans [0, 1].
        tolerance: Fraction du champ à partir de laquelle la zone est réputée
            plein cadre.

    Returns:
        True si le polygone couvre au moins `tolerance` de l'image.
    """
    if len(polygon) < 3:
        return False

    aire = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        aire += x1 * y2 - x2 * y1
    return abs(aire) / 2.0 >= tolerance


class ZoneManager:
    """Convertit les zones de la configuration en zones pixel et teste l'appartenance.

    Les polygones de `config.ZONES` sont exprimés en coordonnées normalisées
    (0-1). Ils ne peuvent donc être convertis en pixels qu'une fois la résolution
    de la vidéo connue, d'où la méthode `initialize()` séparée du constructeur.
    """

    def __init__(self, zones: Sequence[config.SurveillanceZone] | None = None) -> None:
        """Prépare le gestionnaire à partir de la configuration.

        Args:
            zones: Zones à surveiller. `None` = `config.ZONES`.

        Raises:
            ZoneConfigurationError: Si une zone est invalide.
        """
        selected = config.ZONES if zones is None else tuple(zones)
        self._validate(selected)

        self._zones: tuple[config.SurveillanceZone, ...] = selected
        self._by_name: dict[str, config.SurveillanceZone] = {zone.name: zone for zone in selected}

        # Zones dont la frontière est celle de l'image. Voir `_covers_frame`.
        self._borderless: frozenset[str] = frozenset(
            zone.name for zone in selected if _covers_frame(zone.polygon)
        )

        # Résolution inconnue tant qu'aucune frame n'a été vue.
        self._frame_size: tuple[int, int] | None = None
        self._polygons: dict[str, np.ndarray] = {}

    def _validate(self, zones: Sequence[config.SurveillanceZone]) -> None:
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

            # Les surcharges de seuils sont éprouvées ici, au démarrage, contre
            # toutes les règles qui s'appliqueront dans cette zone. Une zone qui
            # exigerait 120 s de présence tout en autorisant un redéclenchement
            # toutes les 60 s produirait un rapport par frame — et on ne le
            # découvrirait qu'en pleine analyse.
            for rule in config.EVENT_RULES:
                if not zone.handles(rule.event_type):
                    continue
                try:
                    zone.rule_for(rule)
                except ValueError as exc:
                    raise ZoneConfigurationError(str(exc)) from exc

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

    def zones_for(self, detection: Detection) -> dict[str, float]:
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
        echelle = max(1.0, float(detection.height))

        marges: dict[str, float] = {}
        for name, polygon in self._polygons.items():
            # `measureDist=True` : on veut la distance **signée** au bord, en
            # pixels — positive dedans, négative dehors. C'est le même lancer de
            # rayon qu'avec `measureDist=False`, avec la distance en prime.
            distance = cv2.pointPolygonTest(polygon, point, True)

            if name in self._borderless:
                # Le bord de l'image n'est pas un mur : un objet ne peut pas en
                # sortir latéralement sans disparaître du champ. Exiger une marge
                # créerait un angle mort tout autour de l'image — une personne
                # dont les pieds touchent le bas du cadre ne serait jamais
                # « entrée ». On rend donc un verdict franc.
                marges[name] = math.inf if distance >= 0 else -math.inf
            else:
                marges[name] = distance / echelle

        return marges

    def zones_for_all(self, detections: Sequence[Detection]) -> dict[int, dict[str, float]]:
        """Localise toutes les détections d'une frame en un appel.

        Args:
            detections: Détections de la frame.

        Returns:
            Un dictionnaire `{index_dans_la_liste: {zone: marge_signée}}`. Les
            détections franchement hors de toute zone sont **absentes** du
            dictionnaire : l'appelant utilise `.get(i, {})`.

        Raises:
            RuntimeError: Si `initialize()` n'a pas été appelé.
        """
        located: dict[int, dict[str, float]] = {}
        for index, detection in enumerate(detections):
            marges = self.zones_for(detection)
            if any(marge >= 0 for marge in marges.values()):
                located[index] = marges
        return located

    def restricted_zone_names(self) -> list[str]:
        """Noms des zones où la seule présence constitue une infraction.

        Dérivé du **type** de zone, plus d'un drapeau indépendant : un booléen
        `restricted` à côté d'un `zone_type` finirait par le contredire.
        """
        return [zone.name for zone in self._zones if zone.restricted]

    def zone(self, name: str) -> config.SurveillanceZone | None:
        """Zone portant ce nom.

        Args:
            name: Nom recherché.

        Returns:
            La zone, ou `None` si elle est inconnue.
        """
        return self._by_name.get(name)

    def zones_handling(self, event_type: config.EventType) -> list[str]:
        """Noms des zones qui acceptent de lever ce type d'incident.

        C'est le **seul** mécanisme de ciblage du projet : une règle ne déclare
        plus les zones où elle s'applique, c'est la zone qui déclare les règles
        qu'elle accepte. Deux mécanismes concurrents auraient exigé une règle de
        résolution de conflit — donc une explication de plus dans le README, et
        une source d'erreur de plus dans la configuration.

        Args:
            event_type: Type d'incident envisagé.

        Returns:
            Les noms des zones concernées, dans l'ordre de la configuration.
        """
        return [zone.name for zone in self._zones if zone.handles(event_type)]

    def rule_for(self, zone_name: str, rule: config.EventRule) -> config.EventRule:
        """Règle telle qu'elle s'applique dans une zone donnée.

        Args:
            zone_name: Zone concernée.
            rule: Règle globale issue de `config.EVENT_RULES`.

        Returns:
            La règle ajustée aux surcharges de la zone, ou la règle d'origine si
            la zone ne surcharge rien — ou si elle est inconnue, auquel cas la
            valeur globale est le repli le plus sûr.
        """
        zone = self._by_name.get(zone_name)
        return rule if zone is None else zone.rule_for(rule)

    def shortest_cooldown(self, rule: config.EventRule) -> float:
        """Plus court délai de garde applicable à cette règle, toutes zones confondues.

        Sert de **pré-filtre** : le moteur écarte très tôt les objets qui ne
        peuvent redéclencher nulle part, sans avoir à évaluer le prédicat. Prendre
        le plus court est le seul choix correct — retenir le délai global écarterait
        à tort un objet qu'une zone plus permissive autoriserait à redéclencher.
        Le délai exact est ensuite revérifié zone par zone au moment de publier.

        Args:
            rule: Règle globale.

        Returns:
            Le délai le plus court, en secondes.
        """
        delais = [
            zone.rule_for(rule).cooldown_s
            for zone in self._zones
            if zone.handles(rule.event_type)
        ]
        return min([rule.cooldown_s, *delais])

    def color_for(self, name: str) -> tuple[int, int, int] | None:
        """Couleur BGR déclarée par une zone.

        Exposée pour que l'interface signale un objet en infraction avec la
        couleur de **sa** zone plutôt qu'avec un rouge codé en dur : deux zones
        de gravités différentes restent alors distinguables sur la vidéo
        annotée.

        Args:
            name: Nom de la zone.

        Returns:
            La couleur, ou `None` si la zone est inconnue.
        """
        for zone in self._zones:
            if zone.name == name:
                return zone.color
        return None

    def alert_color_for(self, zone_names: Iterable[str]) -> tuple[int, int, int] | None:
        """Couleur d'alerte d'un objet occupant plusieurs zones.

        Le tri alphabétique rend le choix reproductible : sans lui, l'ordre
        d'itération d'un `set` ferait changer la couleur d'une exécution à
        l'autre, ce qu'un opérateur interpréterait comme un changement de
        situation.

        Args:
            zone_names: Zones occupées par l'objet.

        Returns:
            La couleur de la première zone restreinte occupée, ou `None` si
            l'objet n'en occupe aucune.
        """
        restreintes = set(self.restricted_zone_names())
        for nom in sorted(set(zone_names) & restreintes):
            return self.color_for(nom)
        return None

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

            # Le type est plus informatif que « restreinte » : un opérateur
            # doit pouvoir lire sur l'image ce que la zone attend.
            label = _ascii_label(f"{name} [{zone.zone_type.value.upper()}]")
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
