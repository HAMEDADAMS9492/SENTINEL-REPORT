"""Franchissement de ligne — compter des passages, et dans quel sens.

Rôle : dire qu'un objet suivi a traversé un segment, et de quel côté vers quel
côté. Le module ne lève aucun incident et ne connaît ni les règles ni les
rapports.

Pourquoi une ligne n'est pas une zone
--------------------------------------
Une ligne a deux points, pas de surface, pas de durée de séjour. En faire un
`zone_type` de `SurveillanceZone` aurait produit une structure dont la moitié
des champs est inapplicable selon la valeur d'un autre champ — exactement le
défaut que le typage des zones cherchait à supprimer. `CrossingLine` est donc une
dataclass séparée, qu'une zone de comptage référence par son nom.

La différence est aussi de nature : une zone répond à « **où** est cet objet ? »,
une ligne à « **qu'a-t-il fait** ? ». La première est un état, la seconde un
événement.

L'algorithme, et pourquoi aucune librairie
--------------------------------------------
Le côté d'un point P par rapport à une ligne AB est donné par le signe du produit
vectoriel `(B−A) × (P−A)`. Positif d'un côté, négatif de l'autre, nul sur la
ligne. Un objet a franchi la ligne entre deux frames si le signe **a changé** ;
le nouveau signe donne le sens.

Un changement de signe ne suffit pourtant pas : il vaut aussi pour un objet qui
passe très loin, au-delà des extrémités du segment — le produit vectoriel décrit
une droite infinie, pas un segment. On vérifie donc que les deux segments (la
ligne, et le déplacement de l'objet) se coupent réellement, par le test
d'orientation classique : quatre produits vectoriels, aucune division, aucun cas
dégénéré à traiter séparément.

Le tout tient en une vingtaine de lignes d'arithmétique. `supervision.LineZone`
rendrait le même service au prix d'une dépendance de ~30 Mo, refusée ailleurs
dans ce projet pour la même raison (voir `zones.py`) : on ne paie pas 30 Mo pour
quatre soustractions et deux multiplications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import config
from sentinel.tracker import TrackedObject

logger = logging.getLogger(__name__)

Point = tuple[float, float]


def side_of(start: Point, end: Point, point: Point) -> float:
    """Produit vectoriel indiquant de quel côté d'une droite tombe un point.

    Args:
        start: Origine de la droite.
        end: Extrémité de la droite.
        point: Point à situer.

    Returns:
        Une valeur positive d'un côté, négative de l'autre, nulle sur la droite.
        La magnitude est proportionnelle à la distance, mais seul le **signe**
        est exploité : une distance en pixels dépendrait de la profondeur.
    """
    (ax, ay), (bx, by) = start, end
    px, py = point
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """Indique si deux segments se coupent.

    Le changement de signe du produit vectoriel décrit une **droite infinie**.
    Sans ce second test, un objet passant à dix mètres au-delà de l'extrémité de
    la ligne serait compté comme l'ayant franchie.

    Args:
        a1: Origine du premier segment.
        a2: Extrémité du premier segment.
        b1: Origine du second segment.
        b2: Extrémité du second segment.

    Returns:
        True si les segments se croisent franchement. Un contact exactement à une
        extrémité (produit nul) est rejeté : c'est un frôlement, pas un
        franchissement, et le compter ferait osciller le total.
    """
    d1 = side_of(b1, b2, a1)
    d2 = side_of(b1, b2, a2)
    d3 = side_of(a1, a2, b1)
    d4 = side_of(a1, a2, b2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0


@dataclass(frozen=True, slots=True)
class LineCrossing:
    """Un franchissement constaté.

    Attributes:
        line_name: Ligne traversée.
        track_id: Identifiant persistant de l'objet.
        class_name: Classe de l'objet.
        direction: Libellé du sens, tel que la configuration le nomme —
            « entrée » et « sortie » par défaut, mais « montée » / « descente »
            conviendraient aussi bien à un escalier.
        video_time: Instant du franchissement, en temps vidéo.
    """

    line_name: str
    track_id: int
    class_name: str
    direction: str
    video_time: float

    @property
    def label(self) -> str:
        """Description lisible, du type « Porte : person #4 — entrée »."""
        return f"{self.line_name} : {self.class_name} #{self.track_id} — {self.direction}"


class LineCounter:
    """Compte les franchissements de lignes, sens par sens.

    Les lignes sont décrites en coordonnées normalisées, comme les zones : la
    même configuration vaut pour toutes les résolutions. Elles ne peuvent donc
    être converties en pixels qu'une fois la première frame connue, d'où
    `initialize()` séparé du constructeur.

        >>> compteur = LineCounter()
        >>> compteur.initialize(frame.shape)
        >>> compteur.update(tracker.active(), video_time=12.0)
        [LineCrossing(line_name='Porte', ...)]
        >>> compteur.net("Porte")
        3
    """

    def __init__(self, lines: Sequence[config.CrossingLine] | None = None) -> None:
        """Prépare le compteur.

        Args:
            lines: Lignes à surveiller. `None` = `config.CROSSING_LINES`.
        """
        self._lines = tuple(config.CROSSING_LINES if lines is None else lines)
        self._by_name = {ligne.name: ligne for ligne in self._lines}

        self._pixels: dict[str, tuple[Point, Point]] = {}
        self._frame_size: tuple[int, int] | None = None
        self._last: dict[int, Point] = {}
        self._crossings: list[LineCrossing] = []

    # -- Cycle de vie ---------------------------------------------------------

    def initialize(self, frame_shape: tuple[int, ...]) -> None:
        """Convertit les lignes normalisées en pixels.

        Args:
            frame_shape: Forme de la frame, `(hauteur, largeur, ...)`.
        """
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if self._frame_size == (width, height):
            return

        self._frame_size = (width, height)
        self._pixels = {
            ligne.name: (
                (ligne.start[0] * width, ligne.start[1] * height),
                (ligne.end[0] * width, ligne.end[1] * height),
            )
            for ligne in self._lines
        }
        # Les positions mémorisées étaient en pixels de l'ancienne résolution :
        # les garder ferait constater des franchissements imaginaires au premier
        # redimensionnement.
        self._last.clear()
        logger.debug("%d ligne(s) converties pour %dx%d.", len(self._pixels), width, height)

    @property
    def is_initialized(self) -> bool:
        """True si les lignes ont été converties en pixels."""
        return self._frame_size is not None

    # -- Observation ----------------------------------------------------------

    def update(
        self, objects: Sequence[TrackedObject], video_time: float
    ) -> list[LineCrossing]:
        """Compare les positions à celles de la frame précédente.

        Args:
            objects: Objets **retenus** par le tracker. La rétention absorbe les
                occlusions courtes ; sans elle, un objet momentanément masqué
                réapparaîtrait de l'autre côté sans qu'aucun segment de
                déplacement ne soit examiné, et le franchissement serait manqué.
            video_time: Temps vidéo de la frame.

        Returns:
            Les franchissements constatés, triés par ligne puis par identifiant
            pour rester reproductibles.
        """
        if not self._pixels:
            return []

        constats: list[LineCrossing] = []
        vus: set[int] = set()

        for obj in objects:
            vus.add(obj.track_id)
            avant = self._last.get(obj.track_id)
            apres = obj.position
            self._last[obj.track_id] = apres
            if avant is None or avant == apres:
                continue

            for name, (debut, fin) in self._pixels.items():
                ligne = self._by_name[name]
                if ligne.classes and obj.class_name not in ligne.classes:
                    continue
                if not segments_intersect(avant, apres, debut, fin):
                    continue

                sens = (
                    ligne.positive_label
                    if side_of(debut, fin, apres) > 0
                    else ligne.negative_label
                )
                constats.append(
                    LineCrossing(name, obj.track_id, obj.class_name, sens, video_time)
                )

        # Les objets disparus libèrent leur position : sans cela, un identifiant
        # réattribué par ByteTrack hériterait de la dernière position d'un autre
        # objet et produirait un franchissement fantôme.
        for track_id in set(self._last) - vus:
            del self._last[track_id]

        constats.sort(key=lambda constat: (constat.line_name, constat.track_id))
        self._crossings.extend(constats)
        return constats

    # -- Consultation ---------------------------------------------------------

    def count(self, line_name: str, direction: str | None = None) -> int:
        """Nombre de franchissements d'une ligne.

        Args:
            line_name: Ligne interrogée.
            direction: Sens compté. `None` = les deux.

        Returns:
            Le nombre de passages.
        """
        return sum(
            1
            for passage in self._crossings
            if passage.line_name == line_name
            and (direction is None or passage.direction == direction)
        )

    def net(self, line_name: str) -> int:
        """Solde d'une ligne : passages dans le sens positif moins l'autre.

        C'est la mesure utile d'un comptage de flux — le nombre de personnes
        actuellement à l'intérieur, si la ligne est la seule porte.

        Args:
            line_name: Ligne interrogée.

        Returns:
            Le solde, éventuellement négatif.
        """
        ligne = self._by_name.get(line_name)
        if ligne is None:
            return 0
        return self.count(line_name, ligne.positive_label) - self.count(
            line_name, ligne.negative_label
        )

    def summary(self, line_name: str) -> str:
        """Résumé lisible d'une ligne, pour la ligne d'état et le rapport.

        Args:
            line_name: Ligne interrogée.

        Returns:
            Un texte du type « Porte : 12 entrée / 9 sortie (solde +3) ».
        """
        ligne = self._by_name.get(line_name)
        if ligne is None:
            return f"{line_name} : ligne inconnue"
        entrants = self.count(line_name, ligne.positive_label)
        sortants = self.count(line_name, ligne.negative_label)
        return (
            f"{line_name} : {entrants} {ligne.positive_label} / "
            f"{sortants} {ligne.negative_label} (solde {self.net(line_name):+d})"
        )

    @property
    def names(self) -> list[str]:
        """Noms des lignes surveillées."""
        return [ligne.name for ligne in self._lines]

    def crossings(self) -> list[LineCrossing]:
        """Tous les franchissements depuis le début de la session."""
        return list(self._crossings)

    def reset(self) -> None:
        """Vide le compteur. À appeler entre deux sessions."""
        self._last.clear()
        self._crossings.clear()

    def __len__(self) -> int:
        """Nombre total de franchissements constatés."""
        return len(self._crossings)

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        return f"LineCounter({len(self._lines)} ligne(s), {len(self._crossings)} passage(s))"
