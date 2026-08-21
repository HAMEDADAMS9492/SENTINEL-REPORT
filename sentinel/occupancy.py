"""Occupation des zones — combien d'objets distincts, depuis combien de temps.

Rôle : compter, et seulement compter. Ce module ne lève aucun incident, ne
connaît ni les règles ni les rapports.

Pourquoi une classe distincte de `ZoneManager`
-----------------------------------------------
`ZoneManager` répond à des questions de **géométrie** : où est ce point, quelle
zone accepte quel type d'incident. Ses réponses ne dépendent que de la frame
courante — il est sans mémoire, et c'est ce qui le rend simple à tester et à
raisonner.

L'occupation est de nature opposée : « sept personnes dans le hall **depuis
douze secondes** » est un fait qui ne peut se constater que dans la durée. Lui
donner un objet à part garde `ZoneManager` géométrique et rend cette mémoire
testable sans polygone, sans image et sans OpenCV.

Compter des identités, pas des détections
------------------------------------------
Le comptage s'appuie sur `track_id`. Compter des détections donnerait un nombre
qui bat au rythme du détecteur : une personne perdue puis retrouvée compterait
deux fois, et un objet momentanément occulté ferait chuter le total. C'est la
même raison qui a rendu le tracker indispensable au chronométrage (§ 3.4 du
README) : sans identité stable, aucune grandeur mesurée dans le temps n'est
fiable.

Ce que ça débloque
-------------------
Deux usages, un seul mécanisme :

* la règle **surdensité** — « plus de N personnes en zone X pendant T secondes » ;
* les transitions de la chronologie — « hall : 2 → 7 personnes », un fait
  marquant qu'aucune règle par objet ne saurait formuler.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import config
from sentinel.tracker import TrackedObject

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OccupancyChange:
    """Une variation du nombre d'occupants d'une zone.

    Attributes:
        zone_name: Zone concernée.
        class_name: Classe comptée.
        before: Nombre d'occupants avant la variation.
        after: Nombre d'occupants après.
        video_time: Instant de la variation, en temps vidéo.
    """

    zone_name: str
    class_name: str
    before: int
    after: int
    video_time: float

    @property
    def label(self) -> str:
        """Description lisible, du type « hall : 2 → 7 personnes »."""
        return f"{self.zone_name} : {self.before} → {self.after} {self.class_name}"

    @property
    def is_increase(self) -> bool:
        """True si la zone s'est remplie."""
        return self.after > self.before


class ZoneOccupancy:
    """Mémoire du nombre d'occupants par zone et par classe.

        >>> occupation = ZoneOccupancy()
        >>> occupation.update(tracker.active(), video_time=12.0)
        >>> occupation.count("Hall", "person")
        7
        >>> occupation.duration_at_least("Hall", "person", 5, video_time=12.0)
        4.5
    """

    def __init__(self, settings: config.OccupancyConfig | None = None) -> None:
        """Prépare un compteur vide.

        Args:
            settings: Réglages de comptage. `None` = `config.OCCUPANCY`.
        """
        self.settings = settings or config.OCCUPANCY
        self._counts: dict[str, Counter[str]] = {}
        self._since: dict[tuple[str, str, int], float] = {}
        self._changes: list[OccupancyChange] = []
        self._video_time: float = 0.0

    # -- Observation ----------------------------------------------------------

    def update(
        self, objects: Sequence[TrackedObject], video_time: float
    ) -> list[OccupancyChange]:
        """Recompte les occupants de chaque zone et rend les variations.

        Args:
            objects: Objets **retenus** par le tracker (`tracker.active()`). Ce
                sont bien les objets retenus qu'il faut passer, pas ceux vus sur
                la frame : la rétention absorbe déjà les occlusions courtes, et
                sans elle un passage derrière un poteau ferait chuter le
                comptage puis le faire remonter.
            video_time: Temps vidéo de la frame.

        Returns:
            Les variations constatées depuis la frame précédente. Une liste vide
            est le cas courant — c'est tout l'intérêt de ne signaler que ce qui
            change.
        """
        self._video_time = video_time
        nouveaux: dict[str, Counter[str]] = {}
        for obj in objects:
            for zone_name in obj.zones:
                nouveaux.setdefault(zone_name, Counter())[obj.class_name] += 1

        variations = self._diff(nouveaux, video_time)
        self._counts = nouveaux
        self._track_thresholds(video_time)
        self._changes.extend(variations)
        return variations

    def _diff(
        self, nouveaux: dict[str, Counter[str]], video_time: float
    ) -> list[OccupancyChange]:
        """Compare le nouvel état à l'ancien.

        Args:
            nouveaux: Comptages de la frame courante.
            video_time: Temps vidéo de la frame.

        Returns:
            Les variations, triées par zone puis par classe pour rester
            reproductibles d'une exécution à l'autre.
        """
        zones = set(nouveaux) | set(self._counts)
        variations: list[OccupancyChange] = []

        for zone_name in sorted(zones):
            avant = self._counts.get(zone_name, Counter())
            apres = nouveaux.get(zone_name, Counter())
            for class_name in sorted(set(avant) | set(apres)):
                ancien, courant = avant.get(class_name, 0), apres.get(class_name, 0)
                if ancien == courant:
                    continue
                if abs(courant - ancien) < self.settings.min_change:
                    # Une variation d'une unité dans une foule n'apprend rien et
                    # noierait la chronologie. Le seuil est configurable.
                    continue
                variations.append(
                    OccupancyChange(zone_name, class_name, ancien, courant, video_time)
                )
        return variations

    def _track_thresholds(self, video_time: float) -> None:
        """Met à jour l'instant de franchissement de chaque seuil surveillé.

        Un seuil est « tenu » tant que le comptage ne redescend pas en dessous.
        Mémoriser l'instant du franchissement plutôt qu'un compteur de frames
        rend la durée indépendante de la cadence d'analyse — c'est le même
        principe que le chronomètre de zone du tracker.

        Args:
            video_time: Temps vidéo de la frame.
        """
        vivants: set[tuple[str, str, int]] = set()
        for zone_name, comptes in self._counts.items():
            for class_name, nombre in comptes.items():
                for seuil in range(1, nombre + 1):
                    cle = (zone_name, class_name, seuil)
                    vivants.add(cle)
                    self._since.setdefault(cle, video_time)

        for cle in list(self._since):
            if cle not in vivants:
                del self._since[cle]

    # -- Consultation ---------------------------------------------------------

    def count(self, zone_name: str, class_name: str | None = None) -> int:
        """Nombre d'occupants d'une zone.

        Args:
            zone_name: Zone interrogée.
            class_name: Classe comptée. `None` = toutes classes confondues.

        Returns:
            Le nombre d'objets distincts, ou 0 si la zone est vide ou inconnue.
        """
        comptes = self._counts.get(zone_name)
        if comptes is None:
            return 0
        return sum(comptes.values()) if class_name is None else comptes.get(class_name, 0)

    def duration_at_least(
        self, zone_name: str, class_name: str, threshold: int, video_time: float
    ) -> float:
        """Depuis combien de temps la zone tient au moins ce nombre d'occupants.

        Args:
            zone_name: Zone interrogée.
            class_name: Classe comptée.
            threshold: Nombre d'occupants à atteindre.
            video_time: Temps vidéo courant.

        Returns:
            La durée en secondes, ou `0.0` si le seuil n'est pas atteint.
        """
        depuis = self._since.get((zone_name, class_name, threshold))
        return 0.0 if depuis is None else max(0.0, video_time - depuis)

    def counts_for(self, zone_name: str) -> dict[str, int]:
        """Comptage par classe d'une zone.

        Args:
            zone_name: Zone interrogée.

        Returns:
            Un dictionnaire `{classe: nombre}`, éventuellement vide.
        """
        return dict(self._counts.get(zone_name, Counter()))

    @property
    def occupied_zones(self) -> list[str]:
        """Zones comptant au moins un occupant, par ordre alphabétique."""
        return sorted(nom for nom, comptes in self._counts.items() if sum(comptes.values()))

    def changes(self, *, notable_only: bool = True) -> list[OccupancyChange]:
        """Variations accumulées depuis le début de la session.

        Args:
            notable_only: Ne rendre que les variations atteignant
                `config.OCCUPANCY.notable_from` occupants. Une zone qui passe de
                0 à 1 personne n'est pas un fait marquant ; une zone qui passe de
                2 à 7 l'est.

        Returns:
            Les variations, dans l'ordre chronologique.
        """
        if not notable_only:
            return list(self._changes)
        plancher = self.settings.notable_from
        return [
            variation
            for variation in self._changes
            if max(variation.before, variation.after) >= plancher
        ]

    def reset(self) -> None:
        """Vide le compteur. À appeler entre deux sessions."""
        self._counts.clear()
        self._since.clear()
        self._changes.clear()
        self._video_time = 0.0

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        occupees = len(self.occupied_zones)
        return f"ZoneOccupancy({occupees} zone(s) occupée(s), {len(self._changes)} variation(s))"


def summarise(occupancy: ZoneOccupancy, zone_names: Iterable[str]) -> str:
    """Résume l'occupation de plusieurs zones en une ligne.

    Args:
        occupancy: Compteur à résumer.
        zone_names: Zones à inclure.

    Returns:
        Un texte du type « Hall : 3 person · Quai : 1 person », ou « — » si
        aucune zone n'est occupée.
    """
    morceaux = []
    for zone_name in zone_names:
        comptes = occupancy.counts_for(zone_name)
        if not comptes:
            continue
        detail = ", ".join(f"{nombre} {classe}" for classe, nombre in sorted(comptes.items()))
        morceaux.append(f"{zone_name} : {detail}")
    return " · ".join(morceaux) if morceaux else "—"
