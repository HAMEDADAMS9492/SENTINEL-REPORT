"""Chronologie de session — les faits marquants, tranche par tranche.

Rôle : observer le déroulement d'une analyse et n'en retenir que ce qui **change**.
C'est la matière de la partie 2 du rapport.

Pourquoi ce module existe plutôt qu'un calcul dans `report.py`
---------------------------------------------------------------
Décider qu'une personne « est entrée dans le champ » ou qu'un objet « reste
prioritaire » est une appréciation métier : elle demande de comparer l'état
présent à l'état précédent, ce que seule une observation continue permet. Un
générateur de rapport, lui, reçoit des faits déjà établis et les met en forme.
Confier la chronologie à `report.py` l'aurait obligé à reconstituer après coup
une histoire qu'il n'a pas vue — au mieux approximativement, au pire faussement.

Le flux reste donc unidirectionnel : `tracker` → `timeline` → `report`.

Ne dire que ce qui change
--------------------------
Une chronologie qui répète « 3 personnes présentes » toutes les 30 secondes
pendant une heure ne se lit pas. Chaque tranche ne retient que les **transitions** :
qui est arrivé, qui est parti, quels incidents ont été levés, quels objets
restent au-dessus du seuil de priorité. Une tranche sans transition est vide, et
le rapport la passe sous silence.

Échantillonnage en temps vidéo
-------------------------------
Le découpage suit `video_time`, jamais l'horloge de traitement. La même vidéo
analysée sur un portable lent ou sur un GPU produit exactement la même
chronologie — condition nécessaire pour qu'un rapport soit vérifiable.

En direct, `video_time` est lui-même l'horloge murale (voir `source.py`) : les
tranches correspondent alors à des minutes réelles, ce qui est le comportement
attendu d'une surveillance continue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

import config
from sentinel.events import Event
from sentinel.tracker import TrackedObject

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Met une durée en forme `mm:ss`, ou `hh:mm:ss` au-delà de l'heure.

    Args:
        seconds: Durée en secondes.

    Returns:
        La durée formatée.
    """
    seconds = max(0.0, seconds)
    heures, reste = divmod(int(seconds), 3600)
    minutes, secondes = divmod(reste, 60)
    if heures:
        return f"{heures:d}:{minutes:02d}:{secondes:02d}"
    return f"{minutes:02d}:{secondes:02d}"


@dataclass(frozen=True, slots=True)
class TimelineFact:
    """Un fait marquant, situé dans une tranche.

    Attributes:
        kind: Nature du fait — `"entree"`, `"sortie"`, `"incident"` ou
            `"priorite"`. Sert au tri et à l'affichage, jamais à une décision.
        label: Description lisible, telle qu'elle apparaîtra dans le rapport.
        video_time: Instant précis du fait, en temps vidéo.
    """

    kind: str
    label: str
    video_time: float

    # Ordre de présentation dans une tranche : les incidents d'abord, le
    # mouvement ensuite. Un lecteur cherche les incidents, pas les allées et
    # venues.
    ORDER = {"incident": 0, "priorite": 1, "entree": 2, "sortie": 3}

    @property
    def rank(self) -> int:
        """Rang d'affichage du fait dans sa tranche."""
        return self.ORDER.get(self.kind, 9)


@dataclass(frozen=True, slots=True)
class TimelineSlice:
    """Une tranche de temps vidéo et ses faits marquants.

    Attributes:
        index: Numéro de la tranche depuis le début de la session.
        start_s: Début de la tranche, en temps vidéo.
        duration_s: Durée nominale de la tranche.
        facts: Faits retenus, déjà triés.
        omitted: Nombre de faits écartés par le plafond de la tranche.
    """

    index: int
    start_s: float
    duration_s: float
    facts: tuple[TimelineFact, ...] = ()
    omitted: int = 0

    @property
    def end_s(self) -> float:
        """Fin de la tranche, en temps vidéo."""
        return self.start_s + self.duration_s

    @property
    def label(self) -> str:
        """Intitulé de la tranche, du type `00:30 – 01:00`."""
        return f"{format_duration(self.start_s)} - {format_duration(self.end_s)}"

    @property
    def is_notable(self) -> bool:
        """True si la tranche contient au moins un fait."""
        return bool(self.facts)


class Timeline:
    """Accumule les faits marquants d'une session, tranche par tranche.

    S'alimente au fil de l'analyse et se consulte à tout moment — c'est ce qui
    permet d'exporter un rapport en cours de surveillance sans interrompre le
    flux :

        >>> timeline = Timeline()
        >>> timeline.observe(12.0, tracker.active(), incidents)
        >>> for tranche in timeline.slices():
        ...     print(tranche.label, [f.label for f in tranche.facts])
    """

    def __init__(self, settings: config.TimelineConfig | None = None) -> None:
        """Prépare une chronologie vide.

        Args:
            settings: Réglages d'échantillonnage. `None` = `config.TIMELINE`.
        """
        self.settings = settings or config.TIMELINE
        self._facts: dict[int, list[TimelineFact]] = {}
        self._present: dict[int, str] = {}
        self._notable: set[int] = set()
        self._last_video_time: float = 0.0
        self._first_wall: datetime | None = None
        self._last_wall: datetime | None = None

    # -- Observation ----------------------------------------------------------

    def observe(
        self,
        video_time: float,
        present: Sequence[TrackedObject],
        events: Iterable[Event] = (),
        wall_time: datetime | None = None,
    ) -> None:
        """Enregistre l'état d'une frame et en déduit les transitions.

        Args:
            video_time: Temps vidéo de la frame.
            present: Objets actuellement en mémoire. Il faut passer les objets
                **retenus par le tracker** (`tracker.active()`) et non ceux vus
                sur cette seule frame : la rétention absorbe déjà les occlusions
                courtes, sinon chaque objet momentanément masqué produirait une
                fausse sortie suivie d'une fausse entrée.
            events: Incidents levés sur cette frame.
            wall_time: Horodatage réel, mémorisé pour dater le rapport.
        """
        self._last_video_time = max(self._last_video_time, video_time)
        if wall_time is not None:
            self._first_wall = self._first_wall or wall_time
            self._last_wall = wall_time

        courant = {obj.track_id: obj.class_name for obj in present}

        for track_id, class_name in courant.items():
            if track_id not in self._present:
                self._record(video_time, "entree", f"{class_name} #{track_id} entre dans le champ")

        for track_id, class_name in self._present.items():
            if track_id not in courant:
                self._record(video_time, "sortie", f"{class_name} #{track_id} quitte le champ")
                self._notable.discard(track_id)

        self._present = courant

        for event in events:
            self._record(
                video_time,
                "incident",
                self._describe_incident(event),
            )
            # Un objet devenu prioritaire est signalé une fois, à ce moment. Le
            # répéter à chaque tranche ferait exactement le bruit que cette
            # chronologie cherche à éviter.
            if self._is_notable(event) and event.track_id not in self._notable:
                self._notable.add(event.track_id)
                self._record(
                    video_time,
                    "priorite",
                    f"{event.class_name} #{event.track_id} passe en priorité "
                    f"{event.priority.level.value}",
                )

    def _is_notable(self, event: Event) -> bool:
        """Indique si un incident hisse son objet au rang des objets à surveiller.

        Args:
            event: Incident à examiner.

        Returns:
            True si sa priorité atteint le seuil configuré.
        """
        if event.priority is None:
            return False
        ordre = [
            config.PriorityLevel.LOW,
            config.PriorityLevel.MEDIUM,
            config.PriorityLevel.HIGH,
            config.PriorityLevel.CRITICAL,
        ]
        return ordre.index(event.priority.level) >= ordre.index(self.settings.notable_from)

    @staticmethod
    def _describe_incident(event: Event) -> str:
        """Formule un incident en une ligne de chronologie.

        Args:
            event: Incident à décrire.

        Returns:
            La ligne, avec le niveau de priorité quand il est connu.
        """
        zone = f" ({event.zone_name})" if event.zone_name else ""
        niveau = f" [{event.priority.level.value}]" if event.priority else ""
        return (
            f"{event.event_type.value.replace('_', ' ')} - "
            f"{event.class_name} #{event.track_id}{zone}{niveau}"
        )

    def _record(self, video_time: float, kind: str, label: str) -> None:
        """Range un fait dans la tranche correspondant à son instant.

        Args:
            video_time: Instant du fait, en temps vidéo.
            kind: Nature du fait.
            label: Description lisible.
        """
        index = int(max(0.0, video_time) // self.settings.slice_seconds)
        self._facts.setdefault(index, []).append(TimelineFact(kind, label, video_time))

    # -- Consultation ---------------------------------------------------------

    def slices(self, *, only_notable: bool = True) -> list[TimelineSlice]:
        """Restitue la chronologie, tranche par tranche.

        Args:
            only_notable: Ne rendre que les tranches contenant au moins un fait.
                C'est le comportement attendu d'un rapport : une tranche sans
                transition n'apprend rien et allonge la lecture.

        Returns:
            Les tranches, dans l'ordre chronologique.
        """
        largeur = self.settings.slice_seconds
        dernier = int(self._last_video_time // largeur)
        plafond = self.settings.max_facts_per_slice

        tranches: list[TimelineSlice] = []
        for index in range(dernier + 1):
            faits = sorted(
                self._facts.get(index, []),
                key=lambda fait: (fait.rank, fait.video_time),
            )
            if only_notable and not faits:
                continue
            tranches.append(
                TimelineSlice(
                    index=index,
                    start_s=index * largeur,
                    duration_s=largeur,
                    facts=tuple(faits[:plafond]),
                    omitted=max(0, len(faits) - plafond),
                )
            )
        return tranches

    @property
    def duration_s(self) -> float:
        """Durée de vidéo couverte par la chronologie, en secondes."""
        return self._last_video_time

    @property
    def started_at(self) -> datetime | None:
        """Horodatage réel de la première observation."""
        return self._first_wall

    @property
    def ended_at(self) -> datetime | None:
        """Horodatage réel de la dernière observation."""
        return self._last_wall

    @property
    def present_count(self) -> int:
        """Nombre d'objets actuellement suivis."""
        return len(self._present)

    def reset(self) -> None:
        """Vide la chronologie. À appeler entre deux sessions."""
        self._facts.clear()
        self._present.clear()
        self._notable.clear()
        self._last_video_time = 0.0
        self._first_wall = None
        self._last_wall = None

    def __len__(self) -> int:
        """Nombre total de faits enregistrés."""
        return sum(len(faits) for faits in self._facts.values())

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        return (
            f"Timeline({len(self)} faits, {format_duration(self.duration_s)} "
            f"de vidéo, tranche {self.settings.slice_seconds:.0f} s)"
        )


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Ce qu'il faut savoir d'une session pour en-tête de rapport.

    Attributes:
        source_label: Désignation lisible de la source, **sans identifiants**.
        is_live: True si la source est un direct — la période se lit alors
            différemment : elle est en cours, pas close.
        model_label: Modèle de détection employé.
        confidence: Seuil de confiance appliqué.
        watched_classes: Classes réellement surveillées.
        duration_factor: Facteur appliqué aux durées de déclenchement.
    """

    source_label: str = "source non précisée"
    is_live: bool = False
    model_label: str = "-"
    confidence: float = 0.0
    watched_classes: tuple[str, ...] = ()
    duration_factor: float = 1.0

    def settings_lines(self) -> list[str]:
        """Résume les réglages actifs, une ligne par élément.

        Returns:
            Les lignes prêtes à être insérées dans un en-tête de rapport.
        """
        classes = ", ".join(self.watched_classes) if self.watched_classes else "toutes"
        lignes = [
            f"Source            : {self.source_label}"
            + (" (flux en direct)" if self.is_live else ""),
            f"Modèle            : {self.model_label}",
            f"Seuil de confiance: {self.confidence:.0%}",
            f"Objets surveillés : {classes}",
        ]
        if self.duration_factor != 1.0:
            lignes.append(f"Durées ajustées   : x {self.duration_factor:g}")
        return lignes
