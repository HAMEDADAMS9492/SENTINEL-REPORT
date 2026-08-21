"""Moteur de règles et objets Event — ÉTAPE 5.

C'est le cœur métier du projet : transformer « un objet suivi se trouve dans une
zone depuis N secondes » en « incident de sécurité qualifié ».

Le moteur ne détecte rien et ne dessine rien. Il consomme des `TrackedObject`
(mémoire temporelle) et des noms de zones (géométrie), applique les `EventRule`
de `config.py`, et produit des `Event` immuables. Ajouter un nouveau type
d'incident consiste à ajouter une `EventRule` dans la configuration et, si
nécessaire, un prédicat dans ce module — jamais à toucher au détecteur ou à
l'interface.

Les trois filtres anti-fausses-alertes
--------------------------------------
Un système d'alerte qui crie tout le temps n'est pas utilisé. Trois garde-fous se
cumulent, et aucun n'est optionnel :

1. **Confirmation** (`tracker.py`) — un objet vu moins de `min_hits` fois
   n'atteint jamais ce module.
2. **Durée minimale** (`rule.min_duration_s`) — c'est ce qui distingue une
   intrusion d'un simple passage. Traverser une zone ne déclenche rien.
3. **Délai de garde** (`rule.cooldown_s`) — un même objet ne peut pas redéclencher
   le même type d'incident avant expiration. Sans lui, une personne restant
   2 minutes en zone restreinte produirait un rapport par frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import config
from sentinel.detection import Detection
from sentinel.tracker import TrackedObject, Tracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PriorityScore:
    """Score de priorité d'un incident, avec le détail de son calcul.

    Le détail n'est pas un ornement : c'est ce qui rend le score **opposable**.
    Un opérateur qui conteste un classement doit pouvoir voir d'où viennent les
    points, et un relecteur doit pouvoir refaire l'addition à la main.

    Attributes:
        value: Score total, arrondi à l'unité.
        level: Niveau lisible correspondant (`config.PriorityLevel`).
        contributions: Détail ligne par ligne, dans l'ordre du calcul.
    """

    value: float
    level: config.PriorityLevel
    contributions: tuple[str, ...] = ()

    @property
    def explanation(self) -> str:
        """Justification lisible, du type « Élevé (72 pts) : intrusion 30 + ... »."""
        detail = " + ".join(self.contributions) if self.contributions else "aucun signal"
        return f"{self.level.value.capitalize()} ({self.value:.0f} pts) : {detail}"


@dataclass(frozen=True, slots=True)
class Event:
    """Un incident de sécurité qualifié.

    Immuable : un incident est un fait constaté, il ne se modifie pas après coup.
    C'est la structure que `ReportGenerator` transforme en rapport.

    Attributes:
        event_id: Identifiant unique de l'incident (ex. `SR-20260813-0001`).
        event_type: Type d'incident (`config.EventType`).
        severity: Gravité (`config.Severity`).
        timestamp: Horodatage réel du déclenchement.
        video_time: Temps vidéo du déclenchement, en secondes.
        track_id: Identifiant de l'objet concerné.
        class_name: Classe de l'objet concerné.
        zone_name: Zone concernée, ou `None` si l'incident n'est pas localisé.
        duration_s: Durée de la condition au moment du déclenchement.
        confidence: Confiance de la détection à l'instant du déclenchement.
        bbox: Boîte englobante de l'objet.
        evidence_path: Chemin de la capture justificative, si elle a été écrite.
        details: Champs libres complémentaires (déplacement mesuré, propriétaire
            présumé, etc.), utilisés par les gabarits de rapport.
        priority: Score de priorité et sa justification. Calculé par
            `EventEngine`, jamais par `report.py` : c'est une **appréciation
            métier**, pas une décision de présentation.
    """

    event_id: str
    event_type: config.EventType
    severity: config.Severity
    timestamp: datetime
    video_time: float
    track_id: int
    class_name: str
    zone_name: str | None
    duration_s: float
    confidence: float
    bbox: tuple[float, float, float, float]
    evidence_path: Path | None = None
    details: dict[str, object] = field(default_factory=dict)
    priority: PriorityScore | None = None

    def to_dict(self) -> dict[str, object]:
        """Représentation sérialisable, pour le tableau Streamlit et l'export CSV/JSON."""
        x1, y1, x2, y2 = self.bbox
        return {
            "event_id": self.event_id,
            "type": self.event_type.value,
            "gravite": self.severity.value,
            "horodatage": self.timestamp.isoformat(timespec="seconds"),
            "temps_video_s": round(self.video_time, 2),
            "track_id": self.track_id,
            "classe": self.class_name,
            "zone": self.zone_name or "",
            "duree_s": round(self.duration_s, 1),
            "confiance": round(self.confidence, 3),
            "bbox": f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}",
            "priorite": self.priority.level.value if self.priority else "",
            "score": round(self.priority.value) if self.priority else 0,
            "justification": self.priority.explanation if self.priority else "",
            "preuve": str(self.evidence_path) if self.evidence_path else "",
            **{f"detail_{key}": value for key, value in self.details.items()},
        }


class EventEngine:
    """Applique les règles de `config.EVENT_RULES` aux objets suivis.

    Exemple d'utilisation prévue :
        >>> engine = EventEngine(zone_manager)
        >>> events = engine.evaluate(tracked_objects, tracker, frame, video_time)
    """

    def __init__(
        self,
        zone_manager,
        rules: Sequence[config.EventRule] | None = None,
        schedule: config.ScheduleConfig | None = None,
        scoring: config.ScoringConfig | None = None,
    ) -> None:
        """Initialise le moteur.

        Args:
            zone_manager: Instance de `sentinel.zones.ZoneManager`.
            rules: Règles à appliquer. `None` = `config.EVENT_RULES`.
            schedule: Horaires d'ouverture. `None` = `config.SCHEDULE`.
            scoring: Barème de priorité. `None` = `config.SCORING`.
        """
        self._zones = zone_manager
        self._rules: tuple[config.EventRule, ...] = tuple(
            config.EVENT_RULES if rules is None else rules
        )
        self._schedule = config.SCHEDULE if schedule is None else schedule
        self._scoring = config.SCORING if scoring is None else scoring
        self._history: list[Event] = []
        self._sequence: dict[str, int] = {}

    def evaluate(
        self,
        tracked_objects: Sequence[TrackedObject],
        tracker: Tracker,
        frame: np.ndarray,
        video_time: float,
        wall_time: datetime | None = None,
    ) -> list[Event]:
        """Évalue toutes les règles sur tous les objets d'une frame.

        Args:
            tracked_objects: Objets confirmés vus sur la frame.
            tracker: Tracker complet (nécessaire pour chercher un propriétaire
                présumé à proximité d'un objet).
            frame: Frame courante, pour la capture de preuve.
            video_time: Temps vidéo de la frame.
            wall_time: Horodatage réel. `None` = maintenant.

        Returns:
            Les événements déclenchés sur cette frame (souvent vide).
        """
        moment = wall_time or datetime.now()
        raised: list[Event] = []

        for obj in tracked_objects:
            for rule in self._rules:
                # 1. La règle concerne-t-elle cette classe d'objet ?
                if rule.classes and obj.class_name not in rule.classes:
                    continue

                # 2. Le délai de garde est-il écoulé ? Ce test vient AVANT le
                #    prédicat : inutile de recalculer une condition dont on sait
                #    déjà que le résultat ne pourra pas être publié.
                if not obj.can_raise(rule.event_type.value, video_time, rule.cooldown_s):
                    continue

                # 3. La condition métier est-elle remplie ?
                outcome = self._check_rule(obj, rule, tracker, video_time, moment)
                if outcome is None:
                    continue

                zone_name, duration_s, details = outcome
                event = self._build_event(
                    obj,
                    rule,
                    wall_time=moment,
                    video_time=video_time,
                    zone_name=zone_name,
                    duration_s=duration_s,
                    details=details,
                )
                event = self._attach_evidence(event, frame, obj)

                obj.mark_event(rule.event_type.value, video_time)
                self._history.append(event)
                raised.append(event)

                logger.info(
                    "Incident %s : %s (objet #%s, zone %s, %.0fs)",
                    event.event_id,
                    event.event_type.value,
                    event.track_id,
                    event.zone_name or "-",
                    event.duration_s,
                )

        return raised

    def _check_rule(
        self,
        obj: TrackedObject,
        rule: config.EventRule,
        tracker: Tracker,
        video_time: float,
        wall_time: datetime,
    ) -> tuple[str | None, float, dict[str, object]] | None:
        """Aiguille vers le prédicat correspondant au type de la règle.

        Args:
            obj: Objet suivi.
            rule: Règle appliquée.
            tracker: Tracker complet.
            video_time: Temps vidéo courant.
            wall_time: Horodatage réel.

        Returns:
            Le triplet `(zone, durée, détails)` si la règle se déclenche, sinon
            `None`.
        """
        event_type = rule.event_type

        if event_type is config.EventType.INTRUSION:
            zone = self._check_intrusion(obj, rule, video_time)
            if zone is None:
                return None
            return zone, obj.dwell_time(zone, video_time), {}

        if event_type is config.EventType.LOITERING:
            zone = self._check_loitering(obj, rule, video_time)
            if zone is None:
                return None
            return zone, obj.dwell_time(zone, video_time), {}

        if event_type is config.EventType.ABANDONED_OBJECT:
            if not self._check_abandoned_object(obj, rule, tracker, video_time):
                return None
            details: dict[str, object] = {
                "deplacement_px": round(obj.displacement(rule.min_duration_s), 1),
                "deplacement_relatif": round(obj.displacement_ratio(rule.min_duration_s), 3),
                "proprietaire_presume": "aucun",
            }
            return self._primary_zone(obj), obj.age, details

        if event_type is config.EventType.AFTER_HOURS:
            if not self._check_after_hours(obj, rule, wall_time):
                return None
            return self._primary_zone(obj), obj.age, {"heure_locale": wall_time.strftime("%H:%M")}

        # Une règle d'un type non géré est ignorée, jamais fatale : ajouter un
        # EventType dans config.py sans son prédicat ne doit pas arrêter
        # l'analyse en cours.
        logger.warning("Type de règle non pris en charge, ignoré : %s", event_type)
        return None

    @staticmethod
    def _primary_zone(obj: TrackedObject) -> str | None:
        """Zone représentative d'un objet occupant éventuellement plusieurs zones.

        Args:
            obj: Objet suivi.

        Returns:
            Un nom de zone déterministe (ordre alphabétique), ou `None`.
        """
        return sorted(obj.zones)[0] if obj.zones else None

    # -- Prédicats par type d'incident ---------------------------------------

    def _check_intrusion(
        self, obj: TrackedObject, rule: config.EventRule, video_time: float
    ) -> str | None:
        """Teste la présence prolongée dans une zone restreinte.

        Args:
            obj: Objet suivi.
            rule: Règle appliquée.
            video_time: Temps vidéo courant.

        Returns:
            Le nom de la zone en infraction, ou `None`.
        """
        candidates = (
            set(rule.zones) if rule.zones else set(self._zones.restricted_zone_names())
        )
        return self._first_zone_over_threshold(obj, candidates, rule.min_duration_s, video_time)

    def _check_loitering(
        self, obj: TrackedObject, rule: config.EventRule, video_time: float
    ) -> str | None:
        """Teste une présence anormalement longue, zone restreinte ou non.

        Même mécanisme que `_check_intrusion` mais sans filtrer sur le drapeau
        `restricted` et avec un seuil de durée bien supérieur : quelqu'un qui
        traverse le hall est normal, quelqu'un qui y stationne 60 secondes ne
        l'est pas forcément.

        Args:
            obj: Objet suivi.
            rule: Règle appliquée.
            video_time: Temps vidéo courant.

        Returns:
            Le nom de la zone concernée, ou `None`.
        """
        candidates = set(rule.zones) if rule.zones else set(obj.zones)
        return self._first_zone_over_threshold(obj, candidates, rule.min_duration_s, video_time)

    @staticmethod
    def _first_zone_over_threshold(
        obj: TrackedObject,
        candidates: set[str],
        min_duration_s: float,
        video_time: float,
    ) -> str | None:
        """Première zone occupée dont la durée de séjour dépasse le seuil.

        Le tri alphabétique n'est pas cosmétique : sans lui, l'ordre d'itération
        d'un `set` rendrait le choix de la zone non reproductible d'une exécution
        à l'autre, et deux analyses de la même vidéo produiraient des rapports
        différents.

        Args:
            obj: Objet suivi.
            candidates: Zones à considérer.
            min_duration_s: Seuil de déclenchement.
            video_time: Temps vidéo courant.

        Returns:
            Le nom de la zone, ou `None`.
        """
        for zone_name in sorted(obj.zones & candidates):
            if obj.dwell_time(zone_name, video_time) >= min_duration_s:
                return zone_name
        return None

    def _check_abandoned_object(
        self, obj: TrackedObject, rule: config.EventRule, tracker: Tracker, video_time: float
    ) -> bool:
        """Teste qu'un objet est immobile depuis longtemps et sans propriétaire.

        Trois conditions cumulées :
        1. `obj.age >= rule.min_duration_s`
        2. `obj.is_stationary(rule.max_movement_px, rule.min_duration_s)`
        3. `tracker.nearest(obj, class_names=("person",), radius_px=...)` est None

        Args:
            obj: Objet suivi.
            rule: Règle appliquée.
            tracker: Tracker, pour la recherche de propriétaire.
            video_time: Temps vidéo courant.

        Returns:
            True si l'objet est considéré comme abandonné.
        """
        if obj.age < rule.min_duration_s:
            return False

        movement_threshold = self._movement_threshold(obj, rule)
        if movement_threshold is not None and not obj.is_stationary(
            movement_threshold, rule.min_duration_s
        ):
            return False

        if rule.requires_no_owner:
            owner = tracker.nearest(
                obj, class_names=("person",), radius_px=self._owner_radius(obj, rule)
            )
            if owner is not None:
                return False

        return True

    @staticmethod
    def _movement_threshold(obj: TrackedObject, rule: config.EventRule) -> float | None:
        """Déplacement maximal toléré, en pixels, pour cet objet précis.

        Le seuil **relatif** l'emporte quand il est défini : 25 px de
        déplacement, c'est un frémissement pour un sac au premier plan et une
        traversée pour un sac au fond du champ. Le rapporter à la hauteur
        apparente de l'objet donne un critère qui garde le même sens à toutes
        les profondeurs. Le seuil en pixels reste le repli.

        Args:
            obj: Objet évalué.
            rule: Règle appliquée.

        Returns:
            Le seuil en pixels, ou `None` si le critère ne s'applique pas.
        """
        if rule.max_movement_ratio is not None:
            return rule.max_movement_ratio * obj.scale_px
        return rule.max_movement_px

    @staticmethod
    def _owner_radius(obj: TrackedObject, rule: config.EventRule) -> float:
        """Rayon de recherche du propriétaire présumé, en pixels.

        Même raisonnement que pour l'immobilité : un rayon fixe de 150 px couvre
        environ un mètre au premier plan et huit mètres au fond du champ.

        Args:
            obj: Objet évalué.
            rule: Règle appliquée.

        Returns:
            Le rayon en pixels.
        """
        if rule.owner_radius_ratio is not None:
            return rule.owner_radius_ratio * obj.scale_px
        return rule.owner_radius_px

    def _check_after_hours(
        self, obj: TrackedObject, rule: config.EventRule, wall_time: datetime
    ) -> bool:
        """Teste une présence en dehors des horaires d'ouverture.

        Args:
            obj: Objet suivi.
            rule: Règle appliquée.
            wall_time: Horodatage réel (c'est bien l'heure murale qui compte ici,
                pas le temps vidéo).

        Returns:
            True si l'heure courante est hors de la plage `config.SCHEDULE`.
        """
        if obj.age < rule.min_duration_s:
            return False
        return self.is_after_hours(wall_time)

    def is_after_hours(self, wall_time: datetime) -> bool:
        """Indique si un instant tombe hors des horaires d'ouverture du site.

        Args:
            wall_time: Instant à tester.

        Returns:
            True si le site est censé être fermé.
        """
        if self._schedule.weekend_closed and wall_time.weekday() >= 5:
            return True

        current = wall_time.time()
        return current < self._schedule.opening or current >= self._schedule.closing

    # -- Construction des incidents -------------------------------------------

    def _build_event(self, obj: TrackedObject, rule: config.EventRule, **kwargs) -> Event:
        """Assemble un `Event` à partir d'un objet et d'une règle.

        Args:
            obj: Objet concerné.
            rule: Règle déclenchée.
            **kwargs: Champs spécifiques (wall_time, video_time, zone_name,
                duration_s, details).

        Returns:
            L'événement construit.
        """
        wall_time: datetime = kwargs["wall_time"]
        return Event(
            event_id=self._next_event_id(wall_time),
            event_type=rule.event_type,
            severity=rule.severity,
            timestamp=wall_time,
            video_time=kwargs["video_time"],
            track_id=obj.track_id,
            class_name=obj.class_name,
            zone_name=kwargs.get("zone_name"),
            duration_s=kwargs["duration_s"],
            confidence=obj.detection.confidence,
            bbox=obj.detection.xyxy,
            details=kwargs.get("details") or {},
            priority=self._score_event(obj, rule, kwargs["duration_s"], wall_time),
        )

    def _score_event(
        self,
        obj: TrackedObject,
        rule: config.EventRule,
        duration_s: float,
        wall_time: datetime,
    ) -> PriorityScore:
        """Calcule le score de priorité d'un incident et le justifie.

        Le score n'apporte **aucune information nouvelle** : il agrège des
        signaux déjà mesurés par le pipeline — une règle qui s'est déclenchée,
        un nombre de secondes, une immobilité constatée, l'heure au calendrier —
        selon un barème publié dans `config.SCORING`. Il ne devine rien, il ne
        classe rien par ressemblance avec des cas passés : il **ordonne** ce que
        les règles ont déjà établi, pour qu'un opérateur sache quoi regarder en
        premier quand trente incidents tombent en même temps.

        Chaque terme du calcul est conservé dans `contributions`, si bien que le
        total se refait à la main.

        Args:
            obj: Objet concerné, porteur de l'historique des règles déjà levées.
            rule: Règle qui vient de se déclencher.
            duration_s: Durée constatée de la condition.
            wall_time: Horodatage réel, qui décide du caractère hors horaires.

        Returns:
            Le score, son niveau et le détail de son calcul.
        """
        bareme = self._scoring
        contributions: list[str] = []

        # 1. Points de base du type de règle.
        score = bareme.points_for(rule.event_type)
        contributions.append(f"{rule.event_type.value} ({score:.0f})")

        minutes = max(0.0, duration_s) / 60.0

        # 2. Durée : une intrusion de dix minutes n'est pas une intrusion de
        #    trois secondes. Plafonnée, sinon une présence très longue écraserait
        #    tous les autres signaux.
        duration_points = min(minutes * bareme.points_per_minute_present, bareme.max_duration_points)
        if duration_points > 0:
            score += duration_points
            contributions.append(f"présence {duration_s:.0f} s (+{duration_points:.0f})")

        # 3. Immobilité : ne concerne que les règles qui la mesurent réellement.
        #    L'attribuer à une règle qui ne l'observe pas serait inventer un
        #    signal.
        mesure_immobilite = (
            rule.max_movement_ratio is not None or rule.max_movement_px is not None
        )
        if mesure_immobilite:
            still_points = min(
                minutes * bareme.points_per_minute_stationary, bareme.max_stationary_points
            )
            if still_points > 0:
                score += still_points
                contributions.append(f"immobilité {duration_s:.0f} s (+{still_points:.0f})")

        # 4. Cumul : le signal le plus fort du barème. `last_event_time` ne
        #    contient que les règles DÉJÀ levées sur cet objet — `mark_event()`
        #    n'est appelé qu'après. Un objet qui a rôdé, puis pénétré en zone
        #    restreinte, puis abandonné un sac raconte une histoire qu'aucune
        #    des trois règles ne dit seule.
        autres = [t for t in obj.last_event_time if t != rule.event_type.value]
        if autres:
            bonus = min(len(autres) * bareme.combination_points, bareme.max_combination_points)
            score += bonus
            contributions.append(f"{len(autres)} autre(s) règle(s) sur cet objet (+{bonus:.0f})")

        # 5. Hors horaires : multiplicateur et non bonus fixe, pour que la
        #    gravité nocturne reste proportionnelle à la gravité diurne. Le même
        #    comportement compte davantage à 3 h du matin qu'à midi.
        if self.is_after_hours(wall_time):
            score *= bareme.after_hours_multiplier
            contributions.append(f"hors horaires (× {bareme.after_hours_multiplier:g})")

        return PriorityScore(
            value=round(score, 1),
            level=bareme.level_for(score),
            contributions=tuple(contributions),
        )


    def _next_event_id(self, wall_time: datetime) -> str:
        """Génère un identifiant d'incident lisible et croissant.

        Format : `{prefix}-{AAAAMMJJ}-{séquence:04d}`, ex. `SR-20260813-0007`.

        Args:
            wall_time: Date de l'incident.

        Returns:
            L'identifiant.
        """
        day = wall_time.strftime("%Y%m%d")
        self._sequence[day] = self._sequence.get(day, 0) + 1
        return f"{config.REPORT.reference_prefix}-{day}-{self._sequence[day]:04d}"

    def _attach_evidence(self, event: Event, frame: np.ndarray, obj: TrackedObject) -> Event:
        """Retourne une copie de l'incident enrichie du chemin de sa preuve.

        `Event` étant immuable, on ne peut pas lui affecter le chemin après coup :
        on reconstruit l'objet. C'est le prix — modeste — de la garantie qu'un
        incident déjà publié ne change jamais.

        Args:
            event: Incident sans preuve.
            frame: Frame au moment de l'incident.
            obj: Objet concerné.

        Returns:
            L'incident, avec `evidence_path` renseigné si la capture a réussi.
        """
        path = self._capture_evidence(frame, obj, event)
        if path is None:
            return event

        # `replace()` plutôt qu'une reconstruction champ par champ : tout champ
        # ajouté à `Event` serait sinon silencieusement perdu ici — c'est
        # exactement ce qui serait arrivé au score de priorité.
        return replace(event, evidence_path=path)

    def _capture_evidence(
        self, frame: np.ndarray, obj: TrackedObject, event: Event
    ) -> Path | None:
        """Écrit la capture justificative sur disque.

        Args:
            frame: Frame au moment de l'incident.
            obj: Objet concerné (pour l'annotation ou le recadrage).
            event: Incident, dont l'identifiant sert au nom de fichier.

        Returns:
            Le chemin du fichier écrit, ou `None` en cas d'échec (une capture
            ratée ne doit pas annuler l'incident).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        settings = config.EVIDENCE
        try:
            filename = settings.filename_template.format(
                timestamp=event.timestamp.strftime(settings.timestamp_format),
                event_type=event.event_type.value,
                track_id=obj.track_id,
                zone=event.zone_name or "hors-zone",
            )
            destination = settings.directory / filename

            image = frame.copy()
            if settings.annotate:
                image = self._annotate_evidence(image, obj, event)
            if settings.crop_to_object:
                image = self._crop_around(image, obj.detection, settings.margin_px)

            settings.directory.mkdir(parents=True, exist_ok=True)
            written = cv2.imwrite(
                str(destination),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality],
            )
            if not written:
                logger.warning("Écriture de la preuve refusée par OpenCV : %s", destination)
                return None
            return destination
        except Exception:
            # Un disque plein ou un nom de fichier invalide ne doit pas faire
            # disparaître l'incident : le rapport existera, simplement sans image.
            logger.exception("Capture de preuve impossible pour l'incident %s.", event.event_id)
            return None

    @staticmethod
    def _annotate_evidence(image: np.ndarray, obj: TrackedObject, event: Event) -> np.ndarray:
        """Entoure l'objet en cause et inscrit le type d'incident.

        Args:
            image: Image à annoter (modifiée en place).
            obj: Objet concerné.
            event: Incident décrit.

        Returns:
            L'image annotée.
        """
        from sentinel.zones import _ascii_label  # translittération partagée

        x1, y1, x2, y2 = obj.detection.as_int_box()
        color = (0, 0, 255)  # rouge BGR : c'est l'objet en cause
        cv2.rectangle(image, (x1, y1), (x2, y2), color, config.UI.box_thickness)

        label = _ascii_label(f"{event.event_type.value} #{obj.track_id}")
        cv2.putText(
            image,
            label,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            config.UI.box_thickness,
            cv2.LINE_AA,
        )
        return image

    @staticmethod
    def _crop_around(image: np.ndarray, detection: Detection, margin_px: int) -> np.ndarray:
        """Recadre l'image autour d'une détection, avec une marge.

        Args:
            image: Image source.
            detection: Détection à cadrer.
            margin_px: Marge ajoutée de chaque côté.

        Returns:
            Le recadrage, borné aux limites de l'image.
        """
        height, width = image.shape[:2]
        x1, y1, x2, y2 = detection.as_int_box()
        return image[
            max(0, y1 - margin_px) : min(height, y2 + margin_px),
            max(0, x1 - margin_px) : min(width, x2 + margin_px),
        ]

    # -- Consultation ----------------------------------------------------------

    @property
    def history(self) -> list[Event]:
        """Tous les incidents levés depuis le démarrage de la session."""
        return list(self._history)

    def clear(self) -> None:
        """Vide l'historique et remet le compteur d'incidents à zéro."""
        self._history.clear()
        self._sequence.clear()

    def __len__(self) -> int:
        """Nombre d'incidents levés."""
        return len(self._history)

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        return f"EventEngine(regles={len(self._rules)}, incidents={len(self._history)})"
