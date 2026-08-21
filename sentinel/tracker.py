"""Mémoire temporelle des objets suivis — ÉTAPE 3.

Rôle : maintenir, pour chaque objet identifié par ByteTrack, l'information que
`events.py` ne peut pas déduire d'une frame isolée — depuis quand l'objet est là,
par où il est passé, combien de temps il est resté dans chaque zone.

Ce module **ne fait aucune association d'identités** : celle-ci est réalisée par
ByteTrack via `Detector.track()`.

Pourquoi l'identité persistante est indispensable
-------------------------------------------------
Le prototype d'origine dédupliquait les objets par proximité des centres d'une
frame à l'autre. Cette approche donne un *comptage* à peu près correct, mais elle
ne produit pas d'identité : rien ne garantit que « la personne près du centre à la
frame 100 » soit celle de la frame 99. Sans identité stable, aucune durée n'est
mesurable, et sans durée il n'existe aucun événement temporel : « présent depuis
3 secondes », « immobile depuis 30 secondes » et « déjà signalé il y a 1 minute »
deviennent tous indécidables. Toutes les règles de `events.py` reposent donc sur
la couche construite ici.

Deux horloges cohabitent volontairement :
    * `video_time` (secondes depuis le début du flux, calculé depuis le numéro de
      frame et le FPS) pilote **toutes les décisions métier** — une vidéo analysée
      en accéléré produit exactement les mêmes événements.
    * `wall_time` (horodatage réel) sert uniquement à **dater le rapport**.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

import config
from sentinel.detection import Detection, count_by_class

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Un objet suivi et sa mémoire temporelle.

    Contrairement à `Detection` (fait figé sur une frame), un `TrackedObject` est
    **mutable** : il agrège l'histoire d'un même objet au fil des frames.

    Attributes:
        track_id: Identifiant persistant fourni par ByteTrack.
        class_name: Classe de l'objet.
        first_seen: Temps vidéo de la première apparition, en secondes.
        last_seen: Temps vidéo de la dernière apparition, en secondes.
        first_seen_wall: Horodatage réel de la première apparition.
        last_seen_wall: Horodatage réel de la dernière apparition.
        detection: Dernière détection connue.
        hits: Nombre de frames où l'objet a été vu.
        misses: Nombre de frames consécutives sans le voir.
        history: Fenêtre glissante de `(temps_video, point_d_appui)`.
        zones: Zones occupées sur la frame courante.
        zone_entry_time: Temps vidéo d'entrée dans chaque zone occupée.
        last_event_time: Temps vidéo du dernier événement levé, par type
            d'événement — support de l'anti-rebond.
    """

    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    first_seen_wall: datetime
    last_seen_wall: datetime
    detection: Detection
    hits: int = 1
    misses: int = 0
    history: deque[tuple[float, tuple[float, float]]] = field(default_factory=deque)
    zones: set[str] = field(default_factory=set)
    zone_entry_time: dict[str, float] = field(default_factory=dict)
    last_event_time: dict[str, float] = field(default_factory=dict)

    # -- État interne de stabilisation ---------------------------------------
    # Votes de classe : YOLO peut changer d'avis d'une frame à l'autre sur un
    # même objet suivi (« truck » puis « bus »). Sans vote, la classe retenue
    # serait celle de la toute première frame, souvent la moins fiable puisque
    # l'objet y est le plus petit ou le plus partiellement visible.
    class_votes: Counter[str] = field(default_factory=Counter)
    # Compteurs d'hystérésis sur les frontières de zone.
    _pending_zones: dict[str, tuple[int, float]] = field(default_factory=dict)
    _absent_zones: dict[str, int] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Durée totale de présence, en secondes de temps vidéo."""
        return self.last_seen - self.first_seen

    @property
    def is_confirmed(self) -> bool:
        """True si l'objet a été vu au moins `config.TRACKING.min_hits` fois.

        Filtre les détections fugaces d'une ou deux frames, qui sont presque
        toujours des faux positifs et généreraient des rapports parasites.
        """
        return self.hits >= config.TRACKING.min_hits

    @property
    def position(self) -> tuple[float, float]:
        """Dernier point d'appui connu (milieu du bord inférieur de la boîte)."""
        return self.detection.anchor

    @property
    def scale_px(self) -> float:
        """Taille apparente de l'objet, en pixels — sa hauteur de boîte.

        Sert d'unité de mesure aux seuils relatifs. Un objet deux fois plus loin
        de la caméra apparaît deux fois plus petit : rapporter une distance à
        cette hauteur donne une grandeur à peu près indépendante de la
        profondeur, sans calibration. C'est une approximation — elle suppose des
        objets de taille physique comparable — mais elle corrige l'essentiel de
        l'erreur commise par un seuil en pixels fixes.

        Returns:
            La hauteur de la dernière boîte connue, au minimum 1 pixel.
        """
        return max(1.0, self.detection.height)

    def update(self, detection: Detection, video_time: float, wall_time: datetime) -> None:
        """Enregistre une nouvelle observation de cet objet.

        Args:
            detection: Détection de la frame courante.
            video_time: Temps vidéo de la frame, en secondes.
            wall_time: Horodatage réel de la frame.
        """
        self.detection = detection
        self.last_seen = video_time
        self.last_seen_wall = wall_time
        self.hits += 1

        # La classe retenue est celle qui a recueilli le plus de votes depuis le
        # début du suivi, et non celle de la première frame. En cas d'égalité,
        # l'ordre alphabétique tranche pour que deux analyses de la même vidéo
        # produisent le même rapport.
        self.class_votes[detection.class_name] += 1
        self.class_name = min(self.class_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        # `misses` compte les absences **consécutives** : toute réapparition le
        # remet à zéro, sinon un objet vu par intermittence finirait par être
        # purgé alors qu'il est bel et bien présent.
        self.misses = 0
        self._push_history(video_time, detection.anchor)

    def mark_missed(self) -> None:
        """Signale que l'objet n'a pas été vu sur la frame courante."""
        self.misses += 1

    def _push_history(self, video_time: float, point: tuple[float, float]) -> None:
        """Ajoute une position et purge ce qui sort de la fenêtre d'historique.

        Args:
            video_time: Temps vidéo de l'observation.
            point: Point d'appui observé.
        """
        self.history.append((video_time, point))
        cutoff = video_time - config.TRACKING.history_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def update_zones(self, zone_names: Iterable[str], video_time: float) -> None:
        """Met à jour les zones occupées et chronomètre les entrées.

        `Tracker` ne connaît pas la géométrie : c'est `ZoneManager` qui calcule
        l'appartenance et le pipeline qui appelle cette méthode. Cette inversion
        garde `tracker.py` indépendant de `zones.py`.

        Args:
            zone_names: Zones contenant l'objet sur la frame courante.
            video_time: Temps vidéo de la frame.
        """
        current = set(zone_names)
        entry_frames = max(1, config.GEOMETRY.min_overlap_frames)
        exit_frames = max(1, config.GEOMETRY.exit_tolerance_frames)

        # -- Entrées : il faut `min_overlap_frames` frames consécutives --------
        for zone in current:
            self._absent_zones.pop(zone, None)  # la zone est de nouveau occupée
            if zone in self.zones:
                continue

            seen, since = self._pending_zones.get(zone, (0, video_time))
            seen += 1
            if seen >= entry_frames:
                self._pending_zones.pop(zone, None)
                self.zones.add(zone)
                # Le chronomètre part du **premier contact**, pas de l'instant
                # de confirmation : les frames d'observation ont bien été
                # passées dans la zone, les décompter fausserait la durée.
                self.zone_entry_time[zone] = since
            else:
                self._pending_zones[zone] = (seen, since)

        # Une entrée en cours de validation qui s'interrompt repart de zéro.
        for zone in list(self._pending_zones):
            if zone not in current:
                del self._pending_zones[zone]

        # -- Sorties : il faut `exit_tolerance_frames` frames consécutives -----
        for zone in list(self.zones - current):
            missed = self._absent_zones.get(zone, 0) + 1
            if missed >= exit_frames:
                self._absent_zones.pop(zone, None)
                self.zones.discard(zone)
                self.zone_entry_time.pop(zone, None)
            else:
                # Absence tolérée : le chronomètre continue de courir. C'est
                # exactement le cas d'une boîte qui tremble sur la frontière.
                self._absent_zones[zone] = missed

    def dwell_time(self, zone_name: str, video_time: float) -> float:
        """Durée passée sans interruption dans une zone.

        Args:
            zone_name: Zone interrogée.
            video_time: Temps vidéo courant.

        Returns:
            La durée en secondes, ou `0.0` si l'objet n'y est pas.
        """
        entry_time = self.zone_entry_time.get(zone_name)
        if entry_time is None:
            return 0.0
        return max(0.0, video_time - entry_time)

    def displacement(self, window_s: float | None = None) -> float:
        """Déplacement maximal observé sur une fenêtre récente, en pixels.

        On mesure l'écart maximal à la position la plus ancienne de la fenêtre,
        et non la distance cumulée : cette dernière gonfle avec le bruit de
        détection (une boîte qui « tremble » de 2 px par frame accumulerait des
        centaines de pixels alors que l'objet est immobile).

        Args:
            window_s: Fenêtre d'observation. `None` = fenêtre configurée.

        Returns:
            L'écart maximal en pixels ; `0.0` si l'historique est insuffisant.
        """
        window = config.TRACKING.history_seconds if window_s is None else window_s
        points = self._points_within(window)
        if len(points) < 2:
            return 0.0

        origin_x, origin_y = points[0]
        return max(math.hypot(x - origin_x, y - origin_y) for x, y in points[1:])

    def displacement_ratio(self, window_s: float | None = None) -> float:
        """Déplacement récent, exprimé en fraction de la hauteur de l'objet.

        Grandeur sans dimension, comparable d'un objet à l'autre et d'un plan à
        l'autre : 0,1 signifie « a bougé d'un dixième de sa propre taille »,
        que l'objet fasse 30 ou 300 pixels de haut.

        Args:
            window_s: Fenêtre d'observation. `None` = fenêtre configurée.

        Returns:
            Le rapport déplacement / hauteur.
        """
        return self.displacement(window_s) / self.scale_px

    def is_stationary(self, max_movement_px: float, window_s: float | None = None) -> bool:
        """Indique si l'objet est resté quasiment immobile.

        Args:
            max_movement_px: Déplacement maximal toléré, en pixels.
            window_s: Fenêtre d'observation, en secondes.

        Returns:
            True si l'objet a été observé sur toute la fenêtre sans dépasser le
            seuil de déplacement.
        """
        window = config.TRACKING.history_seconds if window_s is None else window_s
        if len(self.history) < 2:
            return False

        # Exiger que l'historique **couvre** la fenêtre est essentiel : un sac
        # vu depuis 2 secondes est trivialement « immobile », mais ça ne prouve
        # rien. Sans ce contrôle, la règle « objet abandonné » se déclencherait
        # dès l'apparition de l'objet.
        observed_span = self.history[-1][0] - self.history[0][0]
        if observed_span < window:
            return False

        return self.displacement(window) <= max_movement_px

    def _points_within(self, window_s: float) -> list[tuple[float, float]]:
        """Positions de l'historique comprises dans les `window_s` dernières secondes.

        Args:
            window_s: Profondeur de la fenêtre, en secondes.

        Returns:
            Les points, du plus ancien au plus récent.
        """
        if not self.history:
            return []
        cutoff = self.history[-1][0] - window_s
        return [point for timestamp, point in self.history if timestamp >= cutoff]

    def can_raise(self, event_type: str, video_time: float, cooldown_s: float) -> bool:
        """Vérifie que le délai de garde (anti-rebond) est écoulé.

        Sans ce garde-fou, une personne restant 2 minutes en zone restreinte
        génèrerait un événement par frame, soit environ 3 000 rapports.

        Args:
            event_type: Type d'événement concerné.
            video_time: Temps vidéo courant.
            cooldown_s: Délai minimal entre deux déclenchements.

        Returns:
            True si l'événement peut être levé maintenant.
        """
        last_time = self.last_event_time.get(event_type)
        if last_time is None:
            return True
        return (video_time - last_time) >= cooldown_s

    def mark_event(self, event_type: str, video_time: float) -> None:
        """Enregistre qu'un événement vient d'être levé pour cet objet.

        Args:
            event_type: Type d'événement levé.
            video_time: Temps vidéo du déclenchement.
        """
        self.last_event_time[event_type] = video_time

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        return (
            f"TrackedObject(#{self.track_id}, {self.class_name!r}, "
            f"age={self.age:.1f}s, hits={self.hits}, zones={sorted(self.zones)})"
        )


class Tracker:
    """Maintient l'état temporel des objets suivis par le détecteur.

    Ne détecte rien, n'associe rien : reçoit des détections déjà porteuses d'un
    `track_id` et construit la mémoire dont dépendent les règles d'événements.

    Utilisation prévue :
        >>> tracker = Tracker(detector)
        >>> objets = tracker.update(frame, frame_index=42, fps=25.0)
    """

    def __init__(self, detector, *, max_age_s: float | None = None) -> None:
        """Initialise la mémoire de suivi.

        Args:
            detector: Instance de `sentinel.detector.Detector`.
            max_age_s: Durée de rétention d'un objet non revu. `None` = valeur de
                `config.TRACKING`.
        """
        self._detector = detector
        self._max_age_s: float = (
            config.TRACKING.max_age_s if max_age_s is None else max_age_s
        )
        self._objects: dict[int, TrackedObject] = {}
        self._untracked: list[Detection] = []
        self._video_time: float = 0.0

    def update(
        self,
        frame,
        *,
        frame_index: int,
        fps: float | None = None,
        wall_time: datetime | None = None,
        video_time: float | None = None,
    ) -> list[TrackedObject]:
        """Traite une frame et met à jour l'état de tous les objets.

        Args:
            frame: Image BGR de la frame courante.
            frame_index: Numéro de la frame depuis le début du flux.
            fps: Images par seconde de la source. `None` = valeur par défaut de
                la configuration (les webcams ne l'exposent pas toujours).
            wall_time: Horodatage réel. `None` = maintenant.
            video_time: Temps métier **imposé** par l'appelant. À renseigner pour
                une source en direct, où des images sont volontairement sautées
                pour rattraper le retard : le compteur d'images ne mesure alors
                plus le temps écoulé, et `frame_index / fps` sous-estimerait
                toutes les durées. `None` = calcul depuis le numéro d'image, ce
                qui est le bon choix sur un fichier car il rend l'analyse
                reproductible.

        Returns:
            Les objets confirmés et vus sur cette frame, triés par `track_id`.
        """
        if video_time is not None:
            self._video_time = video_time
        else:
            # Un FPS nul ou absurde (webcam qui ne renseigne pas la propriété)
            # donnerait une division par zéro ou un temps vidéo aberrant : on
            # retombe sur la valeur de configuration plutôt que de propager
            # l'erreur. Le contrôle de vraisemblance est celui de `config.VIDEO`,
            # pas une seconde version locale qui pourrait en diverger.
            self._video_time = frame_index / config.VIDEO.credible_fps(fps)
        moment = wall_time or datetime.now()

        detections = self._detector.track(frame)

        seen: set[int] = set()
        untracked: list[Detection] = []

        for detection in detections:
            track_id = detection.track_id
            if track_id is None:
                # ByteTrack n'a pas encore confirmé cette piste : la détection
                # existe (on peut la dessiner) mais elle n'a pas d'identité, donc
                # aucune mémoire temporelle ne peut lui être attachée.
                untracked.append(detection)
                continue

            existing = self._objects.get(track_id)
            if existing is None:
                self._objects[track_id] = self._create(detection, moment)
            else:
                existing.update(detection, self._video_time, moment)
            seen.add(track_id)

        self._untracked = untracked

        for track_id, obj in self._objects.items():
            if track_id not in seen:
                obj.mark_missed()

        self._purge()

        return sorted(
            (self._objects[track_id] for track_id in seen if self._objects[track_id].is_confirmed),
            key=lambda obj: obj.track_id,
        )

    def _create(self, detection: Detection, wall_time: datetime) -> TrackedObject:
        """Crée l'entrée mémoire d'un objet vu pour la première fois.

        Args:
            detection: Première détection de l'objet.
            wall_time: Horodatage réel.

        Returns:
            Le `TrackedObject` initialisé.
        """
        obj = TrackedObject(
            track_id=detection.track_id,
            class_name=detection.class_name,
            first_seen=self._video_time,
            last_seen=self._video_time,
            first_seen_wall=wall_time,
            last_seen_wall=wall_time,
            detection=detection,
        )
        obj._push_history(self._video_time, detection.anchor)
        return obj

    def _purge(self) -> None:
        """Supprime les objets absents depuis plus de `max_age_s`."""
        # La rétention absorbe les occlusions courtes — sans elle, quelqu'un
        # passant derrière un poteau remettrait à zéro son chronomètre
        # d'intrusion. On raisonne en **temps vidéo** et non en nombre de frames
        # manquées : 3 secondes de tolérance gardent le même sens que la source
        # tourne à 25 ou à 10 images par seconde.
        expired = [
            track_id
            for track_id, obj in self._objects.items()
            if (self._video_time - obj.last_seen) > self._max_age_s
        ]
        for track_id in expired:
            del self._objects[track_id]

        if expired:
            logger.debug("Objets purgés (absents depuis > %.1fs) : %s", self._max_age_s, expired)

    def get(self, track_id: int) -> TrackedObject | None:
        """Retourne un objet suivi par son identifiant, ou `None`."""
        return self._objects.get(track_id)

    def active(self) -> list[TrackedObject]:
        """Tous les objets confirmés actuellement en mémoire, triés par ID."""
        return sorted(
            (obj for obj in self._objects.values() if obj.is_confirmed),
            key=lambda obj: obj.track_id,
        )

    def by_class(self, class_name: str) -> list[TrackedObject]:
        """Objets confirmés d'une classe donnée.

        Args:
            class_name: Nom de la classe recherchée.

        Returns:
            Les objets correspondants.
        """
        return [obj for obj in self.active() if obj.class_name == class_name]

    def nearest(
        self,
        target: TrackedObject,
        *,
        class_names: Sequence[str],
        radius_px: float,
    ) -> TrackedObject | None:
        """Cherche l'objet le plus proche parmi certaines classes.

        Servira à la règle « objet abandonné » (étape 8) : un sac n'est abandonné
        que si aucune personne ne se trouve à proximité.

        La distance est mesurée entre les **points d'appui** et non entre les
        centres de boîte : le centre d'une personne debout se situe à hauteur de
        torse, celui d'un sac posé au sol à 20 cm du sol, ce qui gonflerait
        artificiellement la distance entre un sac et son propriétaire.

        Args:
            target: Objet de référence.
            class_names: Classes acceptables pour le voisin.
            radius_px: Rayon de recherche en pixels.

        Returns:
            L'objet le plus proche dans le rayon, ou `None`.
        """
        accepted = set(class_names)
        target_x, target_y = target.position

        closest: TrackedObject | None = None
        closest_distance = radius_px

        for obj in self._objects.values():
            if obj.track_id == target.track_id or not obj.is_confirmed:
                continue
            if obj.class_name not in accepted:
                continue

            x, y = obj.position
            distance = math.hypot(x - target_x, y - target_y)
            if distance <= closest_distance:
                closest = obj
                closest_distance = distance

        return closest

    @property
    def untracked(self) -> list[Detection]:
        """Détections de la dernière frame sans identifiant confirmé.

        Utile pour l'affichage (montrer toutes les boîtes) sans polluer la
        mémoire temporelle.
        """
        return list(self._untracked)

    @property
    def video_time(self) -> float:
        """Temps vidéo de la dernière frame traitée, en secondes."""
        return self._video_time

    def counts(self) -> dict[str, int]:
        """Nombre d'objets **distincts** actuellement suivis, par classe.

        Remplace le comptage par frame du prototype : une personne présente
        30 secondes est comptée une fois, pas 750 fois.
        """
        # `count_by_class` agrège n'importe quel objet portant un `class_name` :
        # le comptage par classe est écrit une seule fois, pour les détections
        # comme pour les objets suivis.
        return count_by_class(self.active())

    def reset(self) -> None:
        """Vide la mémoire et réinitialise le tracker du détecteur.

        À appeler entre deux sources vidéo.
        """
        self._objects.clear()
        self._untracked = []
        self._video_time = 0.0
        self._detector.reset()

    def __len__(self) -> int:
        """Nombre d'objets en mémoire, confirmés ou non."""
        return len(self._objects)

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        return (
            f"Tracker(objets={len(self._objects)}, confirmes={len(self.active())}, "
            f"t={self._video_time:.2f}s)"
        )
