"""Source d'images unifiée — fichier, webcam ou flux réseau.

Rôle : fournir au pipeline un flux de frames horodatées, **sans qu'il ait à
savoir d'où elles viennent**. `Detector` et `Tracker` reçoivent une image et un
temps ; que celle-ci sorte d'un `.mp4`, d'une webcam USB ou d'une caméra IP en
RTSP ne change rien pour eux.

Sources finies et sources infinies
----------------------------------
La distinction structurante n'est pas le protocole mais la **terminaison** :

* Un **fichier** a une fin, un nombre d'images connu, et se rejoue à l'identique.
  L'analyse est reproductible : deux exécutions produisent les mêmes incidents.
* Une **caméra** n'a pas de fin. L'arrêt vient forcément de l'extérieur — un
  opérateur, une durée maximale — jamais de la source elle-même. Elle ne se
  rejoue pas : ce qui n'a pas été traité est perdu.

Deux horloges, et pourquoi le direct change la réponse
-------------------------------------------------------
Sur un fichier, le temps métier se calcule sans ambiguïté : `frame_index / fps`.
C'est ce qui rend l'analyse indépendante de la vitesse de la machine.

En direct, cette formule devient **fausse**, et c'est le point le plus subtil de
ce module. Si le traitement d'une image prend 140 ms alors que la caméra en
produit une toutes les 40 ms, trois images arrivent pendant qu'on travaille. Il
faut les sauter (voir plus bas) — mais dès lors le compteur d'images ne suit plus
le temps qui passe : 100 images traitées ne signifient plus 4 secondes. Un objet
présent depuis 30 secondes réelles paraîtrait n'avoir que 12 secondes d'âge, et
aucune règle de durée ne serait fiable.

En direct, **l'horloge murale devient donc l'horloge métier** : le temps vidéo
est le temps écoulé depuis le début de la capture. Ce n'est pas un renoncement au
principe des deux horloges, c'en est l'application correcte — sur un fichier, le
temps vidéo est une quantité *reconstruite* ; en direct, il est *observé*.

Stratégie de cadence en direct
-------------------------------
Accumuler les images en retard est le piège classique du RTSP : le tampon du
pilote se remplit, et l'opérateur finit par regarder une scène vieille de
plusieurs minutes tout en croyant voir le direct. Deux mesures se cumulent :

1. `CAP_PROP_BUFFERSIZE = 1` demande au pilote de ne conserver que la dernière
   image. Tous les backends ne l'honorent pas, d'où la seconde mesure.
2. Avant chaque lecture, on estime le nombre d'images arrivées pendant le
   traitement précédent (`temps écoulé × fps`) et on les **écarte avec `grab()`**,
   qui récupère l'image sans la décoder — bien moins coûteux que `read()`.

La règle est donc : en direct, on préfère **perdre des images plutôt que du
temps**. Sur un fichier, l'arbitrage s'inverse — aucune image n'est sautée, la
reproductibilité prime.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

import config
from sentinel.exceptions import SourceDisconnectedError, VideoSourceError

logger = logging.getLogger(__name__)

SourceKind = config.SourceKind


@dataclass(frozen=True, slots=True)
class Frame:
    """Une image et ses deux horodatages.

    Attributes:
        image: Image BGR (convention OpenCV).
        index: Numéro de l'image **traitée** depuis le début de la capture. En
            direct, il ne compte pas les images sautées : ce n'est donc pas une
            mesure du temps, seulement un compteur.
        video_time: Temps métier en secondes. Reconstruit (`index / fps`) sur un
            fichier, observé à l'horloge murale en direct.
        wall_time: Horodatage réel, utilisé pour dater les rapports et évaluer
            la règle « hors horaires ».
        dropped: Nombre d'images écartées juste avant celle-ci pour rattraper le
            retard. Reste à 0 sur un fichier.
    """

    image: np.ndarray
    index: int
    video_time: float
    wall_time: datetime
    dropped: int = 0


def detect_kind(target: str | int | Path) -> SourceKind:
    """Devine la nature d'une source à partir de sa désignation.

    Args:
        target: Chemin de fichier, index de webcam, ou URL de flux.

    Returns:
        Le `SourceKind` correspondant.
    """
    if isinstance(target, int):
        return SourceKind.WEBCAM
    text = str(target)
    if text.isdigit():
        return SourceKind.WEBCAM
    if "://" in text:
        return SourceKind.STREAM
    return SourceKind.FILE


class VideoSource:
    """Flux d'images unifié, quelle que soit son origine.

    S'utilise comme un gestionnaire de contexte, ce qui garantit la libération
    de la capture — indispensable sous Windows, où un fichier resterait
    verrouillé et une webcam allumée jusqu'à l'arrêt du processus :

        >>> with VideoSource.from_webcam(0) as source:
        ...     for frame in source.frames(should_stop=arret_demande):
        ...         traiter(frame.image, frame.video_time)
    """

    def __init__(
        self,
        target: str | int | Path,
        *,
        kind: SourceKind | None = None,
        settings: config.SourceConfig | None = None,
        capture_factory: Callable[..., object] | None = None,
    ) -> None:
        """Prépare la source sans l'ouvrir.

        Args:
            target: Chemin, index de webcam ou URL de flux.
            kind: Nature de la source. `None` = déduite de `target`.
            settings: Réglages d'ouverture. `None` = `config.SOURCE`.
            capture_factory: Fabrique de capture, injectable pour les tests.
                `None` = `cv2.VideoCapture`.
        """
        self.settings = settings or config.SOURCE
        self.kind = kind or detect_kind(target)
        self.target: str | int = (
            int(target) if self.kind is SourceKind.WEBCAM else str(target)
        )
        self._capture_factory = capture_factory or cv2.VideoCapture

        self._capture = None
        self._fps: float = config.VIDEO.default_fps
        self._total_frames: int = 0
        self._index: int = 0
        self._started_at: float = 0.0
        self._started_wall: datetime | None = None
        self._last_read_at: float = 0.0
        self._stopped: bool = False

    # -- Constructeurs de confort --------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "VideoSource":
        """Source finie : un fichier vidéo."""
        return cls(path, kind=SourceKind.FILE, **kwargs)

    @classmethod
    def from_webcam(cls, index: int | None = None, **kwargs) -> "VideoSource":
        """Source infinie : une webcam locale."""
        settings = kwargs.get("settings") or config.SOURCE
        return cls(
            settings.webcam_index if index is None else index,
            kind=SourceKind.WEBCAM,
            **kwargs,
        )

    @classmethod
    def from_stream(cls, url: str, **kwargs) -> "VideoSource":
        """Source infinie : un flux réseau RTSP ou HTTP."""
        return cls(url, kind=SourceKind.STREAM, **kwargs)

    @classmethod
    def from_settings(cls, settings: config.SourceConfig | None = None, **kwargs) -> "VideoSource":
        """Source décrite par la configuration.

        Args:
            settings: Réglages à utiliser. `None` = `config.SOURCE`.

        Returns:
            La source correspondant à `settings.kind`.

        Raises:
            VideoSourceError: Si la désignation attendue par le type est vide.
        """
        settings = settings or config.SOURCE
        if settings.kind is SourceKind.WEBCAM:
            return cls(settings.webcam_index, kind=SourceKind.WEBCAM, settings=settings, **kwargs)
        if settings.kind is SourceKind.STREAM:
            if not settings.stream_url:
                raise VideoSourceError(
                    "Aucune URL de flux renseignée (config.SOURCE.stream_url)."
                )
            return cls(settings.stream_url, kind=SourceKind.STREAM, settings=settings, **kwargs)
        if settings.file_path is None:
            raise VideoSourceError(
                "Aucun fichier vidéo renseigné (config.SOURCE.file_path)."
            )
        return cls(settings.file_path, kind=SourceKind.FILE, settings=settings, **kwargs)

    # -- Cycle de vie ---------------------------------------------------------

    def open(self) -> "VideoSource":
        """Ouvre la capture et lit ses caractéristiques.

        Returns:
            La source elle-même, pour l'enchaînement.

        Raises:
            VideoSourceError: Si la source ne peut pas être ouverte.
        """
        capture = self._new_capture()
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise VideoSourceError(self._unreachable_message())

        self._capture = capture
        self._read_properties()
        self._index = 0
        self._stopped = False
        self._started_at = time.monotonic()
        self._started_wall = datetime.now()
        self._last_read_at = self._started_at

        logger.info(
            "Source %s ouverte : %s (%.1f fps%s)",
            self.kind.value,
            self._describe_target(),
            self._fps,
            f", {self._total_frames} images" if self._total_frames else ", flux continu",
        )
        return self

    def close(self) -> None:
        """Libère la capture. Idempotent."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def stop(self) -> None:
        """Demande l'arrêt de l'itération en cours.

        Un flux en direct n'a pas de fin : l'arrêt vient nécessairement de
        l'extérieur. Cette méthode est le point d'entrée de cette demande.
        """
        self._stopped = True

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- Caractéristiques -----------------------------------------------------

    @property
    def is_live(self) -> bool:
        """True si la source est infinie (webcam ou flux réseau)."""
        return self.kind.is_live

    @property
    def label(self) -> str:
        """Désignation lisible de la source, **sans identifiants**.

        Exposée publiquement parce qu'un rapport doit dire ce qu'il a analysé.
        Un rapport se transmet : le mot de passe d'une URL RTSP ne doit y figurer
        ni en clair, ni dans les journaux.
        """
        return self._describe_target()

    @property
    def fps(self) -> float:
        """Cadence de la source, en images par seconde."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Nombre total d'images, ou 0 si la source est infinie."""
        return self._total_frames

    @property
    def is_open(self) -> bool:
        """True si la capture est ouverte."""
        return self._capture is not None and self._capture.isOpened()

    def progress(self) -> float | None:
        """Avancement dans une source finie, entre 0 et 1.

        Returns:
            La fraction traitée, ou `None` pour une source infinie — un direct
            n'a pas d'avancement, seulement une durée.
        """
        if not self._total_frames:
            return None
        return min(1.0, self._index / self._total_frames)

    # -- Lecture --------------------------------------------------------------

    def read(self) -> Frame | None:
        """Lit la prochaine image exploitable.

        Returns:
            La `Frame` suivante, ou `None` quand la source est épuisée (fichier
            terminé) ou l'arrêt demandé.

        Raises:
            VideoSourceError: Si la source n'est pas ouverte.
            SourceDisconnectedError: Si un flux en direct reste injoignable après
                toutes les tentatives de reconnexion.
        """
        if self._capture is None:
            raise VideoSourceError("Source non ouverte : appelez open() d'abord.")
        if self._stopped:
            return None

        dropped = self._skip_late_frames()

        ok, image = self._capture.read()
        if not ok or image is None:
            if not self.is_live:
                return None  # fin de fichier : terminaison normale
            image = self._reconnect_and_read()
            if image is None:
                return None

        self._last_read_at = time.monotonic()
        frame = Frame(
            image=image,
            index=self._index,
            video_time=self._video_time(),
            wall_time=datetime.now(),
            dropped=dropped,
        )
        self._index += 1
        return frame

    def frames(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
        max_seconds: float | None = None,
    ) -> Iterator[Frame]:
        """Itère sur les images jusqu'à épuisement, arrêt ou durée maximale.

        Args:
            should_stop: Fonction consultée avant chaque image. Retourner True
                interrompt proprement l'itération — c'est le bouton « Arrêter »
                de l'interface, et le seul moyen de terminer un direct.
            max_seconds: Durée maximale de **temps vidéo** à traiter. Garde-fou
                indispensable sur un flux continu.

        Yields:
            Les images successives.
        """
        while True:
            if should_stop is not None and should_stop():
                logger.info("Arrêt demandé après %d images.", self._index)
                return

            frame = self.read()
            if frame is None:
                return

            if max_seconds is not None and frame.video_time > max_seconds:
                logger.info("Durée maximale atteinte (%.0f s de vidéo).", max_seconds)
                return

            yield frame

    # -- Détails d'implémentation --------------------------------------------

    def _new_capture(self):
        """Construit une capture OpenCV avec les délais configurés.

        Returns:
            L'objet capture, ou `None` si sa construction a échoué.
        """
        try:
            capture = self._capture_factory(self.target)
        except Exception:
            logger.exception("Construction de la capture impossible : %s", self._describe_target())
            return None

        # Les délais ne concernent que les flux réseau ; les appliquer à un
        # fichier n'a pas de sens et certains backends s'en plaignent.
        if self.kind is SourceKind.STREAM:
            self._try_set(capture, cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.settings.open_timeout_ms)
            self._try_set(capture, cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.settings.read_timeout_ms)
        if self.is_live:
            # « Ne me garde que la dernière image ». Best effort : plusieurs
            # backends ignorent silencieusement cette propriété, d'où le
            # rattrapage explicite de `_skip_late_frames`.
            self._try_set(capture, cv2.CAP_PROP_BUFFERSIZE, self.settings.buffer_size)
        return capture

    @staticmethod
    def _try_set(capture, prop: int, value: float) -> None:
        """Applique une propriété de capture sans jamais échouer.

        Toutes les propriétés ne sont pas gérées par tous les backends ; un refus
        n'est pas une erreur, seulement une capacité absente.
        """
        try:
            capture.set(prop, value)
        except Exception:  # pragma: no cover - dépend du backend
            logger.debug("Propriété %s non prise en charge par le backend.", prop)

    def _read_properties(self) -> None:
        """Lit cadence et longueur, avec repli sur la configuration."""
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        # Un FPS nul ou absurde est fréquent sur webcam et sur certains flux
        # RTSP : sans repli, tout le temps métier serait faux. Les bornes de
        # vraisemblance sont une valeur métier, donc dans `config.VIDEO`.
        self._fps = config.VIDEO.credible_fps(fps)

        total = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Un direct annonce parfois un nombre d'images fantaisiste : seule une
        # source finie a une longueur.
        self._total_frames = total if (total > 0 and not self.is_live) else 0

    def _video_time(self) -> float:
        """Temps métier de l'image courante, en secondes.

        Returns:
            Le temps écoulé depuis le début de la capture en direct, le rapport
            `index / fps` sur un fichier.
        """
        if self.is_live:
            return time.monotonic() - self._started_at
        return self._index / self._fps

    def _skip_late_frames(self) -> int:
        """Écarte les images accumulées pendant le traitement précédent.

        Returns:
            Le nombre d'images écartées.
        """
        if not self.is_live or not self.settings.drop_late_frames:
            return 0

        elapsed = time.monotonic() - self._last_read_at
        # `- 1` : l'image qu'on va lire juste après est celle qu'on garde.
        late = int(elapsed * self._fps) - 1
        if late <= 0:
            return 0

        late = min(late, self.settings.max_dropped_frames)
        for _ in range(late):
            # `grab()` récupère l'image sans la décoder : c'est précisément ce
            # qu'il faut pour vider un tampon à moindre coût.
            if not self._capture.grab():
                break
        logger.debug("%d image(s) écartée(s) pour rattraper le direct.", late)
        return late

    def _reconnect_and_read(self) -> np.ndarray | None:
        """Tente de rétablir un flux interrompu, puis relit une image.

        Returns:
            L'image obtenue après reconnexion, ou `None` si l'arrêt a été
            demandé entre-temps.

        Raises:
            SourceDisconnectedError: Si toutes les tentatives échouent.
        """
        for attempt in range(1, self.settings.reconnect_attempts + 1):
            if self._stopped:
                return None

            logger.warning(
                "Flux interrompu (%s) ; reconnexion %d/%d dans %.1f s.",
                self._describe_target(),
                attempt,
                self.settings.reconnect_attempts,
                self.settings.reconnect_delay_s,
            )
            time.sleep(self.settings.reconnect_delay_s)

            self.close()
            capture = self._new_capture()
            if capture is None or not capture.isOpened():
                if capture is not None:
                    capture.release()
                continue

            self._capture = capture
            ok, image = capture.read()
            if ok and image is not None:
                logger.info("Flux rétabli après %d tentative(s).", attempt)
                # Le compteur d'images repart de là où il en était, mais le temps
                # métier suit l'horloge murale : la coupure est donc comptée dans
                # les durées, ce qui est le comportement voulu — l'objet est resté
                # dans la scène pendant que le réseau était absent.
                self._last_read_at = time.monotonic()
                return image

        raise SourceDisconnectedError(
            f"Flux perdu et non rétabli après {self.settings.reconnect_attempts} "
            f"tentative(s) : {self._describe_target()}. Vérifiez la caméra et le réseau."
        )

    def _describe_target(self) -> str:
        """Désignation lisible de la source, **sans mot de passe**.

        Une URL RTSP porte souvent des identifiants ; les journaux et les
        messages d'erreur ne doivent jamais les recopier.
        """
        if self.kind is SourceKind.WEBCAM:
            return f"webcam n°{self.target}"
        text = str(self.target)
        if "@" in text and "://" in text:
            scheme, rest = text.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
        return text

    def _unreachable_message(self) -> str:
        """Message d'erreur adapté au type de source."""
        if self.kind is SourceKind.WEBCAM:
            return (
                f"Webcam n°{self.target} inaccessible. Vérifiez qu'aucune autre "
                "application ne l'utilise et que l'accès caméra est autorisé."
            )
        if self.kind is SourceKind.STREAM:
            return (
                f"Flux injoignable : {self._describe_target()}. Vérifiez l'URL, "
                "les identifiants, et que la caméra est accessible depuis ce poste."
            )
        return (
            f"Fichier vidéo illisible : {self.target}. Vérifiez le chemin et que "
            "le format fait partie de "
            f"{', '.join(config.VIDEO.allowed_extensions)}."
        )

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        état = "ouverte" if self.is_open else "fermée"
        return (
            f"VideoSource({self.kind.value}, {self._describe_target()!r}, "
            f"{état}, {self._fps:.1f} fps)"
        )
