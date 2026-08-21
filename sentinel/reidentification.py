"""Ré-association bon marché des pistes perdues — occlusions de 3 à 15 secondes.

Le problème
------------
Au-delà de `TRACKING.max_age_s`, une piste est purgée. L'objet qui réapparaît
reçoit un **nouvel identifiant**, et son chronomètre repart à zéro. Une personne
qui rôde depuis cinquante secondes et passe cinq secondes derrière un camion
redevient « vue à l'instant » : la règle de rôdage ne se déclenche jamais.

Pourquoi ce module est le dernier, et pourquoi il est désactivé par défaut
---------------------------------------------------------------------------
Il contredit une limite que le README revendique (§ 7) : ByteTrack n'utilise pas
l'apparence visuelle. Surtout, **son mode de panne est pire que le problème
qu'il résout**. Un chronomètre remis à zéro fait manquer un incident — c'est un
faux négatif, visible et corrigeable. Une ré-association erronée **fusionne deux
personnes en une seule piste** : le rapport affirme alors qu'une personne est
restée quarante minutes là où deux se sont succédé. Un document présenté comme
opposable énonce un fait faux.

Entre manquer un incident et en fabriquer un, un système de sécurité doit choisir
le premier. D'où deux décisions :

1. **Le doute vaut refus.** Chaque condition doit être franchie ; à la moindre
   ambiguïté — pas d'image, deux candidats également plausibles, un écart au
   seuil — on ne ré-associe pas.
2. **Désactivé par défaut** (`REID.enabled`). L'activer est un choix
   d'exploitation, pris en connaissance du compromis, pas un comportement subi.

Les trois conditions, et pourquoi elles se cumulent
-----------------------------------------------------
Aucune ne suffit seule ; chacune écarte un faux positif que les autres laissent
passer.

* **Position prédite.** L'objet devrait réapparaître là où son mouvement le
  menait. Écarte les candidats de l'autre bout de l'image — mais deux personnes
  marchant côte à côte satisfont toutes deux ce critère.
* **Taille apparente.** Un objet ne change pas brutalement de profondeur.
  Écarte le passant du premier plan confondu avec la silhouette du fond — mais
  deux personnes à la même distance ont la même taille.
* **Histogramme de couleur.** Un manteau rouge ne redevient pas bleu. C'est la
  seule condition qui porte sur l'**apparence**, et donc la seule qui distingue
  deux personnes également placées et également grandes. Elle est aussi la plus
  fragile — un changement d'éclairage la fait chuter — d'où son emploi en
  **confirmation** d'un candidat déjà retenu géométriquement, jamais seule.

Aucune dépendance : `cv2.calcHist` et `cv2.compareHist` font partie d'OpenCV,
déjà obligatoire pour la lecture vidéo et le dessin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

import config
from sentinel.detection import Detection

logger = logging.getLogger(__name__)


def colour_signature(
    image: np.ndarray | None,
    box: tuple[float, float, float, float],
    settings: config.ReidConfig | None = None,
) -> np.ndarray | None:
    """Histogramme teinte-saturation normalisé de la région d'une boîte.

    L'espace HSV plutôt que BGR : la teinte est bien plus stable que l'intensité
    face aux variations d'éclairage, qui sont la première cause de dérive entre
    deux apparitions d'un même objet.

    Args:
        image: Image BGR complète. `None` rend `None`.
        box: Boîte englobante `(x1, y1, x2, y2)` en pixels.
        settings: Réglages. `None` = `config.REID`.

    Returns:
        L'histogramme normalisé, ou `None` si la région est inexploitable — une
        boîte dégénérée, hors cadre, ou une image absente. `None` **signifie
        refus** en aval : sans signature, la troisième condition ne peut pas être
        vérifiée, et une condition non vérifiée n'est pas une condition remplie.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None

    reglages = settings or config.REID
    hauteur, largeur = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(largeur, x2), min(hauteur, y2)
    if x2 - x1 < reglages.min_patch_px or y2 - y1 < reglages.min_patch_px:
        return None

    region = image[y1:y2, x1:x2]
    try:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        histogramme = cv2.calcHist(
            [hsv], [0, 1], None, list(reglages.histogram_bins), [0, 180, 0, 256]
        )
        cv2.normalize(histogramme, histogramme, 0.0, 1.0, cv2.NORM_MINMAX)
        return histogramme
    except cv2.error:  # pragma: no cover - dépend du backend
        logger.debug("Signature couleur impossible sur la région %s", (x1, y1, x2, y2))
        return None


@dataclass(slots=True)
class LostTrack:
    """Une piste purgée, gardée quelque temps au cas où elle réapparaîtrait.

    Attributes:
        track_id: Identifiant qu'elle portait.
        class_name: Classe majoritaire observée.
        position: Dernier point d'appui connu, en pixels.
        velocity: Vitesse estimée `(dx, dy)` en pixels par seconde, pour prédire
            où l'objet devrait réapparaître.
        scale_px: Hauteur apparente à la disparition.
        signature: Histogramme de couleur, ou `None` si la capture a échoué.
        lost_at: Temps vidéo de la disparition.
        memory: État à restaurer si la piste est retrouvée — chronomètres de
            zone, votes de classe, porteur présumé, anti-rebond.
    """

    track_id: int
    class_name: str
    position: tuple[float, float]
    velocity: tuple[float, float]
    scale_px: float
    signature: np.ndarray | None
    lost_at: float
    memory: dict[str, object] = field(default_factory=dict)

    def predicted_position(self, video_time: float) -> tuple[float, float]:
        """Où l'objet devrait se trouver, s'il a poursuivi son mouvement.

        Args:
            video_time: Instant considéré.

        Returns:
            La position prédite, en pixels.
        """
        ecoule = max(0.0, video_time - self.lost_at)
        return (
            self.position[0] + self.velocity[0] * ecoule,
            self.position[1] + self.velocity[1] * ecoule,
        )


class ReidentificationBuffer:
    """Tampon des pistes récemment perdues, et décision de ré-association.

        >>> tampon = ReidentificationBuffer()
        >>> tampon.remember(obj, video_time=40.0, frame=image)
        >>> tampon.match(detection, video_time=46.0, frame=image)
        LostTrack(track_id=7, ...)
    """

    def __init__(self, settings: config.ReidConfig | None = None) -> None:
        """Prépare un tampon vide.

        Args:
            settings: Réglages. `None` = `config.REID`.
        """
        self.settings = settings or config.REID
        self._lost: list[LostTrack] = []
        self.refusals: int = 0
        self.matches: int = 0

    # -- Mémorisation ---------------------------------------------------------

    def remember(self, lost: LostTrack) -> None:
        """Range une piste perdue dans le tampon.

        Args:
            lost: Piste à mémoriser.
        """
        if not self.settings.enabled:
            return
        self._lost.append(lost)

    def expire(self, video_time: float) -> int:
        """Oublie les pistes trop anciennes pour être encore plausibles.

        Args:
            video_time: Temps vidéo courant.

        Returns:
            Le nombre de pistes oubliées.
        """
        avant = len(self._lost)
        limite = self.settings.max_gap_s
        self._lost = [
            piste for piste in self._lost if video_time - piste.lost_at <= limite
        ]
        return avant - len(self._lost)

    # -- Décision -------------------------------------------------------------

    def match(
        self, detection: Detection, video_time: float, frame: np.ndarray | None
    ) -> LostTrack | None:
        """Cherche la piste perdue que cette détection pourrait prolonger.

        **Le doute vaut refus.** Toute condition non franchie, toute information
        manquante et toute ambiguïté entre deux candidats font rendre `None`.

        Args:
            detection: Détection portant un identifiant neuf.
            video_time: Temps vidéo de la frame.
            frame: Image courante, pour la signature couleur. `None` = refus, car
                la troisième condition serait invérifiable.

        Returns:
            La piste retrouvée, ou `None`.
        """
        if not self.settings.enabled or not self._lost:
            return None

        signature = colour_signature(frame, detection.xyxy, self.settings)
        if signature is None:
            # Sans apparence, il ne reste que la géométrie — et deux personnes
            # côte à côte la satisfont toutes deux. On refuse.
            self.refusals += 1
            return None

        retenus = [
            (piste, score)
            for piste, score in (
                (piste, self._score(piste, detection, video_time, signature))
                for piste in self._lost
            )
            if score is not None
        ]
        if not retenus:
            self.refusals += 1
            return None

        retenus.sort(key=lambda couple: -couple[1])
        meilleur, score = retenus[0]

        if len(retenus) > 1:
            _, suivant = retenus[1]
            # Deux candidats presque aussi plausibles : c'est exactement la
            # situation où une erreur fusionnerait deux personnes. On refuse
            # plutôt que de départager au hasard.
            if score - suivant < self.settings.min_margin:
                logger.debug(
                    "Ré-association refusée : deux candidats à %.3f et %.3f.", score, suivant
                )
                self.refusals += 1
                return None

        self._lost.remove(meilleur)
        self.matches += 1
        logger.info(
            "Piste #%s ré-associée à la détection #%s (similarité %.2f).",
            meilleur.track_id,
            detection.track_id,
            score,
        )
        return meilleur

    def _score(
        self,
        lost: LostTrack,
        detection: Detection,
        video_time: float,
        signature: np.ndarray,
    ) -> float | None:
        """Évalue une piste candidate contre les trois conditions.

        Args:
            lost: Piste perdue examinée.
            detection: Détection à rattacher.
            video_time: Temps vidéo courant.
            signature: Histogramme de la détection.

        Returns:
            La similarité de couleur si **toutes** les conditions sont
            franchies, sinon `None`. Rendre la similarité plutôt qu'un booléen
            permet de départager plusieurs candidats géométriquement valides par
            le seul critère qui porte sur l'apparence.
        """
        reglages = self.settings

        # -- Condition 0 : même classe, et un écart de temps plausible --------
        if lost.class_name != detection.class_name:
            return None
        ecart = video_time - lost.lost_at
        if not (reglages.min_gap_s <= ecart <= reglages.max_gap_s):
            return None

        # -- Condition 1 : position prédite -----------------------------------
        prevue = lost.predicted_position(video_time)
        observee = detection.anchor
        distance = float(np.hypot(prevue[0] - observee[0], prevue[1] - observee[1]))
        # Le rayon est relatif à la taille apparente, comme tous les seuils de
        # distance du projet : 200 px sont un pas au premier plan et une
        # traversée au fond du champ.
        if distance > reglages.max_distance_ratio * max(1.0, lost.scale_px):
            return None

        # -- Condition 2 : taille apparente -----------------------------------
        taille = max(1.0, detection.height)
        rapport = taille / max(1.0, lost.scale_px)
        if not (1.0 / reglages.max_scale_ratio <= rapport <= reglages.max_scale_ratio):
            return None

        # -- Condition 3 : apparence ------------------------------------------
        if lost.signature is None:
            return None
        similarite = float(cv2.compareHist(lost.signature, signature, cv2.HISTCMP_CORREL))
        if similarite < reglages.min_similarity:
            return None

        return similarite

    # -- Consultation ---------------------------------------------------------

    @property
    def pending(self) -> list[LostTrack]:
        """Pistes actuellement en attente d'une éventuelle réapparition."""
        return list(self._lost)

    def reset(self) -> None:
        """Vide le tampon. À appeler entre deux sessions."""
        self._lost.clear()
        self.refusals = 0
        self.matches = 0

    def __len__(self) -> int:
        """Nombre de pistes en attente."""
        return len(self._lost)

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        etat = "actif" if self.settings.enabled else "désactivé"
        return (
            f"ReidentificationBuffer({etat}, {len(self._lost)} en attente, "
            f"{self.matches} retrouvée(s), {self.refusals} refus)"
        )
