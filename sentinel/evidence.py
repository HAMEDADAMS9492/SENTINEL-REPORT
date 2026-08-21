"""Capture des images justificatives — écrire la preuve sur disque.

Pourquoi ce module existe
--------------------------
`events.py` faisait deux choses. Son propre docstring affirmait « le moteur ne
détecte rien et ne dessine rien », et cent vingt lignes plus bas il ouvrait des
fichiers, dessinait des rectangles, recadrait des images et écrivait du JPEG.

Ce n'est pas un défaut d'esthétique. Les deux responsabilités ont des raisons de
changer différentes : le moteur change quand une règle métier évolue, la capture
change quand la politique de conservation, le format ou l'anonymisation évoluent.
Les mélanger obligeait à relire du code de règles pour ajouter un floutage.

Ce que la séparation débloque immédiatement
---------------------------------------------
La rétention (`EVIDENCE.retention_days`) et l'accroche de floutage s'accrochent
**ici**, à un endroit qui ne connaît ni les règles ni les scores. Un seul point
de passage garantit qu'aucune image ne s'écrit sans passer par la politique
retenue.

Ce que ce module ne fait pas
------------------------------
Il ne décide pas qu'un incident a lieu et ne calcule aucun score : il reçoit un
`Event` déjà constitué, et rend un chemin. Le flux reste unidirectionnel.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

import config
from sentinel.detection import Detection
from sentinel.events import Event
from sentinel.tracker import TrackedObject

logger = logging.getLogger(__name__)

# Signature d'une fonction d'anonymisation. Elle reçoit l'image, l'objet en
# cause et les autres objets de la frame ; elle rend l'image traitée.
Anonymiser = Callable[[np.ndarray, TrackedObject, Sequence[TrackedObject]], np.ndarray]


def blur_bystanders(
    image: np.ndarray, subject: TrackedObject, others: Sequence[TrackedObject]
) -> np.ndarray:
    """Floute les personnes **autres** que l'objet déclencheur.

    Accroche fournie, désactivée par défaut (`EVIDENCE.blur_bystanders`).

    Ce qu'elle règle, et ce qu'elle ne règle pas
    ---------------------------------------------
    Un rapport d'incident circule : il est relu, transmis, parfois archivé. Rien
    n'oblige à y faire figurer le visage des passants, qui ne sont concernés par
    rien. Flouter ce qui n'est pas en cause est la mesure de minimisation la plus
    simple, et elle ne coûte rien à la valeur probante — la boîte de l'objet
    déclencheur, elle, reste intacte.

    Ce n'est **pas** de l'anonymisation au sens réglementaire : le floutage porte
    sur la boîte entière, pas sur les visages, et il ne s'applique qu'aux objets
    que le détecteur a vus. Une personne manquée par le modèle n'est pas floutée.
    Le présenter autrement serait une promesse que le code ne tient pas.

    Args:
        image: Image à traiter (modifiée en place).
        subject: Objet en cause, laissé intact.
        others: Autres objets de la frame.

    Returns:
        L'image traitée.
    """
    settings = config.EVIDENCE
    hauteur, largeur = image.shape[:2]

    for autre in others:
        if autre.track_id == subject.track_id:
            continue
        if autre.class_name not in settings.blur_classes:
            continue

        x1, y1, x2, y2 = autre.detection.as_int_box()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(largeur, x2), min(hauteur, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        region = image[y1:y2, x1:x2]
        # Le noyau doit être impair et non nul : un flou dimensionné en fraction
        # de la boîte reste efficace quelle que soit la taille apparente.
        noyau = max(3, int(min(x2 - x1, y2 - y1) * settings.blur_strength) | 1)
        image[y1:y2, x1:x2] = cv2.GaussianBlur(region, (noyau, noyau), 0)

    return image


class EvidenceWriter:
    """Écrit les captures justificatives, applique la politique de conservation.

        >>> ecrivain = EvidenceWriter()
        >>> ecrivain.purge_expired()
        3
        >>> event = ecrivain.attach(event, frame, obj, tracker.active())
    """

    def __init__(
        self,
        settings: config.EvidenceConfig | None = None,
        anonymiser: Anonymiser | None = None,
    ) -> None:
        """Prépare l'écrivain.

        Args:
            settings: Réglages de capture. `None` = `config.EVIDENCE`.
            anonymiser: Fonction d'anonymisation. `None` = `blur_bystanders`
                quand `EVIDENCE.blur_bystanders` est actif, sinon aucune.
        """
        self.settings = settings or config.EVIDENCE
        self._anonymiser = anonymiser

    # -- Écriture -------------------------------------------------------------

    def attach(
        self,
        event: Event,
        frame: np.ndarray | None,
        obj: TrackedObject,
        scene: Sequence[TrackedObject] = (),
    ) -> Event:
        """Retourne une copie de l'incident enrichie du chemin de sa preuve.

        `Event` étant immuable, on ne peut pas lui affecter le chemin après
        coup : on reconstruit l'objet. C'est le prix — modeste — de la garantie
        qu'un incident déjà publié ne change jamais.

        Args:
            event: Incident sans preuve.
            frame: Frame au moment de l'incident.
            obj: Objet concerné.
            scene: Autres objets de la frame, pour l'anonymisation.

        Returns:
            L'incident, avec `evidence_path` renseigné si la capture a réussi.
        """
        chemin = self.capture(frame, obj, event, scene)
        if chemin is None:
            return event

        # `replace()` plutôt qu'une reconstruction champ par champ : tout champ
        # ajouté à `Event` serait sinon silencieusement perdu ici — c'est
        # exactement ce qui serait arrivé au score de priorité.
        return replace(event, evidence_path=chemin)

    def capture(
        self,
        frame: np.ndarray | None,
        obj: TrackedObject,
        event: Event,
        scene: Sequence[TrackedObject] = (),
    ) -> Path | None:
        """Écrit la capture justificative sur disque.

        Args:
            frame: Frame au moment de l'incident.
            obj: Objet concerné.
            event: Incident, dont l'horodatage sert au nom de fichier.
            scene: Autres objets de la frame.

        Returns:
            Le chemin du fichier écrit, ou `None` en cas d'échec — une capture
            ratée ne doit pas annuler l'incident.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        settings = self.settings
        try:
            nom = settings.filename_template.format(
                timestamp=event.timestamp.strftime(settings.timestamp_format),
                event_type=event.event_type.value,
                track_id=obj.track_id,
                zone=event.zone_name or "hors-zone",
            )
            destination = settings.directory / nom

            image = frame.copy()
            # L'anonymisation vient **avant** l'annotation : flouter après
            # effacerait le rectangle rouge qui désigne l'objet en cause.
            anonymiser = self._resolve_anonymiser()
            if anonymiser is not None:
                image = anonymiser(image, obj, scene)
            if settings.annotate:
                image = self.annotate(image, obj, event)
            if settings.crop_to_object:
                image = self.crop_around(image, obj.detection, settings.margin_px)

            settings.directory.mkdir(parents=True, exist_ok=True)
            ecrit = cv2.imwrite(
                str(destination),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality],
            )
            if not ecrit:
                logger.warning("Écriture de la preuve refusée par OpenCV : %s", destination)
                return None
            return destination
        except Exception:
            # Un disque plein ou un nom de fichier invalide ne doit pas faire
            # disparaître l'incident : le rapport existera, simplement sans image.
            logger.exception("Capture de preuve impossible pour l'incident %s.", event.event_id)
            return None

    def _resolve_anonymiser(self) -> Anonymiser | None:
        """Fonction d'anonymisation effectivement appliquée.

        Returns:
            Celle qui a été injectée, sinon le floutage par défaut si la
            configuration l'active, sinon `None`.
        """
        if self._anonymiser is not None:
            return self._anonymiser
        return blur_bystanders if self.settings.blur_bystanders else None

    # -- Dessin ---------------------------------------------------------------

    @staticmethod
    def annotate(image: np.ndarray, obj: TrackedObject, event: Event) -> np.ndarray:
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
        couleur = config.EVIDENCE.subject_color
        cv2.rectangle(image, (x1, y1), (x2, y2), couleur, config.UI.box_thickness)

        etiquette = _ascii_label(f"{event.event_type.value} #{obj.track_id}")
        cv2.putText(
            image,
            etiquette,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.EVIDENCE.label_scale,
            couleur,
            config.UI.box_thickness,
            cv2.LINE_AA,
        )
        return image

    @staticmethod
    def crop_around(image: np.ndarray, detection: Detection, margin_px: int) -> np.ndarray:
        """Recadre l'image autour d'une détection, avec une marge.

        Args:
            image: Image source.
            detection: Détection à cadrer.
            margin_px: Marge ajoutée de chaque côté.

        Returns:
            Le recadrage, borné aux limites de l'image.
        """
        hauteur, largeur = image.shape[:2]
        x1, y1, x2, y2 = detection.as_int_box()
        return image[
            max(0, y1 - margin_px) : min(hauteur, y2 + margin_px),
            max(0, x1 - margin_px) : min(largeur, x2 + margin_px),
        ]

    # -- Conservation ---------------------------------------------------------

    def purge_expired(self, *, now: float | None = None) -> int:
        """Supprime les captures plus anciennes que la durée de conservation.

        Pourquoi une purge, et pourquoi au démarrage
        ---------------------------------------------
        Une capture est une image de personnes prise sans leur accord, conservée
        pour un besoin précis et borné dans le temps. Ne jamais l'effacer
        transforme un outil d'analyse en archive permanente, ce qui n'est ni le
        but annoncé, ni défendable.

        Le démarrage est le bon moment : c'est le seul instant où l'on est
        certain qu'aucune analyse n'est en cours, donc qu'aucun fichier examiné
        n'est en train d'être écrit.

        Args:
            now: Horodatage de référence, en secondes epoch. `None` = maintenant.
                Paramétrable pour les tests, qui ne peuvent pas attendre trente
                jours.

        Returns:
            Le nombre de fichiers supprimés.
        """
        jours = self.settings.retention_days
        if jours is None or jours <= 0:
            # `None` = conservation indéfinie. C'est un choix légitime — une
            # instruction judiciaire en cours, par exemple — mais il doit être
            # explicite, pas la valeur par défaut.
            return 0

        limite = (time.time() if now is None else now) - jours * 86_400
        dossier = self.settings.directory
        if not dossier.is_dir():
            return 0

        supprimes = 0
        for chemin in dossier.iterdir():
            if not chemin.is_file() or chemin.name.startswith("."):
                continue
            try:
                if chemin.stat().st_mtime >= limite:
                    continue
                chemin.unlink()
                supprimes += 1
            except OSError:
                # Un fichier verrouillé ou déjà supprimé ne doit pas empêcher le
                # démarrage de l'application.
                logger.warning("Purge impossible pour %s", chemin, exc_info=True)

        if supprimes:
            logger.info(
                "Purge des preuves : %d fichier(s) de plus de %d jour(s) supprimé(s).",
                supprimes,
                jours,
            )
        return supprimes

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        retention = self.settings.retention_days
        duree = "indéfinie" if not retention else f"{retention} j"
        return f"EvidenceWriter({self.settings.directory}, conservation {duree})"
