"""Détection YOLOv8 — le module « perception » de SentinelReport.

Ce module contient toute la logique de détection migrée depuis le prototype
Streamlit, **sans aucune dépendance à Streamlit**. C'est le point central de la
refonte : la logique métier ne doit rien savoir de l'interface qui l'utilise.
On peut ainsi appeler `Detector` depuis Streamlit, depuis un script batch, depuis
un test unitaire ou depuis une future API, sans rien changer.

Répartition des responsabilités avec `tracker.py`
-------------------------------------------------
Ultralytics implémente ByteTrack **à l'intérieur** de l'objet modèle : c'est
`model.track()` qui associe les identifiants, et cet état vit dans le predictor
du modèle. Il serait artificiel de « sortir » cette association du `Detector`,
qui possède le modèle.

Le partage est donc :

* `Detector` (ce module) : possède le modèle, produit des `Detection` — avec ou
  sans `track_id` (association d'identifiants faite par ByteTrack).
* `Tracker` (étape 3) : ne possède pas de modèle. Il consomme les détections
  suivies et maintient la **mémoire temporelle** de chaque objet (première et
  dernière apparition, historique des positions, durée passée dans chaque zone).
  C'est ce dont `events.py` a besoin, et c'est du code métier à nous.

En une phrase : *« l'association d'identifiants est inséparable de l'appel au
modèle, donc elle reste dans le Detector ; la machine à états temporelle est
notre logique, donc elle est dans le Tracker. »*
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

import config
from sentinel.detection import Detection
from sentinel.exceptions import ModelLoadError

if TYPE_CHECKING:  # pragma: no cover - uniquement pour les annotations de type
    from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chargement du modèle (avec cache)
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> str:
    """Détermine le périphérique d'inférence à utiliser.

    Reprend la logique GPU/CPU du prototype, mais sans jamais faire planter
    l'application : si CUDA est demandé et indisponible, on retombe sur le CPU
    avec un avertissement plutôt qu'une exception. `yolov8n.pt` tourne
    correctement sur CPU — une exécution dégradée vaut mieux qu'un crash.

    Args:
        requested: `"auto"`, `"cpu"`, `"mps"` ou un index de GPU (`"0"`).

    Returns:
        L'identifiant de périphérique à passer à Ultralytics.
    """
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA détecté : inférence sur GPU.")
            return "0"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            logger.info("Apple MPS détecté : inférence sur GPU intégré.")
            return "mps"
    except Exception:  # torch absent ou mal installé
        logger.warning("PyTorch indisponible pour la détection du GPU ; repli sur le CPU.")

    logger.info("Aucun accélérateur détecté : inférence sur CPU.")
    return "cpu"


@lru_cache(maxsize=4)
def load_model(weights: str, device: str) -> "YOLO":
    """Charge (et met en cache) un modèle YOLOv8.

    Le prototype utilisait `@st.cache_resource` pour ne pas recharger les poids à
    chaque interaction. On garde l'idée mais avec `functools.lru_cache` : le
    cache devient une propriété du module de détection et non de l'interface, il
    fonctionne donc aussi hors de Streamlit (scripts, tests). `app.py` peut
    toujours l'envelopper dans `@st.cache_resource`, sans conflit.

    Args:
        weights: Chemin ou nom des poids (ex. `"yolov8n.pt"`).
        device: Périphérique déjà résolu (`"cpu"`, `"0"`, `"mps"`).

    Returns:
        L'instance `ultralytics.YOLO` prête à l'emploi.

    Raises:
        ModelLoadError: Si `ultralytics` est absent, si les poids sont
            introuvables ou si le déplacement sur le périphérique échoue.
    """
    # Garde-fou volontaire : `YOLO("models/yolov8m.pt")` ne lève PAS d'erreur si
    # le fichier manque — Ultralytics le télécharge (jusqu'à 130 Mo) à cet
    # emplacement, en pleine exécution. On refuse ce comportement : les poids
    # sont une ressource installée à l'avance par `download_models.py`, jamais
    # récupérée pendant l'analyse d'une vidéo.
    if not Path(weights).is_file():
        raise ModelLoadError(
            f"Fichier de poids introuvable : '{weights}'. Telechargez les modeles "
            "au prealable : python download_models.py"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError(
            "Le paquet 'ultralytics' est introuvable. Installez les dépendances : "
            "pip install -r requirements.txt"
        ) from exc

    try:
        model = YOLO(weights)
        model.to(device)
    except Exception as exc:
        raise ModelLoadError(
            f"Impossible de charger le modèle '{weights}' sur le périphérique "
            f"'{device}' : {exc}"
        ) from exc

    logger.info("Modèle '%s' chargé sur '%s'.", weights, device)
    return model


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class Detector:
    """Encapsule le modèle YOLOv8 et produit des `Detection` normalisées.

    Une instance = un modèle chargé. L'objet est réutilisé sur toute la durée
    d'une session : on ne recharge jamais les poids à chaque frame.

    Exemple :
        >>> detector = Detector()
        >>> detections = detector.track(frame)   # avec identifiants persistants
        >>> for d in detections:
        ...     print(d.track_id, d.class_name, d.confidence)
    """

    def __init__(
        self,
        weights: str | None = None,
        *,
        device: str | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        image_size: int | None = None,
        max_detections: int | None = None,
        tracked_classes: Sequence[str] | None = None,
        tracker_config: str | None = None,
    ) -> None:
        """Initialise le détecteur.

        Tous les paramètres sont optionnels : la valeur par défaut est celle de
        `config.MODEL`. Les arguments ne servent qu'aux surcharges ponctuelles
        (curseurs de l'interface, tests unitaires) — la configuration reste la
        source de vérité.

        Args:
            weights: Poids YOLOv8 à charger.
            device: Périphérique d'inférence, ou `"auto"`.
            confidence: Seuil de confiance minimal.
            iou: Seuil IoU de la NMS.
            image_size: Taille d'entrée du réseau.
            max_detections: Nombre maximal de détections par frame.
            tracked_classes: Noms des classes à conserver. `None` = celles de la
                configuration ; une séquence vide = **toutes** les classes.
            tracker_config: Fichier YAML du tracker (`"bytetrack.yaml"`).

        Raises:
            ModelLoadError: Si le modèle ne peut pas être chargé.
        """
        defaults = config.MODEL

        self.weights: str = weights or defaults.weights
        self.device: str = _resolve_device(device or defaults.device)
        self.confidence: float = defaults.confidence if confidence is None else confidence
        self.iou: float = defaults.iou if iou is None else iou
        self.image_size: int = defaults.image_size if image_size is None else image_size
        self.max_detections: int = (
            defaults.max_detections if max_detections is None else max_detections
        )
        self.tracker_config: str = tracker_config or defaults.tracker_config

        self._model = load_model(self.weights, self.device)
        self._names: dict[int, str] = dict(self._model.names)

        requested = defaults.tracked_classes if tracked_classes is None else tracked_classes
        self._class_ids: list[int] | None = self._resolve_class_ids(requested)

    # -- Configuration des classes -------------------------------------------

    def _resolve_class_ids(self, class_names: Sequence[str]) -> list[int] | None:
        """Traduit des noms de classes en index compris par YOLO.

        Le filtrage par classe est délégué **au modèle** (paramètre `classes=`
        d'Ultralytics) et non fait a posteriori en Python : les détections des
        classes non désirées sont écartées pendant la NMS, ce qui est plus rapide
        et évite de construire des objets qu'on va jeter.

        Un nom inconnu est journalisé en avertissement et ignoré, jamais levé en
        exception : une faute de frappe dans la configuration ne doit pas empêcher
        l'application de démarrer.

        Args:
            class_names: Noms des classes souhaitées. Vide = toutes.

        Returns:
            La liste des index correspondants, ou `None` pour « toutes les
            classes » (valeur attendue par Ultralytics).
        """
        if not class_names:
            return None

        name_to_id = {name: idx for idx, name in self._names.items()}
        resolved: list[int] = []
        unknown: list[str] = []

        for name in class_names:
            if name in name_to_id:
                resolved.append(name_to_id[name])
            else:
                unknown.append(name)

        if unknown:
            logger.warning(
                "Classes inconnues du modèle '%s', ignorées : %s",
                self.weights,
                ", ".join(unknown),
            )
        if not resolved:
            logger.warning(
                "Aucune classe valide sélectionnée : toutes les classes seront détectées."
            )
            return None

        return resolved

    def set_tracked_classes(self, class_names: Sequence[str]) -> None:
        """Change les classes surveillées à chaud (multiselect de l'interface).

        Args:
            class_names: Nouveaux noms de classes. Séquence vide = toutes.
        """
        self._class_ids = self._resolve_class_ids(class_names)

    @property
    def available_classes(self) -> list[str]:
        """Toutes les classes que le modèle sait détecter, triées par index.

        Alimente le `st.multiselect` de l'interface : la liste des choix vient du
        modèle lui-même, elle n'est pas recopiée à la main quelque part.
        """
        return [self._names[idx] for idx in sorted(self._names)]

    @property
    def tracked_classes(self) -> list[str]:
        """Classes actuellement surveillées."""
        if self._class_ids is None:
            return self.available_classes
        return [self._names[idx] for idx in self._class_ids]

    @property
    def model(self) -> "YOLO":
        """Accès au modèle brut, pour les cas non couverts par cette classe."""
        return self._model

    # -- Inférence ------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Détecte les objets sur une frame, **sans suivi**.

        Chaque appel est indépendant : les détections retournées n'ont pas de
        `track_id`. Utile pour l'analyse d'une image fixe ou pour un aperçu de
        réglage des seuils. C'est l'équivalent direct du mode « image » du
        prototype.

        Args:
            frame: Image BGR (convention OpenCV), tableau `(H, W, 3)`.

        Returns:
            Les détections de la frame, éventuellement vide.

        Raises:
            ValueError: Si la frame est `None` ou vide.
        """
        self._validate_frame(frame)
        results = self._infer(frame, track=False)
        return self._parse_results(results)

    def track(self, frame: np.ndarray, *, persist: bool = True) -> list[Detection]:
        """Détecte et suit les objets sur une frame **d'un flux continu**.

        C'est la méthode à utiliser pour la vidéo. Ultralytics applique ByteTrack
        et attribue un `track_id` stable d'une frame à l'autre. C'est ce qui
        remplace la déduplication par proximité des centres du prototype.

        Args:
            frame: Image BGR successive du même flux.
            persist: Conserver l'état du tracker entre les appels. Doit rester
                `True` tant qu'on traite la même vidéo ; appeler `reset()` avant
                de passer à une autre source.

        Returns:
            Les détections de la frame, avec `track_id` renseigné dès que
            ByteTrack a confirmé la piste (`None` sur les toutes premières frames
            d'un nouvel objet).

        Raises:
            ValueError: Si la frame est `None` ou vide.
        """
        self._validate_frame(frame)
        results = self._infer(frame, track=True, persist=persist)
        return self._parse_results(results)

    def reset(self) -> None:
        """Réinitialise l'état interne du tracker.

        À appeler impérativement entre deux vidéos : sans cela, les identifiants
        de la vidéo précédente continuent d'être incrémentés et, pire, ByteTrack
        peut associer un objet de la nouvelle vidéo à une piste de l'ancienne.
        """
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if not trackers:
            return
        for tracker in trackers:
            try:
                tracker.reset()
            except Exception:  # l'API interne d'Ultralytics peut évoluer
                logger.debug("Réinitialisation du tracker impossible.", exc_info=True)

    # -- Détails d'implémentation --------------------------------------------

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Vérifie qu'une frame est exploitable avant de la passer au modèle.

        Args:
            frame: Frame à valider.

        Raises:
            ValueError: Si la frame est absente, vide ou n'a pas 3 dimensions.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("La frame fournie est absente ou n'est pas un tableau NumPy.")
        if frame.size == 0:
            raise ValueError("La frame fournie est vide.")
        if frame.ndim != 3:
            raise ValueError(f"Frame de forme inattendue : {frame.shape}, attendu (H, W, 3).")

    def _infer(self, frame: np.ndarray, *, track: bool, persist: bool = True) -> Any:
        """Appelle le modèle et absorbe les erreurs d'inférence.

        Une frame corrompue au milieu d'une vidéo de 10 minutes ne doit pas faire
        tomber toute l'analyse : on journalise et on retourne un résultat vide, la
        frame est simplement sans détection. Les erreurs *systémiques* (poids
        introuvables, périphérique invalide) ont déjà été levées au chargement,
        donc ce `except` large n'avale pas d'erreur de configuration.

        Args:
            frame: Image BGR.
            track: True pour `model.track()`, False pour `model.predict()`.
            persist: Transmis à `model.track()`.

        Returns:
            La liste de résultats Ultralytics, ou une liste vide en cas d'échec.
        """
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.image_size,
            "max_det": self.max_detections,
            "device": self.device,
            "verbose": False,
        }
        if self._class_ids is not None:
            kwargs["classes"] = self._class_ids

        try:
            if track:
                return self._model.track(
                    frame, persist=persist, tracker=self.tracker_config, **kwargs
                )
            return self._model.predict(frame, **kwargs)
        except Exception:
            logger.exception("Échec de l'inférence sur une frame ; frame ignorée.")
            return []

    def _parse_results(self, results: Any) -> list[Detection]:
        """Convertit la sortie Ultralytics en `Detection`.

        Frontière d'anti-corruption : c'est le **seul** endroit du projet qui
        manipule des tenseurs Ultralytics. Tout le reste travaille sur des
        `Detection`, ce qui rend le pipeline testable avec des objets fabriqués à
        la main, sans modèle ni GPU.

        Args:
            results: Sortie de `model.predict()` / `model.track()`.

        Returns:
            Les détections converties.
        """
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        # boxes.id vaut None en mode predict() (pas de suivi du tout) et tant que
        # ByteTrack n'a confirmé aucune piste sur les premières frames.
        if getattr(boxes, "id", None) is not None:
            track_ids: list[int | None] = boxes.id.cpu().numpy().astype(int).tolist()
        else:
            track_ids = [None] * len(class_ids)

        names = getattr(result, "names", None) or self._names

        detections: list[Detection] = []
        for box, confidence, class_id, track_id in zip(
            xyxy, confidences, class_ids, track_ids
        ):
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=names.get(int(class_id), str(class_id)),
                    confidence=float(confidence),
                    xyxy=(x1, y1, x2, y2),
                    track_id=None if track_id is None else int(track_id),
                )
            )
        return detections

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        return (
            f"Detector(weights={self.weights!r}, device={self.device!r}, "
            f"conf={self.confidence}, classes={len(self.tracked_classes)})"
        )
