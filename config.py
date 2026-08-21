"""Configuration centralisée de SentinelReport.

Principe directeur : **aucune valeur métier codée en dur ailleurs**. Les autres
modules importent `config` et lisent les dataclasses ci-dessous. Cela permet de
changer un seuil, une zone ou une durée sans toucher à une seule ligne de logique.

Toutes les dataclasses sont `frozen=True` : la configuration est immuable au
runtime, ce qui évite qu'un module la modifie par effet de bord.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import time
from enum import Enum
from pathlib import Path
from typing import Final, Sequence

# ---------------------------------------------------------------------------
# 1. Chemins
# ---------------------------------------------------------------------------

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
MODELS_DIR: Final[Path] = BASE_DIR / "models"
EVIDENCE_DIR: Final[Path] = BASE_DIR / "evidence"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"
DATA_DIR: Final[Path] = BASE_DIR / "data"
VIDEOS_DIR: Final[Path] = DATA_DIR / "videos"

# `assets/` n'est PAS créé automatiquement, contrairement aux dossiers ci-dessous :
# c'est une ressource **fournie** (les fichiers de marque), pas une sortie du
# programme. Le créer vide masquerait un dépôt incomplet derrière un dossier
# existant mais sans contenu.
ASSETS_DIR: Final[Path] = BASE_DIR / "assets"

for _directory in (MODELS_DIR, EVIDENCE_DIR, REPORTS_DIR, VIDEOS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Modèle de détection
# ---------------------------------------------------------------------------

# Modèles proposés dans l'interface : nom lisible -> fichier de poids local.
#
# Les chemins sont **absolus** (dérivés de `MODELS_DIR`) et non relatifs : ils
# restent valides quel que soit le répertoire depuis lequel on lance
# `streamlit run app.py` ou `pytest`.
#
# Ces fichiers ne sont PAS téléchargés par l'application : ils sont déposés une
# fois pour toutes par `python download_models.py`. Ajouter une entrée ici suffit
# à la faire apparaître à la fois dans le menu déroulant et dans le script de
# téléchargement — il n'y a qu'une seule liste à maintenir.
#
# Compromis vitesse / précision (mAP COCO indicative, poids officiels v8) :
#   nano   ~3,2 M paramètres  | mAP50-95 ~37,3 | le plus rapide sur CPU
#   small  ~11,2 M paramètres | mAP50-95 ~44,9 | ~2-3x plus lent que nano
#   medium ~25,9 M paramètres | mAP50-95 ~50,2 | ~5-6x plus lent que nano
AVAILABLE_MODELS: Final[dict[str, Path]] = {
    "Rapide (nano)": MODELS_DIR / "yolov8n.pt",
    "Équilibré (small)": MODELS_DIR / "yolov8s.pt",
    "Précis (medium)": MODELS_DIR / "yolov8m.pt",
}

# Modèle sélectionné par défaut au démarrage : le nano, car c'est le seul qui
# tienne le temps réel sur un CPU ordinaire.
DEFAULT_MODEL_LABEL: Final[str] = "Rapide (nano)"
DEFAULT_MODEL_PATH: Final[Path] = AVAILABLE_MODELS[DEFAULT_MODEL_LABEL]

# Aide affichée sous le sélecteur de l'interface.
MODEL_DESCRIPTIONS: Final[dict[str, str]] = {
    "Rapide (nano)": "Le plus rapide. Suffisant pour des personnes et des véhicules proches.",
    "Équilibré (small)": "Meilleur compromis : rattrape les objets petits ou partiellement masqués.",
    "Précis (medium)": "Le plus précis, mais nettement plus lent sur CPU. À réserver au GPU.",
}


def missing_models() -> dict[str, Path]:
    """Liste les modèles déclarés mais absents du disque.

    Sert à l'interface pour afficher un message d'installation clair plutôt que
    de laisser Ultralytics échouer (ou pire, tenter un téléchargement) en pleine
    exécution.

    Returns:
        Les entrées de `AVAILABLE_MODELS` dont le fichier n'existe pas.
    """
    return {label: path for label, path in AVAILABLE_MODELS.items() if not path.is_file()}


@dataclass(frozen=True)
class ModelConfig:
    """Paramètres du modèle YOLOv8 et de l'inférence.

    Attributes:
        weights: Chemin des poids par défaut. Il pointe vers un fichier **local**
            de `models/`, déposé au préalable par `download_models.py` : léger,
            rapide sur CPU, suffisant pour un prototype. L'interface peut le
            remplacer par une autre entrée de `AVAILABLE_MODELS`.
        device: `"auto"` détecte CUDA/MPS et retombe sur `"cpu"`. Peut être
            forcé à `"cpu"`, `"0"` (premier GPU CUDA) ou `"mps"` (Apple Silicon).
        confidence: Seuil de confiance minimal d'une détection [0-1].
        iou: Seuil IoU de la suppression des non-maxima (NMS).
        image_size: Taille d'entrée du réseau (multiple de 32).
        max_detections: Garde-fou contre les scènes saturées.
        tracked_classes: Classes COCO surveillées, par **nom** (et non par index :
            les index sont un détail d'implémentation du modèle, les noms sont
            lisibles et stables si l'on change de modèle).
        tracker_config: Fichier de configuration du tracker fourni par
            Ultralytics. `bytetrack.yaml` ou `botsort.yaml`.
    """

    weights: str = str(DEFAULT_MODEL_PATH)
    device: str = "auto"
    confidence: float = 0.35
    iou: float = 0.50
    image_size: int = 640
    max_detections: int = 100
    tracked_classes: tuple[str, ...] = (
        # Personnes et véhicules (héritage du compteur d'objets existant)
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        # Objets susceptibles d'être abandonnés
        "backpack",
        "handbag",
        "suitcase",
    )
    tracker_config: str = "bytetrack.yaml"


MODEL: Final[ModelConfig] = ModelConfig()


# ---------------------------------------------------------------------------
# 2 bis. Catalogue de classes surveillables
# ---------------------------------------------------------------------------

# Les 80 classes du jeu de données COCO, dans l'ordre des index du modèle.
# Elles sont figées ici plutôt que lues dans les poids : l'interface doit
# pouvoir proposer la liste **avant** qu'un modèle soit chargé, et cette liste
# est identique pour yolov8n, s et m.
COCO_CLASSES: Final[tuple[str, ...]] = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
)


@dataclass(frozen=True)
class ClassGroup:
    """Un regroupement de classes COCO sous un intitulé métier.

    L'opérateur d'un système de sécurité raisonne en « véhicules » ou en
    « bagages », pas en index COCO. Ce niveau de traduction vit dans la
    configuration et nulle part ailleurs : l'interface affiche `label`, le
    détecteur reçoit `classes`.

    Attributes:
        label: Intitulé affiché dans l'interface.
        classes: Classes COCO correspondantes.
        icon: Émoji d'accompagnement, purement visuel.
        note: Avertissement éventuel sur les limites du modèle.
    """

    label: str
    classes: tuple[str, ...]
    icon: str = ""
    note: str = ""


CLASS_GROUPS: Final[tuple[ClassGroup, ...]] = (
    ClassGroup(
        label="Personnes",
        classes=("person",),
        icon="\N{BUST IN SILHOUETTE}",
    ),
    ClassGroup(
        label="Véhicules",
        classes=("car", "truck", "bus", "motorcycle", "bicycle"),
        icon="\N{AUTOMOBILE}",
    ),
    ClassGroup(
        label="Objets dangereux",
        classes=("knife", "scissors", "baseball bat"),
        icon="\N{WARNING SIGN}",
        # Limite structurante, affichée dans l'interface : il n'existe AUCUNE
        # classe d'arme à feu dans COCO. Un pistolet sera au mieux ignoré, au
        # pire classé « cell phone ». Les détecter exigerait un modèle
        # réentraîné sur un jeu de données dédié (voir la feuille de route).
        note=(
            "COCO ne contient ni pistolet ni fusil : seuls couteau, ciseaux et "
            "batte sont détectables. Les armes à feu exigent un modèle réentraîné."
        ),
    ),
    ClassGroup(
        label="Sacs et bagages",
        classes=("backpack", "handbag", "suitcase"),
        icon="\N{SCHOOL SATCHEL}",
    ),
)

DEFAULT_CLASS_GROUPS: Final[tuple[str, ...]] = (
    "Personnes",
    "Véhicules",
    "Sacs et bagages",
)


def classes_for_groups(labels: Sequence[str]) -> list[str]:
    """Traduit des intitulés de groupes en classes COCO, sans doublon.

    Args:
        labels: Intitulés retenus dans l'interface.

    Returns:
        Les classes COCO correspondantes, dans l'ordre de déclaration.
    """
    selected = set(labels)
    resolved: list[str] = []
    for group in CLASS_GROUPS:
        if group.label in selected:
            resolved.extend(name for name in group.classes if name not in resolved)
    return resolved


# ---------------------------------------------------------------------------
# 3. Suivi temporel (tracker.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackingConfig:
    """Paramètres de la mémoire temporelle maintenue par `Tracker`.

    Attributes:
        max_age_s: Durée (en secondes de temps vidéo) pendant laquelle un objet
            non revu reste en mémoire avant d'être purgé. Absorbe les occlusions
            courtes (quelqu'un passe derrière un poteau).
        history_seconds: Fenêtre glissante de positions conservées par objet.
            Sert au calcul d'immobilité (objet abandonné).
        min_hits: Nombre d'apparitions consécutives avant qu'un objet soit
            considéré comme confirmé. Filtre les faux positifs éphémères.
    """

    max_age_s: float = 3.0
    history_seconds: float = 30.0
    min_hits: int = 3


TRACKING: Final[TrackingConfig] = TrackingConfig()


# ---------------------------------------------------------------------------
# 4. Vocabulaire des incidents (events.py)
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Types d'incidents détectés. Hérite de `str` pour une sérialisation JSON directe."""

    INTRUSION = "intrusion_zone_restreinte"
    LOITERING = "presence_prolongee"
    ABANDONED_OBJECT = "objet_abandonne"
    AFTER_HOURS = "presence_hors_horaires"


class Severity(str, Enum):
    """Niveau de gravité, repris tel quel dans le rapport."""

    LOW = "faible"
    MEDIUM = "moyenne"
    HIGH = "elevee"

    @property
    def rank(self) -> int:
        """Rang de gravité croissant, de 0 (faible) à 2 (élevée).

        L'ordre des membres d'une énumération n'est pas un ordre métier : rien
        n'empêche d'insérer un niveau au milieu. Le rendre explicite évite qu'un
        comparateur ne s'appuie sur l'ordre de déclaration, et surtout que
        l'interface ne le réimplémente de son côté par une liste ordonnée en dur.

        Returns:
            Le rang, comparable entre niveaux.
        """
        return _SEVERITY_RANKS[self]


_SEVERITY_RANKS: Final[dict[Severity, int]] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
}


def worst_severity(severities: Sequence[Severity]) -> Severity | None:
    """Gravité la plus élevée d'un ensemble.

    Args:
        severities: Gravités à comparer, éventuellement vide.

    Returns:
        La plus élevée, ou `None` si l'ensemble est vide — un ensemble vide n'a
        pas de pire élément, et rendre `LOW` par défaut ferait croire à un
        incident bénin là où il n'y en a aucun.
    """
    return max(severities, key=lambda niveau: niveau.rank, default=None)


# ---------------------------------------------------------------------------
# 5. Zones surveillées (zones.py)
# ---------------------------------------------------------------------------


class ZoneType(str, Enum):
    """Ce qu'une zone **attend** de ce qui s'y passe.

    Un polygone plus un booléen `restricted` ne suffisait pas à décrire un
    périmètre réel. Un quai de chargement, un hall d'accueil et une réserve
    n'appellent pas les mêmes règles : rôder dans un hall est banal, rôder dans
    une réserve ne l'est pas, et compter les passages à une porte ne doit lever
    aucun incident du tout.

    Le type conditionne donc **quelles règles s'appliquent** dans la zone. Les
    règles cessent d'être globales : c'est la zone qui déclare ce qu'elle
    surveille. Un seul mécanisme de ciblage, donc aucun conflit à arbitrer entre
    « la zone dit non » et « la règle dit oui ».
    """

    FORBIDDEN = "interdite"
    SCHEDULED = "horaires"
    TRANSIT = "transit"
    SENSITIVE = "sensible"
    COUNTING = "comptage"

    @property
    def event_types(self) -> tuple["EventType", ...]:
        """Types d'incidents que ce genre de zone peut lever.

        Returns:
            Les types applicables, éventuellement aucun.
        """
        return _ZONE_EVENT_TYPES[self]

    @property
    def is_restricted(self) -> bool:
        """True si la seule présence y constitue une infraction."""
        return self is ZoneType.FORBIDDEN

    @property
    def description(self) -> str:
        """Phrase explicative, reprise dans l'interface et la documentation."""
        return _ZONE_DESCRIPTIONS[self]


# Ce que chaque type de zone accepte de signaler. Écrit comme une table plutôt
# que dispersé dans le moteur : lire ces cinq entrées doit suffire à savoir ce
# qu'une zone déclenche.
_ZONE_EVENT_TYPES: Final[dict[ZoneType, tuple["EventType", ...]]] = {
    # Toute présence est anormale : rien n'est écarté.
    ZoneType.FORBIDDEN: (
        EventType.INTRUSION,
        EventType.LOITERING,
        EventType.ABANDONED_OBJECT,
        EventType.AFTER_HOURS,
    ),
    # Présence normale aux heures d'ouverture. Pas d'intrusion, donc, mais tout
    # le reste : la nuit, un stationnement prolongé ou un objet déposé comptent.
    ZoneType.SCHEDULED: (
        EventType.LOITERING,
        EventType.ABANDONED_OBJECT,
        EventType.AFTER_HOURS,
    ),
    # On y passe, on n'y reste pas. Traverser ne déclenche rien — c'est la
    # définition même d'un lieu de transit — mais s'y arrêter, si.
    ZoneType.TRANSIT: (EventType.LOITERING, EventType.ABANDONED_OBJECT),
    # Lieu où un objet déposé est le vrai risque : quai, salle d'attente, hall
    # de gare. La présence de personnes y est normale et prolongée.
    ZoneType.SENSITIVE: (EventType.ABANDONED_OBJECT, EventType.AFTER_HOURS),
    # Mesure de flux uniquement. Ne lever aucun incident est une décision, pas
    # un oubli : compter les entrées d'un magasin ne doit pas produire un
    # rapport par client.
    ZoneType.COUNTING: (),
}

_ZONE_DESCRIPTIONS: Final[dict[ZoneType, str]] = {
    ZoneType.FORBIDDEN: "Toute présence y constitue une infraction.",
    ZoneType.SCHEDULED: "Présence normale aux heures d'ouverture, anormale en dehors.",
    ZoneType.TRANSIT: "On y passe : seule la présence prolongée déclenche.",
    ZoneType.SENSITIVE: "Lieu où un objet abandonné est le risque principal.",
    ZoneType.COUNTING: "Mesure de flux : aucun incident n'y est levé.",
}


@dataclass(frozen=True)
class SurveillanceZone:
    """Une zone surveillée : géométrie, nature, et seuils qui lui sont propres.

    Remplace l'ancien `ZoneConfig` (polygone + booléen `restricted`), qui ne
    permettait pas d'exprimer qu'un hall et une réserve n'appellent pas les
    mêmes règles.

    Les sommets restent exprimés en **coordonnées normalisées** (0.0-1.0) : la
    même définition fonctionne quelle que soit la résolution de la vidéo.
    `ZoneManager` les convertit en pixels une fois la première frame connue.

    Surcharges de seuils
    --------------------
    Les trois champs `min_duration_s`, `cooldown_s` et `max_movement_ratio`
    valent `None` par défaut, ce qui signifie « prendre la valeur globale de
    `EVENT_RULES` ». Les renseigner permet d'être plus strict dans une réserve
    que dans un hall sans dupliquer le jeu de règles. `None` et une valeur
    identique à la globale ne sont pas la même chose : la première suit les
    ajustements futurs de la configuration, la seconde les ignore.

    Attributes:
        name: Identifiant lisible, réutilisé dans les rapports. Sert de clé.
        polygon: Sommets ((x, y), ...) normalisés, dans l'ordre du tracé.
        zone_type: Nature de la zone — conditionne les règles applicables.
        color: Couleur d'affichage BGR (convention OpenCV). Sert aussi à colorer
            les objets en infraction dans cette zone.
        draw: Tracer la zone sur la vidéo annotée. À laisser à False pour une
            zone couvrant tout le champ : la remplir teinterait l'image entière
            sans rien apprendre à l'opérateur.
        min_duration_s: Durée de déclenchement propre à la zone. `None` = valeur
            de la règle globale.
        cooldown_s: Délai de garde propre à la zone. `None` = valeur globale.
        max_movement_ratio: Seuil d'immobilité propre à la zone, en fraction de
            la hauteur apparente de l'objet. `None` = valeur globale.
        crossing_line: Nom de la ligne de comptage associée, pour une zone de
            type `COUNTING`. `None` = pas de ligne.
    """

    name: str
    polygon: tuple[tuple[float, float], ...]
    zone_type: ZoneType = ZoneType.FORBIDDEN
    color: tuple[int, int, int] = (0, 0, 255)
    draw: bool = True
    min_duration_s: float | None = None
    cooldown_s: float | None = None
    max_movement_ratio: float | None = None
    crossing_line: str | None = None

    @property
    def restricted(self) -> bool:
        """True si la seule présence y constitue une intrusion.

        Conservé comme propriété **dérivée** du type : le drapeau reste lisible
        là où il suffit (couleur d'alerte, étiquette dessinée) sans redevenir une
        seconde source de vérité qu'on pourrait mettre en contradiction avec le
        type.
        """
        return self.zone_type.is_restricted

    def handles(self, event_type: EventType) -> bool:
        """Indique si cette zone accepte de lever ce type d'incident.

        Args:
            event_type: Type d'incident envisagé.

        Returns:
            True si le type de la zone le déclare.
        """
        return event_type in self.zone_type.event_types

    def rule_for(self, rule: EventRule) -> EventRule:
        """Règle globale ajustée aux seuils propres à cette zone.

        Args:
            rule: Règle issue de `EVENT_RULES`.

        Returns:
            La règle telle qu'elle s'applique **ici**. La règle d'origine est
            rendue telle quelle si la zone ne surcharge rien : pas de copie
            inutile, et l'identité rend visible l'absence de surcharge.

        Raises:
            ValueError: Si les surcharges cassent l'invariant
                `cooldown_s > min_duration_s`. Une zone qui exige 120 s de
                présence mais autorise un redéclenchement toutes les 60 s
                produirait un rapport par frame.
        """
        surcharges = {
            nom: valeur
            for nom, valeur in (
                ("min_duration_s", self.min_duration_s),
                ("cooldown_s", self.cooldown_s),
                ("max_movement_ratio", self.max_movement_ratio),
            )
            if valeur is not None
        }
        if not surcharges:
            return rule

        ajustee = replace(rule, **surcharges)
        if ajustee.cooldown_s <= ajustee.min_duration_s:
            raise ValueError(
                f"Zone « {self.name} » : cooldown_s ({ajustee.cooldown_s:g} s) doit "
                f"dépasser min_duration_s ({ajustee.min_duration_s:g} s) pour la règle "
                f"{rule.event_type.value}, sinon l'anti-rebond est sans effet."
            )
        return ajustee


# Surveillance globale : une seule zone, qui couvre l'intégralité de l'image.
#
# C'est le mode par défaut — tout ce qui est visible est surveillé. Le découpage
# en sous-zones reste possible et pleinement fonctionnel : il suffit de remplacer
# l'entrée ci-dessous par plusieurs `SurveillanceZone` aux polygones normalisés
# (voir la section « Adapter les zones » du README). La machinerie de `zones.py`
# ne change pas ; seuls le périmètre et les règles applicables changent.
#
# Pourquoi conserver une zone plutôt que de supprimer le mécanisme : toutes les
# règles temporelles reposent sur un chronomètre « depuis quand cet objet est-il
# ICI ». Sans périmètre nommé, il n'y a plus de « ici », donc plus de durée, donc
# plus d'incident. La zone plein cadre est le périmètre le plus simple possible.
#
# Son type est `FORBIDDEN` : c'est le seul qui déclare les quatre règles, donc
# celui qui correspond à « surveiller tout ce qui s'affiche ».
ZONES: Final[tuple[SurveillanceZone, ...]] = (
    SurveillanceZone(
        name="Champ de la caméra",
        polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        zone_type=ZoneType.FORBIDDEN,
        color=(0, 0, 255),
        draw=False,
    ),
)


@dataclass(frozen=True)
class GeometryConfig:
    """Comment décider qu'un objet « est dans » une zone.

    Les deux compteurs de frames forment une **hystérésis** : il faut plusieurs
    frames pour entrer dans une zone, et plusieurs frames pour en sortir. Sans
    cette asymétrie, le bruit de détection — une boîte qui tremble de quelques
    pixels sur la frontière — ferait osciller l'appartenance d'une frame à
    l'autre, et **remettrait à zéro le chronomètre d'intrusion** à chaque
    oscillation. Une intrusion de 50 secondes ne serait alors jamais signalée.

    Attributes:
        anchor: Point de la boîte testé contre le polygone.
            `"bottom_center"` = point d'appui au sol : c'est le bon choix pour
            une caméra en plongée, car le centre de la boîte d'une personne
            debout peut être hors zone alors que ses pieds y sont.
        min_overlap_frames: Frames consécutives dans la zone avant de valider
            l'entrée. Filtre les incursions d'une frame dues au bruit.
        exit_tolerance_frames: Frames consécutives hors de la zone avant de
            valider la sortie. Plus élevé que l'entrée : perdre un objet est
            plus fréquent que le détecter à tort, et une sortie prématurée coûte
            un incident manqué.
    """

    anchor: str = "bottom_center"
    min_overlap_frames: int = 2
    exit_tolerance_frames: int = 4


GEOMETRY: Final[GeometryConfig] = GeometryConfig()


# ---------------------------------------------------------------------------
# 6. Règles d'événements (events.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRule:
    """Règle déclenchant un événement.

    Une règle = un type d'événement + les conditions à réunir. `EventEngine` les
    évalue à chaque frame sur chaque objet suivi.

    Attributes:
        event_type: Type d'incident produit.
        classes: Classes concernées. Tuple vide = toutes les classes surveillées.
        min_duration_s: Durée minimale de la condition avant déclenchement.
            C'est ce qui distingue une intrusion d'un simple passage.
        cooldown_s: Délai avant qu'un même objet puisse redéclencher le même
            type d'événement. Évite de générer 200 rapports pour un incident.
        severity: Gravité par défaut associée.
        max_movement_px: Déplacement maximal toléré sur la fenêtre d'observation
            pour considérer l'objet comme immobile. `None` = critère non appliqué.
            **Ignoré si `max_movement_ratio` est renseigné.**
        max_movement_ratio: Même critère, exprimé en **fraction de la hauteur de
            l'objet**. C'est la version qui résiste à la perspective : 25 px de
            déplacement, c'est un frémissement pour un sac au premier plan et une
            traversée complète pour un sac au fond du champ. Rapporter la mesure
            à la taille apparente de l'objet supprime cette dépendance à la
            profondeur, sans calibration ni homographie.
        requires_no_owner: Si True, l'événement n'est levé que si aucune personne
            n'est à proximité de l'objet (règle « objet abandonné »).
        owner_radius_px: Rayon de recherche du propriétaire présumé.
            **Ignoré si `owner_radius_ratio` est renseigné.**
        owner_radius_ratio: Même rayon, en multiple de la hauteur de l'objet
            surveillé. Même argument de perspective : un rayon fixe de 150 px
            couvre un mètre au premier plan et huit mètres au fond.
    """

    event_type: EventType
    classes: tuple[str, ...] = ()
    min_duration_s: float = 3.0
    cooldown_s: float = 60.0
    severity: Severity = Severity.MEDIUM
    max_movement_px: float | None = None
    max_movement_ratio: float | None = None
    requires_no_owner: bool = False
    owner_radius_px: float = 150.0
    owner_radius_ratio: float | None = None


def scaled_rules(
    factor: float, rules: Sequence[EventRule] | None = None
) -> tuple[EventRule, ...]:
    """Applique un facteur aux durées de déclenchement **et** aux délais de garde.

    Multiplier les deux ensemble n'est pas un détail d'implémentation, c'est la
    règle elle-même : ne toucher qu'au seuil casserait l'invariant
    `cooldown_s > min_duration_s`. À facteur 3, le rôdage passerait à 180 s de
    seuil pour 180 s de garde, et un même objet redéclencherait en boucle — le
    système produirait un rapport par frame, ce que les trois filtres
    anti-fausses-alertes existent précisément pour empêcher.

    Cette fonction vit dans `config.py` et non dans l'interface : c'est une
    transformation de règle métier, pas un affichage. Un futur script batch doit
    pouvoir l'appliquer sans importer Streamlit.

    Args:
        factor: Multiplicateur, strictement positif. 1.0 = configuration
            d'origine, rendue telle quelle.
        rules: Règles à ajuster. `None` = `EVENT_RULES`.

    Returns:
        Les règles ajustées. `EVENT_RULES` reste intact.

    Raises:
        ValueError: Si le facteur n'est pas strictement positif — un facteur nul
            rendrait tout déclenchable en permanence.
    """
    if factor <= 0.0:
        raise ValueError(f"Facteur de durées invalide : {factor}. Il doit être > 0.")

    base = tuple(rules) if rules is not None else EVENT_RULES
    if factor == 1.0:
        return base

    return tuple(
        replace(
            rule,
            min_duration_s=rule.min_duration_s * factor,
            cooldown_s=rule.cooldown_s * factor,
        )
        for rule in base
    )


EVENT_RULES: Final[tuple[EventRule, ...]] = (
    EventRule(
        event_type=EventType.INTRUSION,
        classes=("person",),
        min_duration_s=3.0,
        cooldown_s=60.0,
        severity=Severity.HIGH,
    ),
    EventRule(
        # Présence anormalement longue, y compris dans une zone NON restreinte
        # (rôder devant une entrée). C'est le même mécanisme que l'intrusion,
        # avec un seuil de durée bien plus élevé et une gravité moindre.
        event_type=EventType.LOITERING,
        classes=("person",),
        min_duration_s=60.0,
        cooldown_s=180.0,
        severity=Severity.MEDIUM,
    ),
    EventRule(
        event_type=EventType.ABANDONED_OBJECT,
        classes=("backpack", "handbag", "suitcase"),
        min_duration_s=30.0,
        cooldown_s=300.0,
        severity=Severity.HIGH,
        # Seuils relatifs à la taille apparente de l'objet : un sac est immobile
        # s'il n'a pas bougé de plus d'un tiers de sa propre hauteur, et son
        # propriétaire est « à proximité » s'il se tient à moins de trois
        # hauteurs de sac. Ces deux valeurs gardent le même sens au premier plan
        # comme au fond du champ.
        max_movement_ratio=0.35,
        max_movement_px=25.0,  # repli si la boîte est dégénérée
        requires_no_owner=True,
        owner_radius_ratio=3.0,
        owner_radius_px=150.0,
    ),
    EventRule(
        event_type=EventType.AFTER_HOURS,
        classes=("person",),
        min_duration_s=5.0,
        cooldown_s=120.0,
        severity=Severity.MEDIUM,
    ),
)


@dataclass(frozen=True)
class ScheduleConfig:
    """Plage horaire d'ouverture du site (pour la règle « hors horaires »).

    Attributes:
        opening: Heure d'ouverture.
        closing: Heure de fermeture.
        weekend_closed: Si True, samedi et dimanche sont entièrement hors horaires.
    """

    opening: time = time(7, 30)
    closing: time = time(19, 0)
    weekend_closed: bool = True


SCHEDULE: Final[ScheduleConfig] = ScheduleConfig()

# ---------------------------------------------------------------------------
# 6 bis. Score de priorité (events.py)
# ---------------------------------------------------------------------------


class PriorityLevel(str, Enum):
    """Niveau de priorité affiché à l'opérateur.

    Un nombre brut ne se lit pas : « 72 points » n'aide personne à décider quoi
    regarder en premier. Le score sert au **tri**, le niveau à la **lecture**.
    """

    LOW = "faible"
    MEDIUM = "moyen"
    HIGH = "eleve"
    CRITICAL = "critique"


@dataclass(frozen=True)
class ScoringConfig:
    """Barème du score de priorité — entièrement paramétrable ici.

    Principe directeur : **aucun jugement magique**. Chaque point provient d'un
    signal déjà mesuré ailleurs dans le pipeline — une règle qui s'est
    déclenchée, un nombre de secondes, un déplacement en pixels, une heure au
    calendrier. Il n'y a ni modèle appris, ni pondération opaque, ni seuil
    découvert par les données. Le score n'ajoute donc aucune information : il
    **ordonne** celle qui existe déjà, selon un barème publié et discutable.

    Conséquence pratique : deux analyses de la même vidéo produisent exactement
    les mêmes scores, et tout écart s'explique par une ligne de ce fichier.

    Attributes:
        base_points: Points attribués par type de règle déclenchée. Un objet
            abandonné pèse davantage qu'un simple rôdage, parce qu'il demande
            une intervention plus urgente.
        points_per_minute_present: Points par minute de présence dans la zone
            concernée. Une intrusion de dix minutes est plus préoccupante qu'une
            intrusion de cinq secondes.
        max_duration_points: Plafond des points de durée. Sans plafond, une
            présence très longue écraserait tous les autres signaux, et un
            objet oublié pendant une heure paraîtrait plus urgent qu'une
            intrusion en cours.
        points_per_minute_stationary: Points par minute d'immobilité, pour les
            règles qui la mesurent (objet abandonné).
        max_stationary_points: Plafond des points d'immobilité.
        after_hours_multiplier: Multiplicateur appliqué au total quand
            l'incident survient hors des horaires de `SCHEDULE`. Le même
            comportement est plus grave à 3 h du matin qu'à midi — c'est un
            multiplicateur et non un bonus fixe, pour que la gravité de nuit
            reste proportionnelle à celle du jour.
        combination_points: Points par règle **supplémentaire** déjà déclenchée
            sur le même objet. C'est le signal le plus fort du barème : une
            personne qui a rôdé, puis pénétré en zone restreinte, puis abandonné
            un sac raconte une histoire qu'aucune des trois règles ne dit seule.
        max_combination_points: Plafond du cumul.
        thresholds: Seuils de traduction en niveaux, du plus élevé au plus bas.
            Le premier seuil atteint l'emporte.
    """

    base_points: tuple[tuple[EventType, float], ...] = (
        (EventType.ABANDONED_OBJECT, 35.0),
        (EventType.INTRUSION, 30.0),
        (EventType.LOITERING, 20.0),
        (EventType.AFTER_HOURS, 15.0),
    )
    points_per_minute_present: float = 10.0
    max_duration_points: float = 30.0
    points_per_minute_stationary: float = 8.0
    max_stationary_points: float = 20.0
    after_hours_multiplier: float = 1.5
    combination_points: float = 15.0
    max_combination_points: float = 30.0
    thresholds: tuple[tuple[float, PriorityLevel], ...] = (
        (90.0, PriorityLevel.CRITICAL),
        (60.0, PriorityLevel.HIGH),
        (30.0, PriorityLevel.MEDIUM),
        (0.0, PriorityLevel.LOW),
    )

    def points_for(self, event_type: EventType) -> float:
        """Points de base d'un type de règle.

        Args:
            event_type: Type d'incident déclenché.

        Returns:
            Les points au barème, ou 0 si le type n'y figure pas — un type
            inconnu ne doit pas empêcher le calcul du reste.
        """
        for declared, points in self.base_points:
            if declared is event_type:
                return points
        return 0.0

    def level_for(self, score: float) -> PriorityLevel:
        """Traduit un score en niveau lisible.

        Args:
            score: Score total de l'incident.

        Returns:
            Le niveau correspondant au premier seuil atteint.
        """
        for threshold, level in self.thresholds:
            if score >= threshold:
                return level
        return PriorityLevel.LOW


SCORING: Final[ScoringConfig] = ScoringConfig()

@dataclass(frozen=True)
class TimelineConfig:
    """Échantillonnage de la chronologie (partie 2 du rapport).

    Le découpage se fait en **temps vidéo** et non en temps de traitement : une
    même vidéo analysée sur un portable lent ou sur un GPU doit produire la même
    chronologie. C'est le principe des deux horloges appliqué au rapport.

    Attributes:
        slice_seconds: Durée d'une tranche. 30 s est un compromis : assez court
            pour situer un événement, assez long pour que le rapport d'une heure
            de vidéo tienne en deux pages.
        max_facts_per_slice: Nombre maximal de faits retenus par tranche. Une
            tranche saturée n'informe plus ; au-delà, on résume le surplus.
        notable_from: Niveau de priorité à partir duquel un objet est signalé
            dans la chronologie même sans incident nouveau.
    """

    slice_seconds: float = 30.0
    max_facts_per_slice: int = 8
    notable_from: PriorityLevel = PriorityLevel.HIGH


TIMELINE: Final[TimelineConfig] = TimelineConfig()




# ---------------------------------------------------------------------------
# 7. Preuves visuelles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceConfig:
    """Capture des images justificatives.

    Attributes:
        directory: Dossier de destination.
        filename_template: Gabarit du nom de fichier. Champs disponibles :
            `timestamp`, `event_type`, `track_id`, `zone`.
        jpeg_quality: Qualité JPEG [1-100].
        annotate: Si True, la capture contient les boîtes et les zones dessinées.
        margin_px: Marge ajoutée autour de l'objet si `crop_to_object` est actif.
        crop_to_object: Sauvegarder un recadrage centré sur l'objet en plus de
            la frame complète.
    """

    directory: Path = EVIDENCE_DIR
    filename_template: str = "{timestamp}_{event_type}_id{track_id}.jpg"
    timestamp_format: str = "%Y%m%d-%H%M%S"
    jpeg_quality: int = 90
    annotate: bool = True
    margin_px: int = 40
    crop_to_object: bool = False


EVIDENCE: Final[EvidenceConfig] = EvidenceConfig()


# ---------------------------------------------------------------------------
# 8. Génération de rapports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportConfig:
    """Paramètres du générateur de rapports.

    Attributes:
        organization: Nom affiché en en-tête.
        site_name: Site surveillé.
        operator: Nom de l'opérateur par défaut (modifiable dans l'interface).
        directory: Dossier de sortie des PDF/textes.
        language: Langue des gabarits.
        include_evidence_image: Intégrer la capture au PDF.
        reference_prefix: Préfixe du numéro d'incident (ex. SR-20260813-0001).
    """

    organization: str = "SentinelReport"
    site_name: str = "Site non specifie"
    operator: str = "Operateur non specifie"
    directory: Path = REPORTS_DIR
    language: str = "fr"
    include_evidence_image: bool = True
    reference_prefix: str = "SR"


REPORT: Final[ReportConfig] = ReportConfig()


@dataclass(frozen=True)
class LLMConfig:
    """Point d'extension optionnel : reformulation du rapport par un LLM.

    Désactivé par défaut. Le pipeline nominal fonctionne **entièrement sans
    LLM** (gabarits de texte déterministes) ; le LLM n'est qu'une couche de
    rédaction par-dessus des faits déjà établis par la vision, jamais une source
    de faits. C'est un choix défendable en entrevue : un rapport d'incident doit
    être reproductible et vérifiable.

    Attributes:
        enabled: Active la reformulation.
        model: Identifiant du modèle Claude.
        api_key_env: Nom de la variable d'environnement contenant la clé.
        max_tokens: Plafond de la réponse.
        timeout_s: Délai au-delà duquel on retombe sur le gabarit.
        system_prompt: Cadrage du rôle du modèle.
    """

    enabled: bool = False
    model: str = "claude-opus-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 1024
    timeout_s: float = 20.0
    system_prompt: str = (
        "Tu rédiges des rapports d'incident de sécurité en français, dans un "
        "style factuel et neutre. Tu n'utilises QUE les faits fournis : "
        "n'invente aucun détail, aucune identité, aucune cause."
    )

    @property
    def api_key(self) -> str | None:
        """Clé API lue depuis l'environnement, ou None si absente."""
        return os.environ.get(self.api_key_env)


LLM: Final[LLMConfig] = LLMConfig()


# ---------------------------------------------------------------------------
# 9. Traitement vidéo et interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoConfig:
    """Lecture de la source vidéo.

    Attributes:
        frame_stride: Traiter une frame sur N. 1 = toutes. Augmenter allège le
            CPU au prix de la précision temporelle.
        default_fps: FPS supposé si la source n'expose pas l'information
            (fréquent avec les webcams).
        max_width: Redimensionnement de la frame avant inférence.
        webcam_index: Index de la webcam pour `cv2.VideoCapture`.
        allowed_extensions: Extensions acceptées à l'upload.
        min_fps: Cadence minimale crédible annoncée par une source. En dessous,
            la valeur est jugée fantaisiste et `default_fps` prend le relais.
            Une webcam qui annonce 0 fps donnerait une division par zéro ; un
            conteneur mal muxé qui annonce 0.04 fps rendrait toutes les durées
            absurdes.
        max_fps: Cadence maximale crédible. Certains flux annoncent 1000 fps.
    """

    frame_stride: int = 1
    default_fps: float = 25.0
    max_width: int = 1280
    webcam_index: int = 0
    allowed_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    min_fps: float = 1.0
    max_fps: float = 240.0

    def credible_fps(self, announced: float | None) -> float:
        """Cadence retenue pour une source, après contrôle de vraisemblance.

        Args:
            announced: Cadence annoncée par la source, éventuellement absente.

        Returns:
            La cadence annoncée si elle est crédible, sinon `default_fps`.
        """
        if announced is None:
            return self.default_fps
        return announced if self.min_fps <= announced <= self.max_fps else self.default_fps


VIDEO: Final[VideoConfig] = VideoConfig()


class SourceKind(str, Enum):
    """Nature de la source d'images.

    La distinction structurante n'est pas « d'où vient l'image » mais **la
    source a-t-elle une fin**. Un fichier se termine, une caméra jamais : cela
    change l'arrêt, l'horloge de référence et la stratégie de cadence.
    """

    FILE = "fichier"
    WEBCAM = "webcam"
    STREAM = "flux"

    @property
    def is_live(self) -> bool:
        """True si la source est un direct, donc infinie."""
        return self is not SourceKind.FILE


@dataclass(frozen=True)
class SourceConfig:
    """Ouverture et pilotage de la source d'images.

    Attributes:
        kind: Nature par défaut de la source.
        file_path: Fichier analysé quand `kind` vaut `FILE`.
        webcam_index: Index passé à OpenCV quand `kind` vaut `WEBCAM`.
        stream_url: URL RTSP/HTTP quand `kind` vaut `STREAM`. Forme habituelle :
            `rtsp://utilisateur:motdepasse@192.168.1.10:554/stream1`.
        open_timeout_ms: Délai maximal d'établissement de la connexion. Sans lui,
            `cv2.VideoCapture` sur une IP injoignable bloque parfois plusieurs
            dizaines de secondes — l'interface paraîtrait figée.
        read_timeout_ms: Délai maximal d'attente d'une image du flux.
        reconnect_attempts: Nombre de reconnexions tentées avant d'abandonner.
            Une coupure réseau brève ne doit pas arrêter une surveillance.
        reconnect_delay_s: Attente entre deux tentatives.
        drop_late_frames: En direct, ignorer les images accumulées pendant le
            traitement pour ne garder que la plus récente. Voir `sentinel.source`.
        max_dropped_frames: Plafond de sécurité sur le nombre d'images sautées
            d'un coup, pour qu'un long décrochage ne bloque pas la boucle.
        buffer_size: Taille de tampon demandée au pilote (`CAP_PROP_BUFFERSIZE`).
            1 = « donne-moi toujours la dernière image ». Tous les pilotes ne
            l'honorent pas, d'où le filet de sécurité `drop_late_frames`.
    """

    kind: SourceKind = SourceKind.FILE
    file_path: Path | None = None
    webcam_index: int = 0
    stream_url: str = ""
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    reconnect_attempts: int = 3
    reconnect_delay_s: float = 2.0
    drop_late_frames: bool = True
    max_dropped_frames: int = 60
    buffer_size: int = 1


SOURCE: Final[SourceConfig] = SourceConfig()


@dataclass(frozen=True)
class BrandConfig:
    """Fichiers d'identité visuelle.

    **Les champs sont nommés d'après le fond sur lequel le logo se pose**, et non
    d'après la couleur du logo lui-même. C'est volontaire : « logo clair » est
    ambigu (encre claire ? destiné à un fond clair ?) et l'erreur d'orientation
    est invisible en relecture de code — elle ne se voit qu'à l'affichage, sous
    la forme d'un logo illisible sur son propre fond.

    Correspondance avec les fichiers fournis :

    * `logo_on_dark` → `logo-sombre.svg` : encre claire (`#F8FAFC`) sur pavé
      anthracite `#0F172A` intégré au fichier. C'est la version de l'en-tête.
    * `logo_on_light` → `logo-claire.svg` : encre anthracite `#0F172A` sur fond
      **transparent**. C'est la version des supports blancs — rapport PDF,
      impression, aperçu sur fond clair.

    Attributes:
        directory: Dossier des fichiers de marque.
        logo_on_dark: Logo à poser sur un fond sombre.
        logo_on_light: Logo à poser sur un fond clair.
        favicon: Icône d'onglet du navigateur.
        header_width_px: Largeur d'affichage du logo dans l'en-tête. Les SVG
            fournis déclarent `width="100%"` sans dimension intrinsèque : une
            largeur explicite est **obligatoire** pour un rendu déterministe.
        anthracite: Couleur de fond de la marque, reprise des fichiers SVG.
    """

    directory: Path = ASSETS_DIR
    logo_on_dark: Path = ASSETS_DIR / "logo-sombre.svg"
    logo_on_light: Path = ASSETS_DIR / "logo-claire.svg"
    favicon: Path = ASSETS_DIR / "favicon.svg"
    header_width_px: int = 380
    anthracite: str = "#0F172A"

    def logo_for(self, *, dark_background: bool) -> Path:
        """Retourne le logo adapté au fond sur lequel il sera posé.

        Args:
            dark_background: True si le support est sombre (en-tête de l'appli),
                False s'il est clair (rapport PDF, impression).

        Returns:
            Le chemin du fichier à afficher.
        """
        return self.logo_on_dark if dark_background else self.logo_on_light

    def missing(self) -> dict[str, Path]:
        """Fichiers de marque déclarés mais absents du disque.

        Returns:
            Les entrées `{nom_lisible: chemin}` introuvables.
        """
        candidates = {
            "logo (fond sombre)": self.logo_on_dark,
            "logo (fond clair)": self.logo_on_light,
            "favicon": self.favicon,
        }
        return {label: path for label, path in candidates.items() if not path.is_file()}


BRAND: Final[BrandConfig] = BrandConfig()


@dataclass(frozen=True)
class UIConfig:
    """Paramètres de l'interface Streamlit.

    Attributes:
        page_title: Titre de l'onglet navigateur.
        page_icon: Emoji d'onglet, utilisé **en repli** si `BRAND.favicon` est
            introuvable. Un onglet sans icône vaut mieux qu'une application qui
            refuse de démarrer parce qu'il manque un fichier décoratif.
        layout: Disposition Streamlit.
        show_confidence: Afficher la confiance sur les boîtes.
        show_track_id: Afficher l'ID de suivi sur les boîtes.
        box_thickness: Épaisseur des annotations.
        zone_opacity: Opacité du remplissage des zones [0-1].
        box_color: Couleur BGR d'une boîte sans anomalie. La couleur d'une boîte
            **en infraction** n'est pas ici : elle vient de la zone concernée
            (`ZoneConfig.color`), pour qu'une zone rouge et une zone orange se
            distinguent aussi sur les objets qu'elles signalent.
        label_scale: Facteur d'échelle du texte des étiquettes (`cv2.putText`).
        label_offset_px: Décalage vertical de l'étiquette au-dessus de la boîte.
    """

    page_title: str = "SentinelReport"
    page_icon: str = "\U0001f6e1"
    layout: str = "wide"
    show_confidence: bool = True
    show_track_id: bool = True
    box_thickness: int = 2
    zone_opacity: float = 0.25
    box_color: tuple[int, int, int] = (0, 200, 0)
    label_scale: float = 0.5
    label_offset_px: int = 6


UI: Final[UIConfig] = UIConfig()


# ---------------------------------------------------------------------------
# 10. Journalisation
# ---------------------------------------------------------------------------

LOG_LEVEL: Final[str] = os.environ.get("SENTINEL_LOG_LEVEL", "INFO")
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
