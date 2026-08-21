"""SentinelReport — détection d'incidents de sécurité par vision par ordinateur.

Le paquet expose les briques du pipeline. Regrouper les modules dans un paquet
(plutôt que de les laisser à la racine) évite les collisions de noms avec des
modules tiers (`tracker`, `events`, `report` sont des noms très courants) et rend
les imports explicites : `from sentinel.detector import Detector`.

Ce que `__all__` dit, et ce qu'il ne dit pas
---------------------------------------------
La liste ci-dessous décrit l'**ordre du pipeline**, pas un classement par
importance : une source produit des images, un détecteur en tire des détections,
un tracker leur donne une mémoire, une géométrie les situe, un moteur les
qualifie, une chronologie les raconte, un générateur les met en forme. Lire
`__all__` doit suffire à retrouver le sens de lecture — l'information ne remonte
jamais.

Les noms privés restent privés : `_write` et `_to_latin1` de `report.py` sont
partagés avec `session_report.py` parce qu'ils règlent une fois pour toutes le
piège du curseur `multi_cell` et la substitution des caractères hors latin-1,
mais ils ne font pas partie du contrat du paquet.
"""

from __future__ import annotations

__version__ = "0.1.0"

# -- Contrat de données -------------------------------------------------------
from sentinel.detection import Detection

# -- Erreurs métier -----------------------------------------------------------
from sentinel.exceptions import (
    ModelLoadError,
    ReportGenerationError,
    SentinelError,
    SourceDisconnectedError,
    VideoSourceError,
    ZoneConfigurationError,
)

# -- Le pipeline, dans son sens de lecture ------------------------------------
from sentinel.source import Frame, VideoSource, detect_kind
from sentinel.detector import Detector
from sentinel.tracker import TrackedObject, Tracker
from sentinel.zones import ZoneManager
from sentinel.occupancy import OccupancyChange, ZoneOccupancy
from sentinel.crossing import LineCounter, LineCrossing
from sentinel.events import Event, EventEngine, PriorityScore
from sentinel.evidence import EvidenceWriter, blur_bystanders
from sentinel.timeline import SessionContext, Timeline, TimelineFact, TimelineSlice
from sentinel.report import ReportGenerator
from sentinel.session_report import SessionReportGenerator, sort_by_priority

__all__ = [
    # Contrat de données
    "Detection",
    # Source
    "Frame",
    "VideoSource",
    "detect_kind",
    # Perception
    "Detector",
    # Mémoire temporelle
    "TrackedObject",
    "Tracker",
    # Géométrie
    "ZoneManager",
    # Occupation et flux
    "OccupancyChange",
    "ZoneOccupancy",
    "LineCounter",
    "LineCrossing",
    # Règles métier
    "Event",
    "EventEngine",
    "PriorityScore",
    # Preuves
    "EvidenceWriter",
    "blur_bystanders",
    # Chronologie
    "SessionContext",
    "Timeline",
    "TimelineFact",
    "TimelineSlice",
    # Restitution
    "ReportGenerator",
    "SessionReportGenerator",
    "sort_by_priority",
    # Erreurs
    "ModelLoadError",
    "ReportGenerationError",
    "SentinelError",
    "SourceDisconnectedError",
    "VideoSourceError",
    "ZoneConfigurationError",
    "__version__",
]
