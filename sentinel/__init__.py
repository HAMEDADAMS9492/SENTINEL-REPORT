"""SentinelReport — détection d'incidents de sécurité par vision par ordinateur.

Le paquet expose les briques du pipeline. Regrouper les modules dans un paquet
(plutôt que de les laisser à la racine) évite les collisions de noms avec des
modules tiers (`tracker`, `events`, `report` sont des noms très courants) et rend
les imports explicites : `from sentinel.detector import Detector`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from sentinel.detection import Detection
from sentinel.detector import Detector
from sentinel.exceptions import (
    ModelLoadError,
    SentinelError,
    VideoSourceError,
    ZoneConfigurationError,
)
from sentinel.tracker import TrackedObject, Tracker

__all__ = [
    "Detection",
    "Detector",
    "ModelLoadError",
    "SentinelError",
    "TrackedObject",
    "Tracker",
    "VideoSourceError",
    "ZoneConfigurationError",
    "__version__",
]
