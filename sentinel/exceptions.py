"""Exceptions du domaine SentinelReport.

Une hiérarchie d'exceptions maison permet à l'interface (`app.py`) d'attraper
`SentinelError` et d'afficher un message clair à l'utilisateur, au lieu de laisser
remonter une `RuntimeError` d'Ultralytics ou une `AttributeError` d'OpenCV que
personne ne peut interpréter.
"""

from __future__ import annotations


class SentinelError(Exception):
    """Classe de base de toutes les erreurs métier de SentinelReport."""


class ModelLoadError(SentinelError):
    """Le modèle YOLO n'a pas pu être chargé.

    Causes typiques : fichier de poids introuvable, `ultralytics` non installé,
    téléchargement impossible (pas de réseau au premier lancement), GPU demandé
    mais indisponible.
    """


class VideoSourceError(SentinelError):
    """La source vidéo (fichier, webcam ou flux réseau) est inexploitable.

    Couvre l'échec d'ouverture : fichier introuvable, webcam déjà utilisée par
    une autre application, URL RTSP injoignable, identifiants refusés.
    """


class SourceDisconnectedError(VideoSourceError):
    """Un flux en direct s'est interrompu et n'a pas pu être rétabli.

    Distinguée de `VideoSourceError` parce que la conduite à tenir diffère : une
    source qui n'a jamais démarré relève d'une erreur de configuration à
    corriger, tandis qu'un flux perdu après plusieurs reconnexions relève d'un
    incident réseau — l'analyse déjà effectuée reste valide et exploitable.
    """


class ZoneConfigurationError(SentinelError):
    """Une zone de `config.ZONES` est invalide (moins de 3 sommets, coordonnées
    hors de l'intervalle [0, 1], noms dupliqués...)."""


class ReportGenerationError(SentinelError):
    """Le rapport n'a pas pu être produit (gabarit manquant, écriture PDF impossible)."""
