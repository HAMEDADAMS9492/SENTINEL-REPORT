"""Doublures de zones partagées par les tests du moteur de règles.

Pourquoi une doublure commune plutôt qu'une par fichier
--------------------------------------------------------
`EventEngine` interroge désormais le gestionnaire de zones sur quatre points :
les noms restreints, les zones qui acceptent un type d'incident, la règle
applicable dans une zone, et le plus court délai de garde. Une doublure par
fichier de test aurait multiplié les versions de ce contrat, et une doublure qui
ment sur l'interface est pire que pas de doublure du tout : elle fait passer des
tests que le vrai code ferait échouer.

Cette doublure reste **minimale** — pas de polygone, pas de conversion en pixels,
pas d'OpenCV. C'est ce qui permet aux tests du moteur de tourner en millisecondes
sans vidéo ni résolution d'image.
"""

from __future__ import annotations

from typing import Sequence

import config


class FakeZones:
    """Gestionnaire de zones réduit au contrat que le moteur consomme.

    Construite depuis de vraies `config.SurveillanceZone`, elle applique donc les
    **vraies** règles de typage et de surcharge — seule la géométrie est absente.
    """

    def __init__(self, zones: Sequence[config.SurveillanceZone]) -> None:
        """Prépare la doublure.

        Args:
            zones: Zones surveillées, telles que la configuration les décrirait.
        """
        self._zones = tuple(zones)
        self._by_name = {zone.name: zone for zone in self._zones}

    # -- Fabriques de confort -------------------------------------------------

    @classmethod
    def restricted(cls, *names: str) -> "FakeZones":
        """Zones interdites : toutes les règles y sont actives.

        C'est l'équivalent de l'ancien `restricted=True`, et le comportement de
        la zone plein cadre livrée par défaut.
        """
        return cls([_zone(nom, config.ZoneType.FORBIDDEN) for nom in names])

    @classmethod
    def of_types(cls, **types: config.ZoneType) -> "FakeZones":
        """Zones nommées, chacune de son type.

        Exemple : `FakeZones.of_types(Quai=ZoneType.FORBIDDEN, Hall=ZoneType.TRANSIT)`.
        """
        return cls([_zone(nom, genre) for nom, genre in types.items()])

    # -- Contrat consommé par EventEngine -------------------------------------

    @property
    def names(self) -> list[str]:
        """Noms de toutes les zones surveillées."""
        return [zone.name for zone in self._zones]

    def restricted_zone_names(self) -> list[str]:
        """Zones où la seule présence constitue une infraction."""
        return [zone.name for zone in self._zones if zone.restricted]

    def zone(self, name: str) -> config.SurveillanceZone | None:
        """Zone portant ce nom, ou `None`."""
        return self._by_name.get(name)

    def zones_handling(self, event_type: config.EventType) -> list[str]:
        """Zones qui acceptent de lever ce type d'incident."""
        return [zone.name for zone in self._zones if zone.handles(event_type)]

    def rule_for(self, zone_name: str, rule: config.EventRule) -> config.EventRule:
        """Règle telle qu'elle s'applique dans une zone."""
        zone = self._by_name.get(zone_name)
        return rule if zone is None else zone.rule_for(rule)

    def shortest_cooldown(self, rule: config.EventRule) -> float:
        """Plus court délai de garde applicable, toutes zones confondues."""
        delais = [
            zone.rule_for(rule).cooldown_s
            for zone in self._zones
            if zone.handles(rule.event_type)
        ]
        return min([rule.cooldown_s, *delais])

    def color_for(self, name: str) -> tuple[int, int, int] | None:
        """Couleur déclarée par une zone."""
        zone = self._by_name.get(name)
        return None if zone is None else zone.color


def _zone(
    name: str,
    zone_type: config.ZoneType = config.ZoneType.FORBIDDEN,
    **surcharges: float,
) -> config.SurveillanceZone:
    """Zone plein cadre d'un type donné, avec surcharges éventuelles.

    La géométrie est sans importance pour les tests du moteur : ce qui compte
    est le type et les seuils.
    """
    return config.SurveillanceZone(
        name=name,
        polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        zone_type=zone_type,
        **surcharges,
    )


zone = _zone
