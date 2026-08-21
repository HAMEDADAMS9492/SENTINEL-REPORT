"""`config.validate()` — une configuration fautive ne doit pas démarrer.

Le défaut visé
---------------
Une faute de frappe dans `config.py` ne produit pas une erreur : elle produit une
**règle silencieusement inopérante**. Une zone dont le polygone déborde de [0, 1]
ne couvre rien ; une ligne dégénérée ne compte jamais ; un délai de garde plus
court que la durée de déclenchement fait crier le système sans arrêt. Ces défauts
ne se voient qu'en relisant un rapport vide ou saturé — c'est-à-dire après
l'analyse, quand il est trop tard.

La validation tourne à l'import du module. Le coût est de quelques microsecondes,
le bénéfice est qu'une configuration fautive s'arrête tout de suite, avec le nom
de l'élément en cause. Un message qui dirait seulement « configuration invalide »
obligerait à relire neuf cents lignes.
"""

from __future__ import annotations

from datetime import time

import pytest

import config


def _valide(monkeypatch: pytest.MonkeyPatch, **remplacements) -> None:
    """Rejoue `config.validate()` avec des valeurs substituées."""
    for nom, valeur in remplacements.items():
        monkeypatch.setattr(config, nom, valeur, raising=True)
    config.validate()


def _zone(**kwargs) -> config.SurveillanceZone:
    """Zone plein cadre, valide par défaut."""
    defauts = dict(
        name="Quai",
        polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        zone_type=config.ZoneType.FORBIDDEN,
    )
    defauts.update(kwargs)
    return config.SurveillanceZone(**defauts)


def _ligne(**kwargs) -> config.CrossingLine:
    """Ligne valide par défaut."""
    defauts = dict(name="Porte", start=(0.2, 0.5), end=(0.8, 0.5))
    defauts.update(kwargs)
    return config.CrossingLine(**defauts)


# ---------------------------------------------------------------------------
# La configuration livrée est valide
# ---------------------------------------------------------------------------


def test_the_shipped_configuration_passes() -> None:
    """Le test le plus important : ce qui est livré démarre."""
    config.validate()


def test_validation_runs_at_import() -> None:
    """Rien n'est à appeler à la main : importer `config` suffit à contrôler."""
    source = (config.BASE_DIR / "config.py").read_text(encoding="utf-8")

    assert "\nvalidate()\n" in source, "L'appel au chargement a disparu."


# ---------------------------------------------------------------------------
# Règles
# ---------------------------------------------------------------------------


def test_a_cooldown_shorter_than_the_duration_is_refused(monkeypatch) -> None:
    """L'invariant fondateur de l'anti-rebond.

    Un même objet redéclencherait en boucle, et le système produirait un rapport
    par frame — exactement ce que les trois filtres existent pour empêcher.
    """
    fautive = config.EventRule(
        event_type=config.EventType.INTRUSION, min_duration_s=60.0, cooldown_s=30.0
    )

    with pytest.raises(ValueError, match="cooldown_s"):
        _valide(monkeypatch, EVENT_RULES=(fautive,))


def test_a_negative_duration_is_refused(monkeypatch) -> None:
    """Une durée négative rendrait toute condition vraie."""
    fautive = config.EventRule(
        event_type=config.EventType.INTRUSION, min_duration_s=-1.0, cooldown_s=60.0
    )

    with pytest.raises(ValueError, match="négatif"):
        _valide(monkeypatch, EVENT_RULES=(fautive,))


def test_an_unknown_class_is_refused(monkeypatch) -> None:
    """Une règle qui cible « pistolet » ne se déclenchera jamais.

    C'est le pire défaut possible pour un système d'alerte, parce qu'il est
    **silencieux** — même raisonnement que l'avertissement affiché dans
    l'interface sur l'absence d'armes à feu dans COCO.
    """
    fautive = config.EventRule(
        event_type=config.EventType.INTRUSION, classes=("pistolet",), cooldown_s=60.0
    )

    with pytest.raises(ValueError, match="pistolet"):
        _valide(monkeypatch, EVENT_RULES=(fautive,))


def test_a_crowd_of_one_is_refused(monkeypatch) -> None:
    """Une surdensité à moins de deux occupants n'a pas de sens."""
    fautive = config.EventRule(
        event_type=config.EventType.OVERCROWDING, cooldown_s=60.0, min_occupancy=1
    )

    with pytest.raises(ValueError, match="min_occupancy"):
        _valide(monkeypatch, EVENT_RULES=(fautive,))


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------


def test_an_empty_zone_list_is_refused(monkeypatch) -> None:
    """Sans périmètre, il n'y a plus de « ici », donc plus de durée mesurable."""
    with pytest.raises(ValueError, match="Aucune zone"):
        _valide(monkeypatch, ZONES=())


def test_duplicate_zone_names_are_refused(monkeypatch) -> None:
    """Les noms servent de clés dans les rapports et les chronomètres."""
    with pytest.raises(ValueError, match="ce nom"):
        _valide(monkeypatch, ZONES=(_zone(), _zone()))


def test_a_polygon_with_two_vertices_is_refused(monkeypatch) -> None:
    """Deux points ne délimitent pas une surface."""
    with pytest.raises(ValueError, match="sommet"):
        _valide(monkeypatch, ZONES=(_zone(polygon=((0.0, 0.0), (1.0, 1.0))),))


@pytest.mark.parametrize("point", [(1.5, 0.5), (-0.1, 0.5), (0.5, 2.0)])
def test_coordinates_outside_the_unit_square_are_refused(monkeypatch, point) -> None:
    """Un polygone hors bornes ne couvre rien, en silence."""
    polygone = ((0.0, 0.0), (1.0, 0.0), point)

    with pytest.raises(ValueError, match="normalisées"):
        _valide(monkeypatch, ZONES=(_zone(polygon=polygone),))


def test_a_malformed_vertex_is_refused(monkeypatch) -> None:
    """Un sommet à trois coordonnées est une faute de frappe, pas une intention."""
    polygone = ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5, 0.5))

    with pytest.raises(ValueError, match="n'est pas"):
        _valide(monkeypatch, ZONES=(_zone(polygon=polygone),))


def test_an_incoherent_zone_override_is_refused(monkeypatch) -> None:
    """Les surcharges sont éprouvées contre toutes les règles applicables."""
    fautive = _zone(min_duration_s=10_000.0)

    with pytest.raises(ValueError, match="cooldown_s"):
        _valide(monkeypatch, ZONES=(fautive,))


def test_a_zone_pointing_at_a_missing_line_is_refused(monkeypatch) -> None:
    """Une référence morte produirait un comptage éternellement à zéro."""
    fautive = _zone(zone_type=config.ZoneType.COUNTING, crossing_line="Inexistante")

    with pytest.raises(ValueError, match="Inexistante"):
        _valide(monkeypatch, ZONES=(fautive,), CROSSING_LINES=())


def test_a_zone_pointing_at_an_existing_line_is_accepted(monkeypatch) -> None:
    """Le cas nominal du couple zone de comptage / ligne."""
    _valide(
        monkeypatch,
        ZONES=(_zone(zone_type=config.ZoneType.COUNTING, crossing_line="Porte"),),
        CROSSING_LINES=(_ligne(),),
    )


# ---------------------------------------------------------------------------
# Lignes
# ---------------------------------------------------------------------------


def test_a_degenerate_line_is_refused(monkeypatch) -> None:
    """Un point n'a pas de côté : aucun franchissement ne serait jamais compté."""
    fautive = _ligne(start=(0.5, 0.5), end=(0.5, 0.5))

    with pytest.raises(ValueError, match="confondues"):
        _valide(monkeypatch, CROSSING_LINES=(fautive,))


def test_line_coordinates_must_be_normalised(monkeypatch) -> None:
    """Même contrainte que les zones : la configuration ignore la résolution."""
    with pytest.raises(ValueError, match="normalisées"):
        _valide(monkeypatch, CROSSING_LINES=(_ligne(end=(1.4, 0.5)),))


def test_duplicate_line_names_are_refused(monkeypatch) -> None:
    """Les noms servent de clés au comptage."""
    with pytest.raises(ValueError, match="ce nom"):
        _valide(monkeypatch, CROSSING_LINES=(_ligne(), _ligne()))


def test_identical_direction_labels_are_refused(monkeypatch) -> None:
    """« 12 passage / 9 passage » ne se lit pas."""
    fautive = _ligne(positive_label="passage", negative_label="passage")

    with pytest.raises(ValueError, match="même libellé"):
        _valide(monkeypatch, CROSSING_LINES=(fautive,))


# ---------------------------------------------------------------------------
# Horaires
# ---------------------------------------------------------------------------


def test_a_closing_time_before_the_opening_is_refused(monkeypatch) -> None:
    """En l'état, tout instant serait « hors horaires ».

    Le système signalerait alors une présence hors horaires à midi, ce que
    personne ne mettrait en doute avant d'avoir relu la configuration.
    """
    fautif = config.ScheduleConfig(opening=time(19, 0), closing=time(7, 30))

    with pytest.raises(ValueError, match="ouverture"):
        _valide(monkeypatch, SCHEDULE=fautif)


# ---------------------------------------------------------------------------
# Les messages sont exploitables
# ---------------------------------------------------------------------------


def test_the_message_names_the_element_at_fault(monkeypatch) -> None:
    """« Configuration invalide » obligerait à relire neuf cents lignes."""
    fautive = _zone(name="Réserve arrière", min_duration_s=10_000.0)

    with pytest.raises(ValueError) as erreur:
        _valide(monkeypatch, ZONES=(fautive,))

    assert "Réserve arrière" in str(erreur.value)


def test_the_message_explains_the_consequence(monkeypatch) -> None:
    """Dire ce qui casse vaut mieux que dire ce qui est interdit."""
    fautive = config.EventRule(
        event_type=config.EventType.INTRUSION, min_duration_s=60.0, cooldown_s=30.0
    )

    with pytest.raises(ValueError) as erreur:
        _valide(monkeypatch, EVENT_RULES=(fautive,))

    assert "rapport par frame" in str(erreur.value)
