"""Tests de fumée — SQUELETTE (étoffés au fil des étapes).

Objectif : vérifier que la configuration est cohérente et que les modules
s'importent, **sans charger le modèle YOLO ni lire de vidéo**. Ces tests doivent
tourner en moins d'une seconde sur n'importe quelle machine.

Lancement : `pytest`
"""

from __future__ import annotations

import pytest

import config


def test_config_paths_exist() -> None:
    """Les dossiers déclarés dans la configuration sont créés à l'import."""
    assert config.EVIDENCE_DIR.is_dir()
    assert config.REPORTS_DIR.is_dir()


def test_thresholds_are_in_range() -> None:
    """Les seuils du modèle sont dans des intervalles valides."""
    assert 0.0 < config.MODEL.confidence <= 1.0
    assert 0.0 < config.MODEL.iou <= 1.0
    assert config.MODEL.image_size % 32 == 0


def test_zone_polygons_are_normalized() -> None:
    """Chaque zone a au moins 3 sommets, tous dans [0, 1]."""
    for zone in config.ZONES:
        assert len(zone.polygon) >= 3, f"Zone '{zone.name}' : moins de 3 sommets."
        for x, y in zone.polygon:
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, f"Zone '{zone.name}' hors bornes."


def test_zone_names_are_unique() -> None:
    """Deux zones ne peuvent pas porter le même nom (elles servent de clés)."""
    names = [zone.name for zone in config.ZONES]
    assert len(names) == len(set(names))


def test_event_rules_reference_known_classes() -> None:
    """Toute classe citée dans une règle est bien surveillée par le détecteur."""
    tracked = set(config.MODEL.tracked_classes)
    for rule in config.EVENT_RULES:
        for class_name in rule.classes:
            assert class_name in tracked, (
                f"La règle {rule.event_type.value} cible '{class_name}', "
                "absente de MODEL.tracked_classes."
            )


def test_cooldown_exceeds_duration() -> None:
    """Le délai de garde doit dépasser la durée de déclenchement.

    Sinon un même objet peut redéclencher l'événement immédiatement après, ce qui
    annule l'effet de l'anti-rebond.
    """
    for rule in config.EVENT_RULES:
        assert rule.cooldown_s > rule.min_duration_s, rule.event_type.value


def test_brand_assets_are_present() -> None:
    """Les trois fichiers d'identité visuelle sont là où la configuration les attend."""
    assert config.BRAND.missing() == {}


def test_logo_orientation_matches_its_background() -> None:
    """Chaque logo porte bien l'encre adaptée au fond auquel il est destiné.

    Ce test existe parce que l'erreur inverse — la version « fond clair »
    (encre anthracite) posée sur l'en-tête anthracite — produit un logo
    **invisible** que ni l'interpréteur ni une relecture de code ne signalent.

    Les deux couleurs viennent de la charte : `#F8FAFC` pour l'encre claire,
    `#0F172A` pour l'anthracite. À mettre à jour si la charte change.
    """
    light_ink, anthracite = "#F8FAFC", config.BRAND.anthracite

    on_dark = config.BRAND.logo_for(dark_background=True).read_text(encoding="utf-8")
    on_light = config.BRAND.logo_for(dark_background=False).read_text(encoding="utf-8")

    assert light_ink in on_dark, "Le logo des fonds sombres doit être en encre claire."
    assert light_ink not in on_light, "Le logo des fonds clairs ne doit pas être en encre claire."
    assert anthracite in on_light, "Le logo des fonds clairs doit être en encre anthracite."


@pytest.mark.parametrize(
    "module_name",
    [
        "sentinel.detection",
        "sentinel.detector",
        "sentinel.tracker",
        "sentinel.zones",
        "sentinel.events",
        "sentinel.report",
    ],
)
def test_modules_import(module_name: str) -> None:
    """Chaque module du paquet s'importe sans erreur."""
    __import__(module_name)


# TODO (étape 2) : test_detector_parses_results — construire un faux objet
#   « results » et vérifier que _parse_results produit les bonnes Detection,
#   sans charger de modèle.
# TODO (étape 3) : test_tracker_dwell_time — simuler 3 frames et vérifier le
#   chronomètre de zone ; test_tracker_purges_expired_objects.
# TODO (étape 4) : test_zone_membership — point clairement dedans / dehors.
# TODO (étape 5) : test_cooldown_blocks_duplicate_events.
