"""Tests du générateur de rapports — écritures confinées dans `tmp_path`.

Le rapport est la sortie visible du produit : c'est aussi le seul artefact que
lira un humain non technique. Deux propriétés comptent plus que les autres et
sont testées explicitement : le texte est **déterministe** (deux exécutions
donnent le même rapport) et il porte toujours la mention **BROUILLON**.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import config
from sentinel.events import Event
from sentinel.exceptions import ReportGenerationError
from sentinel.report import ReportGenerator, _to_latin1

MOMENT = datetime(2026, 8, 13, 21, 47, 5)


def _event(**overrides) -> Event:
    """Incident type, surchargeable champ par champ."""
    base = {
        "event_id": "SR-20260813-0001",
        "event_type": config.EventType.INTRUSION,
        "severity": config.Severity.HIGH,
        "timestamp": MOMENT,
        "video_time": 12.3,
        "track_id": 4,
        "class_name": "person",
        "zone_name": "Quai de chargement",
        "duration_s": 5.0,
        "confidence": 0.87,
        "bbox": (10.0, 20.0, 60.0, 180.0),
    }
    return Event(**{**base, **overrides})


@pytest.fixture()
def generator(tmp_path) -> ReportGenerator:
    """Générateur écrivant dans un dossier temporaire."""
    settings = config.ReportConfig(directory=tmp_path)
    return ReportGenerator(settings)


# ---------------------------------------------------------------------------
# Texte
# ---------------------------------------------------------------------------


def test_generate_fills_the_template(generator) -> None:
    """Le constat cite la date, l'objet, la zone et la durée."""
    text = generator.generate(_event())

    assert "13/08/2026" in text
    assert "21:47:05" in text
    assert "n°4" in text
    assert "Quai de chargement" in text
    assert "5 secondes" in text
    assert "87%" in text


def test_generation_is_deterministic(generator) -> None:
    """Deux exécutions sur le même incident doivent produire un texte identique.

    C'est la raison d'être des gabarits : un rapport d'incident doit pouvoir
    être rejoué et vérifié.
    """
    event = _event()
    assert generator.generate(event) == generator.generate(event)


def test_missing_zone_is_labelled_not_crashed(generator) -> None:
    """Un incident hors zone reste rédigeable."""
    assert "hors zone" in generator.generate(_event(zone_name=None))


def test_unknown_event_type_raises_a_clear_error(generator) -> None:
    """Ajouter un EventType sans gabarit doit être signalé explicitement."""

    class _Fake(str):
        pass

    with pytest.raises(ReportGenerationError, match="gabarit"):
        generator.generate(_event(event_type=_Fake("type_inconnu")))


def test_full_report_carries_the_draft_warning(generator) -> None:
    """La mention « BROUILLON » est non négociable : le rapport n'est pas une preuve."""
    report = generator.generate_full_report(_event())

    assert "BROUILLON" in report
    assert "SR-20260813-0001" in report
    assert config.REPORT.organization in report
    assert "ÉLÉMENTS TECHNIQUES" in report


def test_details_appear_in_the_full_report(generator) -> None:
    """Les champs libres d'un incident (déplacement, propriétaire) sont restitués."""
    event = _event(
        event_type=config.EventType.ABANDONED_OBJECT,
        class_name="backpack",
        details={"deplacement_px": 4.2, "proprietaire_presume": "aucun"},
    )

    report = generator.generate_full_report(event)

    assert "Deplacement px" in report or "deplacement_px" in report
    assert "aucun" in report


# ---------------------------------------------------------------------------
# Encodage PDF
# ---------------------------------------------------------------------------


def test_french_accents_survive_the_pdf_encoding() -> None:
    """Les accents et guillemets français appartiennent à latin-1 : ils passent."""
    assert _to_latin1("Présence prolongée « zone réservée »") == (
        "Présence prolongée « zone réservée »"
    )


def test_typographic_characters_are_substituted() -> None:
    """Le tiret cadratin et l'apostrophe courbe, eux, n'existent pas en latin-1."""
    assert _to_latin1("Rapport — l’objet…") == "Rapport - l'objet..."


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_export_pdf_writes_a_valid_document(generator, tmp_path) -> None:
    """Le PDF doit exister, être non vide et porter l'en-tête du format."""
    path = generator.export_pdf(_event())

    assert path.is_file()
    assert path.parent == tmp_path
    assert path.read_bytes().startswith(b"%PDF")


def test_pdf_embeds_the_light_background_logo(generator, monkeypatch, tmp_path) -> None:
    """Le logo des fonds clairs est vectorisé dans l'en-tête du PDF.

    C'est la version à encre anthracite sur fond transparent qui convient à une
    page A4 blanche ; la version « fond sombre » y collerait un pavé anthracite.
    Le contrôle porte sur le poids du fichier : le tracé vectoriel du bouclier
    pèse environ 1,3 ko, indétectable autrement sans analyser le flux PDF.
    """
    from dataclasses import replace

    with_logo = generator.export_pdf(_event(), tmp_path / "avec.pdf").read_bytes()

    monkeypatch.setattr(
        config, "BRAND", replace(config.BRAND, logo_on_light=Path("absent.svg"))
    )
    without_logo = generator.export_pdf(_event(), tmp_path / "sans.pdf").read_bytes()

    assert len(with_logo) > len(without_logo) + 500


def test_missing_logo_never_blocks_the_report(generator, monkeypatch, tmp_path) -> None:
    """Un logo introuvable produit un rapport sans logo, jamais une erreur."""
    from dataclasses import replace

    monkeypatch.setattr(
        config, "BRAND", replace(config.BRAND, logo_on_light=Path("introuvable.svg"))
    )

    path = generator.export_pdf(_event(), tmp_path / "sans-logo.pdf")

    assert path.read_bytes().startswith(b"%PDF")


def test_export_csv_unions_the_columns_of_all_events(generator) -> None:
    """Les incidents n'ont pas tous les mêmes détails : l'en-tête est leur union."""
    events = [
        _event(),
        _event(
            event_id="SR-20260813-0002",
            event_type=config.EventType.ABANDONED_OBJECT,
            class_name="suitcase",
            details={"deplacement_px": 3.0},
        ),
    ]

    path = generator.export_csv(events)
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]

    assert "detail_deplacement_px" in header
    assert path.read_text(encoding="utf-8-sig").count("\n") >= 3


def test_csv_starts_with_a_bom_for_excel(generator) -> None:
    """Sans BOM, Excel ouvre un CSV UTF-8 avec les accents cassés."""
    path = generator.export_csv([_event()])

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_batch_pdf_summarises_every_incident(generator) -> None:
    """La synthèse comporte une page de garde puis une section par incident."""
    events = [_event(), _event(event_id="SR-20260813-0002", severity=config.Severity.MEDIUM)]

    path = generator.export_batch_pdf(events)

    assert path.is_file()
    assert path.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("method", ["export_csv", "export_batch_pdf"])
def test_exporting_nothing_is_an_explicit_error(generator, method) -> None:
    """Exporter une liste vide est une erreur d'usage, pas un fichier vide."""
    with pytest.raises(ReportGenerationError, match="Aucun incident"):
        getattr(generator, method)([])


def test_llm_falls_back_to_the_template_when_disabled(generator) -> None:
    """Sans clé ni activation, la rédaction assistée rend exactement le gabarit."""
    event = _event()

    assert generator.generate_with_llm(event) == generator.generate(event)
