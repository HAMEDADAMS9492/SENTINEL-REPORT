"""Le rapport d'incident affiche le score — et ne le calcule pas.

L'asymétrie corrigée ici
------------------------
Le score de priorité était visible dans trois sorties sur quatre : le tableau à
l'écran, le journal CSV (via `Event.to_dict()`) et le rapport de session. Le
rapport d'incident isolé — le document qu'un opérateur imprime et joint à une
main courante — n'en disait rien. Un même incident portait donc un « critique »
à l'écran et aucune priorité sur le papier.

Le test canari du flux unidirectionnel
---------------------------------------
La correction est facile à faire *mal* : il suffit de recomposer le score dans
`report.py` à partir du type d'incident et de la durée. Le résultat serait juste
le premier jour et faux le jour où le barème de `config.SCORING` change — le
papier contredirait alors l'écran, et le rapport cesserait d'être opposable.

D'où la double vérification : le score doit **apparaître**, et `report.py` ne
doit contenir aucune trace de son calcul.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import config
from sentinel.events import Event, PriorityScore
from sentinel.report import ReportGenerator

T0 = datetime(2026, 8, 21, 22, 15, 0)


def _event(priority: PriorityScore | None) -> Event:
    """Incident minimal, avec ou sans score."""
    return Event(
        event_id="SR-0007",
        event_type=config.EventType.INTRUSION,
        severity=config.Severity.HIGH,
        timestamp=T0,
        video_time=94.5,
        track_id=3,
        class_name="person",
        zone_name="Quai de chargement",
        duration_s=42.0,
        confidence=0.87,
        bbox=(10.0, 20.0, 60.0, 220.0),
        priority=priority,
    )


SCORE = PriorityScore(
    150.0,
    config.PriorityLevel.CRITICAL,
    ("intrusion (100)", "durée 42 s (+20)", "hors horaires (x 1.5)"),
)


# ---------------------------------------------------------------------------
# Le score apparaît
# ---------------------------------------------------------------------------


def test_the_full_report_states_the_priority_level() -> None:
    """L'en-tête porte le niveau : c'est ce qu'on lit en diagonale."""
    texte = ReportGenerator().generate_full_report(_event(SCORE))

    assert "Priorité" in texte
    assert config.PriorityLevel.CRITICAL.value in texte


def test_the_full_report_states_the_score_value() -> None:
    """Le nombre figure, arrondi comme partout ailleurs."""
    assert "150" in ReportGenerator().generate_full_report(_event(SCORE))


def test_the_full_report_shows_how_the_score_was_obtained() -> None:
    """Sans le détail, un score n'est qu'une opinion chiffrée.

    Un opérateur qui conteste un classement doit pouvoir refaire l'addition ;
    c'est ce qui rend le score contestable, donc défendable.
    """
    texte = ReportGenerator().generate_full_report(_event(SCORE))

    for contribution in SCORE.contributions:
        assert contribution in texte, contribution


def test_the_screen_and_the_paper_tell_the_same_story() -> None:
    """La justification du rapport est **exactement** celle portée par l'incident.

    Pas « la même à peu près » : la même chaîne. C'est ce qui garantit qu'aucune
    reformulation locale ne s'est glissée entre les deux sorties.
    """
    event = _event(SCORE)

    assert event.priority.explanation in ReportGenerator().generate_full_report(event)


def test_an_unscored_incident_says_so_plainly() -> None:
    """Un incident sans score ne doit ni planter ni afficher « 0 ».

    Un zéro se lirait « rien à signaler », alors que l'information manque —
    deux situations qu'un rapport n'a pas le droit de confondre.
    """
    texte = ReportGenerator().generate_full_report(_event(None))

    assert "non calculée" in texte or "non calculé" in texte
    assert "0 pts" not in texte


def test_the_template_context_exposes_the_priority_fields() -> None:
    """Les gabarits peuvent citer la priorité comme n'importe quel autre champ."""
    contexte = ReportGenerator()._template_context(_event(SCORE))

    assert contexte["priorite"] == config.PriorityLevel.CRITICAL.value
    assert contexte["score"] == "150"
    assert contexte["justification"] == SCORE.explanation


def test_the_pdf_carries_the_priority(tmp_path: Path) -> None:
    """Le PDF n'est pas une sortie de seconde classe : il porte la même donnée."""
    generateur = ReportGenerator()
    avec = generateur.export_pdf(_event(SCORE), tmp_path / "avec.pdf")
    sans = generateur.export_pdf(_event(None), tmp_path / "sans.pdf")

    assert avec.is_file() and sans.is_file()
    # Le PDF avec priorité contient une ligne de plus : il est nécessairement
    # plus lourd. Comparer les tailles évite de dépendre d'un extracteur de
    # texte, donc d'ajouter une dépendance pour un test.
    assert avec.stat().st_size > sans.stat().st_size


# ---------------------------------------------------------------------------
# Le score n'est pas recalculé — le canari du flux unidirectionnel
# ---------------------------------------------------------------------------


def test_the_report_module_never_computes_a_score() -> None:
    """`report.py` ne contient aucune trace du calcul de priorité.

    Même verrou que celui posé sur `session_report.py`. Si le barème était
    réimplémenté ici, la même donnée serait produite à deux endroits et
    divergerait au premier ajustement de `config.SCORING`.
    """
    source = (Path(__file__).resolve().parents[1] / "sentinel" / "report.py").read_text(
        encoding="utf-8"
    )

    assert "PriorityScore(" not in source
    assert "points_for" not in source
    assert "level_for" not in source
    assert "SCORING" not in source


def test_the_report_module_does_not_import_the_scoring_types() -> None:
    """Seul `Event` traverse la frontière : le rapport lit un fait, pas un calcul."""
    source = (Path(__file__).resolve().parents[1] / "sentinel" / "report.py").read_text(
        encoding="utf-8"
    )

    assert "import PriorityScore" not in source
    assert "PriorityScore," not in source


@pytest.mark.parametrize(
    "niveau, valeur",
    [
        (config.PriorityLevel.LOW, 10.0),
        (config.PriorityLevel.MEDIUM, 45.0),
        (config.PriorityLevel.HIGH, 80.0),
        (config.PriorityLevel.CRITICAL, 160.0),
    ],
)
def test_the_report_repeats_whatever_level_it_is_given(
    niveau: config.PriorityLevel, valeur: float
) -> None:
    """Le rapport restitue le niveau porté par l'incident, sans le rejuger.

    Y compris un couple (niveau, score) incohérent : ce serait un défaut du
    moteur, pas du rapport, et le masquer ici rendrait le défaut indétectable.
    """
    event = _event(PriorityScore(valeur, niveau, ("test",)))

    texte = ReportGenerator().generate_full_report(event)

    assert niveau.value in texte
    assert f"{valeur:.0f}" in texte
