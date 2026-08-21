"""Rapport de session à deux niveaux — extension de `ReportGenerator`.

Le rapport d'incident isolé (`report.py`) répond à « que s'est-il passé à
21 h 47 ? ». Celui-ci répond à une autre question, qui est celle d'un chef de
poste en fin de service : **« que s'est-il passé pendant la surveillance, et par
quoi dois-je commencer ? »**

D'où deux parties, et pas une de plus :

* **Partie 1 — Incidents prioritaires.** Les incidents triés par score
  décroissant. C'est la liste d'action : le premier de la liste est celui à
  regarder en premier, et la justification du score dit pourquoi.
* **Partie 2 — Chronologie.** Le déroulé par tranches de temps vidéo, ne
  retenant que les faits marquants. C'est la mise en récit : elle raconte
  l'enchaînement que la partie 1, triée par gravité, désordonne nécessairement.

Les deux parties disent la même chose dans deux ordres différents, et c'est
voulu : trier par priorité fait perdre la causalité, trier par temps fait perdre
l'urgence.

Ce module reste **purement présentation**. Il ne calcule aucun score — ils sont
portés par les `Event` — et n'établit aucun fait — ils viennent de `Timeline`.
Il trie, met en forme, et écrit.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import config
from sentinel.events import Event
from sentinel.exceptions import ReportGenerationError
from sentinel.report import ReportGenerator, _to_latin1, _write
from sentinel.timeline import SessionContext, Timeline, format_duration

logger = logging.getLogger(__name__)

MENTION_BROUILLON = (
    "Document généré automatiquement par analyse vidéo. Il constitue un "
    "BROUILLON : sa validation par un opérateur humain est obligatoire avant "
    "toute exploitation ou transmission. Le système ne juge aucune faute et "
    "n'identifie personne."
)


def sort_by_priority(events: Sequence[Event]) -> list[Event]:
    """Trie les incidents du plus prioritaire au moins prioritaire.

    Le tri est **total et déterministe** : à score égal, l'incident le plus
    récent d'abord, puis l'identifiant. Sans cette règle de départage, deux
    exécutions pourraient présenter les mêmes incidents dans un ordre différent
    et le rapport cesserait d'être reproductible.

    Args:
        events: Incidents à trier.

    Returns:
        Une nouvelle liste triée ; l'entrée n'est pas modifiée.
    """
    return sorted(
        events,
        key=lambda event: (
            -(event.priority.value if event.priority else 0.0),
            -event.video_time,
            event.event_id,
        ),
    )


class SessionReportGenerator(ReportGenerator):
    """Produit le rapport de session, en texte et en PDF.

    Hérite de `ReportGenerator` : les gabarits par incident, l'encodage latin-1
    et l'export CSV restent partagés. Ce qui s'ajoute ici est la **structure du
    document**, pas la façon de décrire un incident.

    Fonctionne indifféremment sur une vidéo terminée et sur une surveillance en
    cours : rien dans la génération ne suppose que la session est close, ce qui
    permet d'exporter un état à tout moment.
    """

    # -- En-tête --------------------------------------------------------------

    def header_lines(
        self,
        events: Sequence[Event],
        timeline: Timeline,
        context: SessionContext | None = None,
        *,
        generated_at: datetime | None = None,
    ) -> list[str]:
        """Compose l'en-tête : période, source, réglages, décompte.

        Args:
            events: Incidents de la session.
            timeline: Chronologie observée.
            context: Description de la session. `None` = valeurs par défaut.
            generated_at: Date d'édition. `None` = maintenant.

        Returns:
            Les lignes de l'en-tête.
        """
        contexte = context or SessionContext()
        edition = generated_at or datetime.now()

        debut = timeline.started_at
        fin = timeline.ended_at
        if debut and fin:
            periode = (
                f"{debut.strftime('%d/%m/%Y %H:%M:%S')} - {fin.strftime('%H:%M:%S')}"
            )
        else:
            periode = "non renseignée"
        if contexte.is_live:
            periode += " (surveillance en cours)"

        niveaux = self._count_by_level(events)
        repartition = (
            ", ".join(f"{niveau} : {nombre}" for niveau, nombre in niveaux.items())
            or "aucun"
        )

        lignes = [
            f"{self.config.organization} - RAPPORT DE SESSION",
            f"Site              : {self.config.site_name}",
            f"Opérateur         : {self.config.operator}",
            f"Édité le          : {edition.strftime('%d/%m/%Y à %H:%M:%S')}",
            f"Période analysée  : {periode}",
            f"Durée de vidéo    : {format_duration(timeline.duration_s)}",
        ]
        lignes += contexte.settings_lines()
        lignes += [
            f"Incidents         : {len(events)}",
            f"Par priorité      : {repartition}",
        ]
        return lignes

    @staticmethod
    def _count_by_level(events: Sequence[Event]) -> dict[str, int]:
        """Compte les incidents par niveau de priorité, du plus grave au moins grave.

        Args:
            events: Incidents à agréger.

        Returns:
            Un dictionnaire ordonné `{niveau: nombre}`, sans les niveaux absents.
        """
        ordre = (
            config.PriorityLevel.CRITICAL,
            config.PriorityLevel.HIGH,
            config.PriorityLevel.MEDIUM,
            config.PriorityLevel.LOW,
        )
        compte: dict[str, int] = {}
        for niveau in ordre:
            nombre = sum(
                1 for event in events if event.priority and event.priority.level is niveau
            )
            if nombre:
                compte[niveau.value] = nombre
        return compte

    # -- Rapport texte --------------------------------------------------------

    def generate_session_report(
        self,
        events: Sequence[Event],
        timeline: Timeline,
        context: SessionContext | None = None,
        *,
        generated_at: datetime | None = None,
    ) -> str:
        """Assemble le rapport complet en texte.

        Args:
            events: Incidents de la session.
            timeline: Chronologie observée.
            context: Description de la session.
            generated_at: Date d'édition.

        Returns:
            Le rapport structuré, prêt à être affiché ou exporté.
        """
        separateur = "=" * 74
        lignes = [separateur]
        lignes += self.header_lines(events, timeline, context, generated_at=generated_at)
        lignes += [separateur, ""]

        lignes += ["PARTIE 1 - INCIDENTS PRIORITAIRES", "-" * 74]
        if not events:
            lignes.append("Aucun incident détecté sur la période.")
        for rang, event in enumerate(sort_by_priority(events), start=1):
            lignes += self._incident_block(rang, event)

        lignes += ["", "PARTIE 2 - CHRONOLOGIE", "-" * 74]
        lignes.append(
            f"Faits marquants par tranche de "
            f"{timeline.settings.slice_seconds:.0f} s de temps vidéo. "
            "Les tranches sans changement ne sont pas listées."
        )
        lignes.append("")
        tranches = timeline.slices()
        if not tranches:
            lignes.append("Aucun mouvement enregistré.")
        for tranche in tranches:
            lignes.append(f"[{tranche.label}]")
            for fait in tranche.facts:
                lignes.append(f"    - {fait.label}")
            if tranche.omitted:
                lignes.append(f"    - ... et {tranche.omitted} autre(s) fait(s)")

        lignes += ["", separateur, MENTION_BROUILLON, separateur]
        return "\n".join(lignes)

    def _incident_block(self, rank: int, event: Event) -> list[str]:
        """Met en forme un incident de la partie 1.

        Args:
            rank: Rang dans la liste triée.
            event: Incident à décrire.

        Returns:
            Les lignes du bloc.
        """
        priorite = (
            event.priority.level.value.upper() if event.priority else "non calculée"
        )
        lignes = [
            "",
            f"{rank}. [{priorite}] {event.event_id} - "
            f"{event.event_type.value.replace('_', ' ')}",
            f"   Horodatage    : {event.timestamp.strftime('%d/%m/%Y %H:%M:%S')} "
            f"(t = {format_duration(event.video_time)})",
            f"   Objet         : #{event.track_id} {event.class_name}",
            f"   Zone          : {event.zone_name or 'hors zone'}",
            f"   Durée         : {event.duration_s:.0f} s",
        ]
        if event.priority:
            lignes.append(f"   Score         : {event.priority.explanation}")
        lignes.append(
            f"   Preuve        : {event.evidence_path or 'non disponible'}"
        )
        return lignes

    # -- Export PDF -----------------------------------------------------------

    def export_session_pdf(
        self,
        events: Sequence[Event],
        timeline: Timeline,
        context: SessionContext | None = None,
        output_path: Path | None = None,
        *,
        generated_at: datetime | None = None,
    ) -> Path:
        """Exporte le rapport de session en PDF.

        Args:
            events: Incidents de la session.
            timeline: Chronologie observée.
            context: Description de la session.
            output_path: Destination. `None` = `reports/rapport-session.pdf`.
            generated_at: Date d'édition.

        Returns:
            Le chemin du fichier écrit.

        Raises:
            ReportGenerationError: Si l'écriture échoue.
        """
        destination = output_path or self.config.directory / "rapport-session.pdf"

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            self._pdf_header(pdf, "RAPPORT DE SESSION")

            pdf.set_font("Helvetica", "", 9)
            for ligne in self.header_lines(
                events, timeline, context, generated_at=generated_at
            )[1:]:
                _write(pdf, 5, ligne)
            pdf.ln(3)

            # Partie 1
            pdf.set_font("Helvetica", "B", 12)
            _write(pdf, 7, "PARTIE 1 - INCIDENTS PRIORITAIRES")
            pdf.set_font("Helvetica", "", 9)
            if not events:
                _write(pdf, 5, "Aucun incident détecté sur la période.")
            for rang, event in enumerate(sort_by_priority(events), start=1):
                pdf.set_font("Helvetica", "B", 10)
                niveau = event.priority.level.value.upper() if event.priority else "-"
                _write(
                    pdf,
                    6,
                    f"{rang}. [{niveau}] {event.event_id} - "
                    f"{event.event_type.value.replace('_', ' ')}",
                )
                pdf.set_font("Helvetica", "", 9)
                for ligne in self._incident_block(rang, event)[2:]:
                    _write(pdf, 5, ligne.strip())
                self._pdf_evidence(pdf, event)
                pdf.ln(2)

            # Partie 2
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            _write(pdf, 7, "PARTIE 2 - CHRONOLOGIE")
            pdf.set_font("Helvetica", "I", 8)
            _write(
                pdf,
                5,
                f"Faits marquants par tranche de "
                f"{timeline.settings.slice_seconds:.0f} s de temps vidéo.",
            )
            pdf.ln(1)
            pdf.set_font("Helvetica", "", 9)
            tranches = timeline.slices()
            if not tranches:
                _write(pdf, 5, "Aucun mouvement enregistré.")
            for tranche in tranches:
                pdf.set_font("Helvetica", "B", 9)
                _write(pdf, 5, f"[{tranche.label}]")
                pdf.set_font("Helvetica", "", 9)
                for fait in tranche.facts:
                    _write(pdf, 5, f"    - {fait.label}")
                if tranche.omitted:
                    _write(pdf, 5, f"    - ... et {tranche.omitted} autre(s) fait(s)")

            pdf.ln(3)
            pdf.set_font("Helvetica", "I", 8)
            _write(pdf, 5, MENTION_BROUILLON)

            destination.parent.mkdir(parents=True, exist_ok=True)
            pdf.output(str(destination))
        except ReportGenerationError:
            raise
        except Exception as exc:
            raise ReportGenerationError(
                f"Écriture du rapport de session impossible ({destination}) : {exc}"
            ) from exc

        logger.info(
            "Rapport de session écrit : %s (%d incidents, %d tranches)",
            destination,
            len(events),
            len(timeline.slices()),
        )
        return destination

    # -- Export CSV -----------------------------------------------------------

    def export_timeline_csv(
        self, timeline: Timeline, output_path: Path | None = None
    ) -> Path:
        """Exporte la chronologie en CSV, une ligne par fait.

        Complète `export_csv()`, qui exporte les incidents. Les deux fichiers
        répondent à des besoins distincts : l'un se prête au tri et au comptage,
        l'autre à la reconstitution d'un déroulé.

        Args:
            timeline: Chronologie à exporter.
            output_path: Destination. `None` = `reports/chronologie.csv`.

        Returns:
            Le chemin du fichier écrit.

        Raises:
            ReportGenerationError: Si la chronologie est vide ou l'écriture
                impossible.
        """
        tranches = timeline.slices()
        if not tranches:
            raise ReportGenerationError("Aucun fait à exporter dans la chronologie.")

        destination = output_path or self.config.directory / "chronologie.csv"
        lignes = [
            {
                "tranche": tranche.label,
                "debut_s": round(tranche.start_s, 1),
                "temps_video_s": round(fait.video_time, 1),
                "nature": fait.kind,
                "fait": fait.label,
            }
            for tranche in tranches
            for fait in tranche.facts
        ]

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # `utf-8-sig` : sans BOM, Excel casse les accents à l'ouverture.
            with destination.open("w", encoding="utf-8-sig", newline="") as flux:
                writer = csv.DictWriter(flux, fieldnames=list(lignes[0]))
                writer.writeheader()
                writer.writerows(lignes)
        except OSError as exc:
            raise ReportGenerationError(
                f"Écriture de la chronologie impossible ({destination}) : {exc}"
            ) from exc

        logger.info("Chronologie CSV écrite : %s (%d faits)", destination, len(lignes))
        return destination
