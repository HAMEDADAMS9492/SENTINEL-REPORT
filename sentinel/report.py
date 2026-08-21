"""Génération de rapports d'incident — ÉTAPE 6.

Rôle : transformer un `Event` (structure de données) en rapport lisible par un
humain — texte d'abord, PDF ensuite.

Décision structurante : le chemin nominal est **déterministe**, par gabarits de
texte. Un rapport d'incident doit être reproductible et vérifiable : deux
exécutions sur la même vidéo doivent produire exactement le même texte. La
rédaction assistée par LLM (`generate_with_llm`) est une couche optionnelle
**par-dessus des faits déjà établis** par la vision — jamais une source de faits
— avec repli automatique sur le gabarit en cas d'échec.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Sequence

import config
from sentinel.events import Event
from sentinel.exceptions import ReportGenerationError

logger = logging.getLogger(__name__)


# Gabarits français, un par type d'incident. Champs disponibles dans le format :
# event_id, timestamp, date, heure, class_name, track_id, zone_name, duration_s,
# confidence, organization, site_name, operator, severity.
TEMPLATES: dict[config.EventType, str] = {
    config.EventType.INTRUSION: (
        "Le {date} à {heure}, une présence de type « {class_name} » (objet suivi "
        "n°{track_id}) a été détectée dans la zone restreinte « {zone_name} » "
        "pendant {duration_s:.0f} secondes. Confiance de détection : {confidence:.0%}."
    ),
    config.EventType.LOITERING: (
        "Le {date} à {heure}, une présence de type « {class_name} » (objet suivi "
        "n°{track_id}) a stationné dans la zone « {zone_name} » pendant "
        "{duration_s:.0f} secondes, au-delà du seuil configuré."
    ),
    config.EventType.ABANDONED_OBJECT: (
        "Le {date} à {heure}, un objet de type « {class_name} » (objet suivi "
        "n°{track_id}) est demeuré immobile et sans surveillance dans la zone "
        "« {zone_name} » pendant {duration_s:.0f} secondes."
    ),
    config.EventType.AFTER_HOURS: (
        "Le {date} à {heure}, en dehors des horaires d'ouverture, une présence de "
        "type « {class_name} » (objet suivi n°{track_id}) a été détectée dans la "
        "zone « {zone_name} » pendant {duration_s:.0f} secondes."
    ),
}

# Caractères typographiques absents du jeu latin-1 utilisé par les polices de
# base de fpdf2. Les remplacer vaut mieux que d'embarquer une police Unicode de
# plusieurs mégaoctets dans le dépôt.
_PDF_SUBSTITUTIONS = {
    "—": "-",  # tiret cadratin
    "–": "-",  # tiret demi-cadratin
    "’": "'",  # apostrophe typographique
    "‘": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    " ": " ",  # espace insécable
    " ": " ",  # espace fine insécable
}


def _write(pdf, height: float, text: str) -> None:
    """Écrit un paragraphe et ramène le curseur à la marge gauche.

    Point de passage unique vers `multi_cell`, pour une raison précise : par
    défaut fpdf2 laisse le curseur **à droite** de la cellule qu'il vient
    d'écrire (`new_x=RIGHT`). L'appel suivant avec une largeur automatique
    dispose alors d'un espace nul et lève « Not enough horizontal space to
    render a single character ». En centralisant `new_x="LMARGIN",
    new_y="NEXT"`, le piège est réglé une fois pour toutes.

    Args:
        pdf: Instance `FPDF` en cours.
        height: Hauteur de ligne.
        text: Texte à écrire, translittéré au passage.
    """
    pdf.multi_cell(w=0, h=height, text=_to_latin1(text), new_x="LMARGIN", new_y="NEXT")


def _to_latin1(text: str) -> str:
    """Rend un texte imprimable par les polices de base de fpdf2.

    Les accents français (é, à, ç) et les guillemets « » appartiennent à
    latin-1 et passent tels quels ; seuls les signes typographiques hors jeu
    sont substitués. Ce qui resterait impossible à encoder est remplacé par
    « ? » plutôt que de faire échouer l'export : un rapport avec un caractère
    approximatif vaut mieux qu'un rapport absent.

    Args:
        text: Texte source.

    Returns:
        Un texte encodable en latin-1.
    """
    for source, replacement in _PDF_SUBSTITUTIONS.items():
        text = text.replace(source, replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


class ReportGenerator:
    """Produit les rapports d'incident, en texte puis en PDF.

    Utilisation prévue :
        >>> generator = ReportGenerator()
        >>> texte = generator.generate(event)
        >>> chemin = generator.export_pdf(event)
    """

    def __init__(self, report_config: config.ReportConfig | None = None) -> None:
        """Initialise le générateur.

        Args:
            report_config: Paramètres de rapport. `None` = `config.REPORT`.
        """
        self.config = config.REPORT if report_config is None else report_config
        self.config.directory.mkdir(parents=True, exist_ok=True)

    def generate(self, event: Event) -> str:
        """Rédige le corps du rapport à partir du gabarit correspondant.

        Args:
            event: Incident à décrire.

        Returns:
            Le texte du rapport.

        Raises:
            ReportGenerationError: Si aucun gabarit ne correspond au type
                d'incident ou si un champ du gabarit est manquant.
        """
        template = TEMPLATES.get(event.event_type)
        if template is None:
            raise ReportGenerationError(
                f"Aucun gabarit défini pour le type d'incident '{event.event_type}'. "
                "Ajoutez-le dans sentinel/report.py::TEMPLATES."
            )

        try:
            return template.format(**self._template_context(event))
        except KeyError as exc:
            raise ReportGenerationError(
                f"Champ {exc} absent du contexte pour l'incident {event.event_id}. "
                "Le gabarit et _template_context() ont divergé."
            ) from exc

    def _template_context(self, event: Event) -> dict[str, object]:
        """Construit le dictionnaire de champs attendu par les gabarits.

        Args:
            event: Incident source.

        Returns:
            Les valeurs à injecter (dates déjà formatées en français, zone
            remplacée par « hors zone » si `None`, etc.).
        """
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "date": event.timestamp.strftime("%d/%m/%Y"),
            "heure": event.timestamp.strftime("%H:%M:%S"),
            "class_name": event.class_name,
            "track_id": event.track_id,
            "zone_name": event.zone_name or "hors zone",
            "duration_s": event.duration_s,
            "confidence": event.confidence,
            "organization": self.config.organization,
            "site_name": self.config.site_name,
            "operator": self.config.operator,
            "severity": event.severity.value,
            **event.details,
        }

    def generate_full_report(self, event: Event) -> str:
        """Assemble un rapport complet : en-tête, corps, métadonnées, pied de page.

        Args:
            event: Incident à décrire.

        Returns:
            Le rapport structuré prêt à être affiché ou exporté.
        """
        context = self._template_context(event)
        separator = "=" * 68

        lines = [
            separator,
            f"{self.config.organization} — RAPPORT D'INCIDENT",
            separator,
            f"Référence    : {event.event_id}",
            f"Site         : {self.config.site_name}",
            f"Opérateur    : {self.config.operator}",
            f"Date         : {context['date']} à {context['heure']}",
            f"Type         : {event.event_type.value}",
            f"Gravité      : {event.severity.value}",
            "",
            "CONSTAT",
            "-" * 68,
            self.generate(event),
            "",
            "ÉLÉMENTS TECHNIQUES",
            "-" * 68,
            f"Objet suivi        : #{event.track_id} ({event.class_name})",
            f"Zone               : {context['zone_name']}",
            f"Durée constatée    : {event.duration_s:.1f} s",
            f"Confiance          : {event.confidence:.1%}",
            f"Temps vidéo        : {event.video_time:.2f} s",
            f"Boîte englobante   : {tuple(round(v) for v in event.bbox)}",
            f"Preuve visuelle    : {event.evidence_path or 'non disponible'}",
        ]

        for key, value in event.details.items():
            lines.append(f"{key.replace('_', ' ').capitalize():19}: {value}")

        lines += [
            "",
            separator,
            "Document généré automatiquement par analyse vidéo. Il constitue un",
            "BROUILLON : sa validation par un opérateur humain est obligatoire",
            "avant toute exploitation ou transmission.",
            separator,
        ]
        return "\n".join(lines)

    def export_pdf(self, event: Event, output_path: Path | None = None) -> Path:
        """Exporte un incident en PDF via fpdf2.

        Args:
            event: Incident à exporter.
            output_path: Destination. `None` = `config.REPORT.directory` avec un
                nom dérivé de `event.event_id`.

        Returns:
            Le chemin du fichier écrit.

        Raises:
            ReportGenerationError: Si l'écriture échoue.
        """
        destination = output_path or self.config.directory / f"{event.event_id}.pdf"

        try:
            from fpdf import FPDF
        except ImportError as exc:
            raise ReportGenerationError(
                "Le paquet 'fpdf2' est introuvable. Installez les dépendances : "
                "pip install -r requirements.txt"
            ) from exc

        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            self._pdf_header(pdf, f"RAPPORT D'INCIDENT — {event.event_id}")
            self._pdf_event_section(pdf, event)
            self._pdf_evidence(pdf, event)

            destination.parent.mkdir(parents=True, exist_ok=True)
            pdf.output(str(destination))
        except Exception as exc:
            raise ReportGenerationError(
                f"Écriture du PDF impossible ({destination}) : {exc}"
            ) from exc

        logger.info("Rapport PDF écrit : %s", destination)
        return destination

    def export_batch_pdf(self, events: Sequence[Event], output_path: Path | None = None) -> Path:
        """Exporte plusieurs incidents dans un rapport de synthèse unique.

        Args:
            events: Incidents à inclure, dans l'ordre chronologique.
            output_path: Destination du PDF.

        Returns:
            Le chemin du fichier écrit.

        Raises:
            ReportGenerationError: Si la liste est vide ou si l'écriture échoue.
        """
        if not events:
            raise ReportGenerationError("Aucun incident à exporter.")

        destination = output_path or self.config.directory / "synthese-incidents.pdf"

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            self._pdf_header(pdf, "SYNTHÈSE DES INCIDENTS")

            by_type: dict[str, int] = {}
            by_severity: dict[str, int] = {}
            for event in events:
                by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
                by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1

            pdf.set_font("Helvetica", "", 11)
            _write(pdf, 6, f"Total : {len(events)} incident(s)")
            pdf.ln(2)
            for label, counts in (("Par type", by_type), ("Par gravité", by_severity)):
                pdf.set_font("Helvetica", "B", 11)
                _write(pdf, 6, label)
                pdf.set_font("Helvetica", "", 11)
                for key, count in sorted(counts.items()):
                    _write(pdf, 6, f"    {key} : {count}")
                pdf.ln(2)

            for event in events:
                pdf.add_page()
                self._pdf_event_section(pdf, event)
                self._pdf_evidence(pdf, event)

            destination.parent.mkdir(parents=True, exist_ok=True)
            pdf.output(str(destination))
        except ReportGenerationError:
            raise
        except Exception as exc:
            raise ReportGenerationError(
                f"Écriture de la synthèse impossible ({destination}) : {exc}"
            ) from exc

        logger.info("Synthèse PDF écrite : %s (%d incidents)", destination, len(events))
        return destination

    # -- Briques PDF ----------------------------------------------------------

    def _pdf_header(self, pdf, title: str) -> None:
        """Écrit l'en-tête commun à tous les documents.

        Args:
            pdf: Instance `FPDF` en cours.
            title: Titre du document.
        """
        # Version « fond clair » : encre anthracite sur fond transparent, la
        # seule qui convienne à une page A4 blanche. La version fond sombre y
        # collerait un pavé anthracite en haut du document.
        logo = config.BRAND.logo_on_light
        if logo.is_file():
            try:
                # fpdf2 vectorise les SVG (module `fpdf.svg`) : le logo reste net
                # à l'impression, sans passer par une image matricielle.
                pdf.image(str(logo), x=10, y=8, w=34)
                pdf.set_y(32)
            except Exception:
                # Un logo illisible ne doit jamais empêcher la production d'un
                # rapport : on continue sans lui.
                logger.warning("Logo absent du PDF (illisible) : %s", logo, exc_info=True)

        pdf.set_font("Helvetica", "B", 15)
        _write(pdf, 9, f"{self.config.organization}")
        pdf.set_font("Helvetica", "B", 12)
        _write(pdf, 7, title)
        pdf.set_font("Helvetica", "", 10)
        _write(pdf, 6, f"Site : {self.config.site_name}")
        _write(pdf, 6, f"Opérateur : {self.config.operator}")
        pdf.ln(3)

    def _pdf_event_section(self, pdf, event: Event) -> None:
        """Écrit le bloc décrivant un incident.

        Args:
            pdf: Instance `FPDF` en cours.
            event: Incident à décrire.
        """
        pdf.set_font("Helvetica", "B", 12)
        _write(pdf, 7, f"{event.event_id} — {event.event_type.value}")
        pdf.set_font("Helvetica", "", 11)
        _write(pdf, 6, self.generate(event))
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 9)
        facts = [
            f"Gravité : {event.severity.value}",
            f"Objet suivi : #{event.track_id} ({event.class_name})",
            f"Zone : {event.zone_name or 'hors zone'}",
            f"Durée : {event.duration_s:.1f} s",
            f"Confiance : {event.confidence:.1%}",
            f"Temps vidéo : {event.video_time:.2f} s",
        ]
        for fact in facts:
            _write(pdf, 5, fact)
        pdf.ln(2)

        pdf.set_font("Helvetica", "I", 8)
        _write(pdf, 5, "Document généré automatiquement. BROUILLON : validation par un "
                "opérateur humain obligatoire avant exploitation.")
        pdf.ln(2)

    def _pdf_evidence(self, pdf, event: Event) -> None:
        """Insère la capture justificative si elle existe.

        Args:
            pdf: Instance `FPDF` en cours.
            event: Incident concerné.
        """
        if not self.config.include_evidence_image or event.evidence_path is None:
            return
        if not Path(event.evidence_path).is_file():
            logger.warning("Preuve introuvable, absente du PDF : %s", event.evidence_path)
            return

        try:
            pdf.set_font("Helvetica", "B", 10)
            _write(pdf, 6, "Preuve visuelle")
            pdf.image(str(event.evidence_path), w=170)
        except Exception:
            # Une image illisible ne doit pas annuler un rapport dont le texte
            # est valide.
            logger.exception("Insertion de la preuve impossible : %s", event.evidence_path)

    # -- Autres sorties -------------------------------------------------------

    def generate_with_llm(self, event: Event) -> str:
        """Reformule le rapport avec un LLM — POINT D'EXTENSION.

        Contrat impératif : le LLM ne reçoit **que** les faits déjà établis
        (type, horodatage, zone, durée, classe, confiance) et ne doit rien
        inventer. En cas d'erreur — clé absente, réseau indisponible, délai
        dépassé — la méthode retombe silencieusement sur `generate()` : la
        rédaction assistée est un confort, jamais une dépendance.

        Args:
            event: Incident à décrire.

        Returns:
            Le texte reformulé, ou le texte du gabarit en cas d'échec.
        """
        if not config.LLM.enabled or not config.LLM.api_key:
            return self.generate(event)

        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=config.LLM.api_key, timeout=config.LLM.timeout_s
            )
            facts = "\n".join(f"{key} : {value}" for key, value in event.to_dict().items())
            response = client.messages.create(
                model=config.LLM.model,
                max_tokens=config.LLM.max_tokens,
                system=config.LLM.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Rédige le constat de ce rapport d'incident à partir "
                            "des seuls faits suivants :\n\n" + facts
                        ),
                    }
                ],
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            return text.strip() or self.generate(event)
        except Exception:
            logger.exception("Rédaction LLM indisponible ; repli sur le gabarit.")
            return self.generate(event)

    def export_csv(self, events: Sequence[Event], output_path: Path | None = None) -> Path:
        """Exporte le journal des incidents en CSV, pour analyse externe.

        Args:
            events: Incidents à exporter.
            output_path: Destination du fichier.

        Returns:
            Le chemin du fichier écrit.

        Raises:
            ReportGenerationError: Si la liste est vide ou si l'écriture échoue.
        """
        if not events:
            raise ReportGenerationError("Aucun incident à exporter.")

        destination = output_path or self.config.directory / "incidents.csv"
        rows = [event.to_dict() for event in events]

        # Les incidents n'ont pas tous les mêmes clés `detail_*` : l'en-tête est
        # l'union de toutes les colonnes rencontrées, sinon DictWriter lèverait
        # sur le premier incident portant un détail supplémentaire.
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # `utf-8-sig` : sans le BOM, Excel affiche « intrusion_zone_restreinte »
            # avec des accents cassés à l'ouverture d'un CSV UTF-8.
            with destination.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, restval="")
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            raise ReportGenerationError(
                f"Écriture du CSV impossible ({destination}) : {exc}"
            ) from exc

        logger.info("Journal CSV écrit : %s (%d incidents)", destination, len(rows))
        return destination
