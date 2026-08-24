"""Interface Streamlit de SentinelReport — ÉTAPE 7.

Rôle : orchestrer les modules et présenter les résultats. **Aucune logique
métier ici** — pas de calcul de durée, pas de règle d'événement, pas de seuil en
dur. Ce fichier lit la configuration, appelle le pipeline et affiche.

C'est le test de l'architecture : si une règle métier devait être écrite dans
`app.py`, c'est qu'elle est mal placée ailleurs.

Lancement :
    streamlit run app.py
"""

from __future__ import annotations

import base64
import inspect
import logging
import tempfile
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Final, NamedTuple

import cv2
import streamlit as st

import config
from sentinel.detector import Detector
from sentinel.events import Event, EventEngine
from sentinel.exceptions import ModelLoadError, SentinelError, SourceDisconnectedError
from sentinel.crossing import LineCounter
from sentinel.evidence import EvidenceWriter
from sentinel.report import ReportGenerator
from sentinel.session_report import SessionReportGenerator
from sentinel.source import VideoSource, detect_kind
from sentinel.timeline import SessionContext, Timeline
from sentinel.tracker import Tracker
from sentinel.zones import ZoneManager

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("sentinel.app")


# ---------------------------------------------------------------------------
# Compatibilité entre versions de Streamlit
# ---------------------------------------------------------------------------

# Streamlit a changé de convention pour « occuper toute la largeur » :
#
#   * jusqu'à la 1.4x  ->  use_container_width=True
#   * à partir de la 1.49 -> width="stretch", et use_container_width est déprécié
#
# Les deux API coexistent mal : passer `width="stretch"` à une version 1.44 lève
# `TypeError: '<=' not supported between instances of 'str' and 'int'`, et
# `use_container_width` inonde les versions récentes d'avertissements.
#
# `st.button` sert de sonde fiable : ce widget n'a JAMAIS accepté de largeur
# numérique, donc la présence du paramètre `width` signale à coup sûr la nouvelle
# convention. On teste une fois, à l'import, plutôt qu'à chaque appel.
_WIDTH_TOKEN_SUPPORTED: Final[bool] = "width" in inspect.signature(st.button).parameters


def stretch() -> dict[str, object]:
    """Arguments demandant à un widget d'occuper toute la largeur disponible.

    À dérouler avec `**stretch()` sur `st.button`, `st.image` et `st.dataframe`.

    Returns:
        Le paramètre adapté à la version de Streamlit en cours d'exécution.
    """
    if _WIDTH_TOKEN_SUPPORTED:
        return {"width": "stretch"}
    return {"use_container_width": True}


# ---------------------------------------------------------------------------
# Identité visuelle
# ---------------------------------------------------------------------------


def configure_page() -> None:
    """Configure l'onglet du navigateur. **À appeler avant tout autre appel `st`.**

    `st.set_page_config()` doit être la première commande Streamlit exécutée par
    le script : Streamlit lève une exception si un élément a déjà été rendu. Elle
    est donc appelée en tête de `main()`, et nulle part ailleurs.

    Le favicon est un fichier local passé par son chemin. Streamlit lit le SVG,
    l'encode en `data:image/svg+xml;base64,...` et l'injecte dans le `<link
    rel="icon">` de la page — aucun fichier n'est servi en tant qu'URL, ce qui
    évite tout problème de chemin relatif côté navigateur.

    Si le fichier est absent, on retombe sur l'emoji de `config.UI.page_icon`.
    Ce repli est explicite parce que Streamlit, lui, **échoue en silence** :
    `set_page_config` attrape l'erreur de lecture et transmet la chaîne telle
    quelle au navigateur, qui affiche un onglet sans icône sans rien signaler.
    """
    favicon = config.BRAND.favicon
    page_icon: str = str(favicon) if favicon.is_file() else config.UI.page_icon

    st.set_page_config(
        page_title=config.UI.page_title,
        page_icon=page_icon,
        layout=config.UI.layout,
    )


def inject_styles() -> None:
    """Ajoute les quelques règles CSS que le thème ne couvre pas.

    Le gros du travail est fait par `.streamlit/config.toml` — palette, fonds,
    couleur d'accent — parce que c'est l'API officielle et qu'elle atteint aussi
    les composants internes. Ce complément se limite à ce qui n'est pas
    paramétrable : l'écrin du logo, les intitulés de section et les pastilles de
    classes. Il est **purement cosmétique** : si un sélecteur venait à changer
    dans une version future de Streamlit, la mise en page resterait fonctionnelle.
    """
    st.markdown(
        f"""
        <style>
          /* Écrin du logo : dégradé ardoise très clair, qui met en valeur le
             pavé anthracite du fichier SVG posé dessus. */
          .sr-logo-card {{
              background: linear-gradient(135deg, #FFFFFF 0%, #E3EAF4 100%);
              border: 1px solid #D3DCE8;
              border-radius: 16px;
              padding: 18px 20px 10px 20px;
              box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
          }}
          .sr-title {{
              color: {config.BRAND.anthracite};
              font-size: 1.55rem;
              font-weight: 800;
              letter-spacing: -0.015em;
              margin: 0 0 4px 0;
          }}
          .sr-subtitle {{ color: #64748B; font-size: 0.92rem; margin: 0; }}

          /* Écran d'initialisation, affiché le temps que le modèle se charge
             et que la cadence temps réel s'établisse. Une attente sans signe de
             vie se lit comme une panne : l'anneau tourne, la barre défile, et le
             compte à rebours dit combien de temps il reste. */
          .sr-boot {{
              display: flex;
              flex-direction: column;
              align-items: center;
              justify-content: center;
              gap: 14px;
              min-height: 260px;
              padding: 32px 24px;
              border-radius: 14px;
              border: 1px dashed #C7D3E3;
              background: linear-gradient(135deg, #FFFFFF 0%, #EEF3F9 100%);
          }}
          .sr-boot-ring {{
              width: 46px;
              height: 46px;
              border-radius: 50%;
              border: 3px solid #D8E1EC;
              border-top-color: {config.BRAND.steel_blue};
              animation: sr-spin 0.9s linear infinite;
          }}
          @keyframes sr-spin {{ to {{ transform: rotate(360deg); }} }}

          .sr-boot-text {{
              color: {config.BRAND.anthracite};
              font-size: 1.02rem;
              font-weight: 800;
              letter-spacing: 0.20em;
              margin: 0;
          }}
          /* Les points sont animés en CSS plutôt que réécrits depuis Python :
             réécrire le bloc dix fois par seconde ferait clignoter la page et
             multiplierait les manipulations du DOM pendant la phase la plus
             fragile de la session. */
          .sr-boot-text::after {{
              content: "";
              animation: sr-dots 1.5s steps(1, end) infinite;
          }}
          @keyframes sr-dots {{
              0%   {{ content: ""; }}
              20%  {{ content: "."; }}
              40%  {{ content: ".."; }}
              60%  {{ content: "..."; }}
              80%  {{ content: "...."; }}
              100% {{ content: "....."; }}
          }}

          .sr-boot-track {{
              width: min(320px, 70%);
              height: 4px;
              border-radius: 2px;
              background: #DDE5EF;
              overflow: hidden;
          }}
          .sr-boot-fill {{
              width: 38%;
              height: 100%;
              border-radius: 2px;
              background: {config.BRAND.steel_blue};
              animation: sr-slide 1.25s ease-in-out infinite;
          }}
          /* Barre indéterminée : on ne connaît pas la durée réelle du
             chargement, et une barre de progression qui mentirait sur son
             avancement serait pire que pas de barre du tout. */
          @keyframes sr-slide {{
              0%   {{ transform: translateX(-100%); }}
              100% {{ transform: translateX(320%); }}
          }}

          .sr-boot-hint {{ color: #64748B; font-size: 0.86rem; margin: 0; text-align: center; }}

          /* Intitulés de section, dans la page comme dans la barre latérale. */
          .sr-section {{
              text-transform: uppercase;
              letter-spacing: 0.09em;
              font-size: 0.72rem;
              font-weight: 700;
              color: #64748B;
              margin: 2px 0 6px 0;
          }}

          /* Pastilles listant les classes réellement transmises au modèle. */
          .sr-chip {{
              display: inline-block;
              background: #FFFFFF;
              border: 1px solid #CBD5E1;
              border-radius: 999px;
              padding: 2px 11px;
              margin: 2px 4px 2px 0;
              font-size: 0.78rem;
              color: #334155;
          }}

          /* Barre latérale : réglages plus compacts et mieux séparés. */
          [data-testid="stSidebar"] hr {{ margin: 0.6rem 0; }}
          [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}

          /* Bandeau d'état : les chiffres priment sur leurs intitulés. */
          [data-testid="stMetric"] {{
              background: #FFFFFF;
              border: 1px solid #DCE3ED;
              border-radius: 12px;
              padding: 10px 14px;
          }}
          [data-testid="stMetricLabel"] p {{
              font-size: 0.74rem !important;
              text-transform: uppercase;
              letter-spacing: 0.07em;
              color: #64748B;
          }}
          [data-testid="stMetricValue"] {{
              font-size: 1.5rem;
              color: {config.BRAND.anthracite};
          }}

          /* Cartes : même arrondi et même trait que l'écrin du logo. */
          [data-testid="stExpander"] details,
          [data-testid="stVerticalBlockBorderWrapper"] {{
              border-radius: 14px;
              border-color: #DCE3ED;
          }}

          /* Tableau des incidents : coins arrondis comme les autres cartes. */
          [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _logo_card_html(path: Path) -> str:
    """Construit l'écrin du logo en **un seul bloc HTML autonome**.

    Point crucial de robustesse : chaque appel à `st.markdown` doit contenir du
    HTML **équilibré**. Ouvrir un `<div>` dans un appel et le fermer dans un
    autre paraît fonctionner, mais le navigateur referme automatiquement le
    premier bloc et laisse la balise fermante du second orpheline. Le DOM réel
    diverge alors de celui que React croit gérer, et le premier réaffichage
    échoue sur `NotFoundError: Failed to execute 'removeChild' on 'Node'` — une
    erreur qui survient loin de sa cause, typiquement pendant la boucle vidéo qui
    rafraîchit ses emplacements des dizaines de fois par seconde.

    Le SVG est donc encodé en `data:` et inséré dans le même bloc que son écrin.
    Bénéfice secondaire : la largeur et la hauteur sont fixées explicitement sur
    la balise `<img>`, ce qui règle l'absence de dimension intrinsèque des SVG
    fournis (`width="100%"`), sans modifier les fichiers.

    Args:
        path: Chemin du logo SVG.

    Returns:
        Le fragment HTML complet de la carte.
    """
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    width = config.BRAND.header_width_px
    return (
        '<div class="sr-logo-card">'
        f'<img src="data:image/svg+xml;base64,{encoded}" '
        f'width="{width}" style="width:{width}px;height:auto;display:block;" '
        f'alt="{config.REPORT.organization}">'
        "</div>"
    )


def _boot_screen_html(title: str = "INITIALISATION", hint: str = "") -> str:
    """Écran de chargement, en **un seul bloc HTML autonome**.

    Même exigence que pour l'écrin du logo : ouvrir une balise dans un appel à
    `st.markdown` et la fermer dans un autre laisse le DOM réel diverger de celui
    que React croit gérer, et le premier réaffichage échoue sur
    `NotFoundError: removeChild`. Le risque est ici maximal, puisque ce bloc est
    précisément remplacé par la première image de la vidéo.

    Args:
        title: Texte principal, sans les points de suspension — ils sont animés
            en CSS.
        hint: Ligne d'explication sous la barre.

    Returns:
        Le fragment HTML complet.
    """
    ligne = f'<p class="sr-boot-hint">{escape(hint)}</p>' if hint else ""
    return (
        '<div class="sr-boot">'
        '<div class="sr-boot-ring"></div>'
        f'<p class="sr-boot-text">{escape(title)}</p>'
        '<div class="sr-boot-track"><div class="sr-boot-fill"></div></div>'
        f"{ligne}"
        "</div>"
    )


def render_header() -> None:
    """Affiche l'en-tête de l'application avec le logo de marque.

    Le logo retenu est celui destiné aux **fonds sombres** : le fichier embarque
    son propre pavé anthracite, qui ressort sur le fond bleuté très clair de la
    zone principale. Il reste donc lisible quel que soit le thème du navigateur.
    """
    logo = config.BRAND.logo_for(dark_background=True)

    logo_column, title_column = st.columns([2, 3], vertical_alignment="center")

    with logo_column:
        if logo.is_file():
            st.markdown(_logo_card_html(logo), unsafe_allow_html=True)
        else:
            # Repli textuel : l'application reste parfaitement utilisable sans
            # ses fichiers de marque.
            st.title(config.UI.page_title)
            st.caption("Logo introuvable — l'application fonctionne normalement.")

    with title_column:
        st.markdown(
            '<p class="sr-title">Détection d\'incidents de sécurité</p>'
            '<p class="sr-subtitle">Analyse vidéo · suivi multi-objets · '
            "rapports horodatés avec preuve visuelle</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{config.REPORT.organization} · {config.REPORT.site_name} · "
            "les rapports produits sont des brouillons à valider par un opérateur."
        )

    _warn_about_missing_assets()


def _warn_about_missing_assets() -> None:
    """Signale une seule fois les fichiers de marque manquants.

    Le message est volontairement placé dans la barre latérale et non dans le
    corps de la page : un fichier décoratif absent est un défaut d'installation,
    pas un incident de sécurité — il ne doit pas concurrencer visuellement les
    alertes du tableau de bord.
    """
    missing = config.BRAND.missing()
    if not missing:
        return

    details = "\n".join(f"- {label} : `{path}`" for label, path in missing.items())
    st.sidebar.warning(
        "Fichiers d'identité visuelle introuvables :\n\n"
        f"{details}\n\n"
        f"Placez-les dans `{config.BRAND.directory}` pour rétablir l'affichage."
    )


# Note : le logo destiné aux supports clairs n'a pas d'accesseur ici. Il est lu
# directement depuis `config.BRAND.logo_on_light` par `report.py`, qui l'insère
# en tête des PDF — un intermédiaire dans `app.py` n'aurait fait qu'ajouter une
# indirection entre la configuration et son unique utilisateur.


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)  # l'écran d'initialisation s'en charge
def load_detector(weights: str) -> Detector:
    """Charge le détecteur correspondant à un fichier de poids **local**.

    `@st.cache_resource` mémorise le résultat en fonction des arguments : tant
    que `weights` ne change pas, les rerun de Streamlit (chaque clic, chaque
    curseur déplacé) réutilisent la même instance. Dès que l'utilisateur choisit
    un autre modèle, l'argument change, la clé de cache change, et une nouvelle
    instance est construite — les anciennes restant en cache, revenir au modèle
    précédent est instantané.

    Args:
        weights: Chemin du fichier `.pt` déjà présent sur le disque. C'est une
            `str` et non un `Path` pour que la clé de cache soit stable et
            lisible.

    Returns:
        Le détecteur prêt à l'emploi.

    Raises:
        ModelLoadError: Si le fichier de poids est absent (script de
            téléchargement pas encore lancé) ou illisible.
    """
    if not Path(weights).is_file():
        raise ModelLoadError(
            f"Fichier de poids introuvable : {weights}. "
            "Lancez d'abord : python download_models.py"
        )
    return Detector(weights=weights)


class Pipeline(NamedTuple):
    """Les composants d'une analyse, assemblés pour une session.

    Un tuple nommé plutôt qu'un tuple anonyme : `pipeline[4]` ne dit pas ce
    qu'il désigne, et l'ordre finit toujours par changer.

    Attributes:
        detector: Perception — frame vers détections.
        tracker: Mémoire temporelle des objets suivis.
        zones: Géométrie — appartenance aux polygones.
        events: Moteur de règles.
        generator: Générateur de rapports, d'incident **et** de session.
        timeline: Chronologie alimentée au fil de l'analyse.
        lines: Compteur de franchissements. Inerte tant que
            `config.CROSSING_LINES` est vide, ce qui est le cas par défaut.
    """

    detector: Detector
    tracker: Tracker
    zones: ZoneManager
    events: EventEngine
    generator: SessionReportGenerator
    timeline: Timeline
    lines: LineCounter


def build_pipeline(weights: str, settings: dict[str, object]) -> Pipeline:
    """Assemble les composants du pipeline pour une analyse.

    **Volontairement non mis en cache**, contrairement au détecteur. `Tracker`,
    `ZoneManager`, `EventEngine` et `Timeline` portent l'état d'une analyse :
    identifiants attribués, chronomètres de zone, historique d'incidents, faits
    marquants. Les conserver d'un rerun à l'autre ferait démarrer une nouvelle
    vidéo avec la mémoire de la précédente. Leur construction est de toute façon
    gratuite — seuls les 6 à 50 Mo de poids YOLO méritaient un cache, et
    `load_detector` s'en charge.

    Le générateur est un `SessionReportGenerator` : il hérite de
    `ReportGenerator`, donc les rapports d'incident isolés sont inchangés, et il
    sait en plus assembler le rapport de session à deux niveaux.

    Args:
        weights: Chemin du modèle sélectionné.
        settings: Réglages issus de `render_sidebar()`.

    Returns:
        Les composants prêts à l'emploi.

    Raises:
        SentinelError: Si un composant ne peut pas être initialisé.
    """
    detector = load_detector(weights)
    detector.confidence = float(settings["confidence"])
    detector.set_tracked_classes(list(settings["classes"]))
    detector.reset()

    tracker = Tracker(detector)
    zone_manager = ZoneManager()
    event_engine = EventEngine(zone_manager, rules=scaled_rules(settings))

    return Pipeline(
        detector=detector,
        tracker=tracker,
        zones=zone_manager,
        events=event_engine,
        generator=SessionReportGenerator(),
        timeline=Timeline(),
        lines=LineCounter(),
    )


def session_context(source: VideoSource, settings: dict[str, object]) -> SessionContext:
    """Décrit la session pour l'en-tête du rapport.

    Un rapport qui ne dit pas avec quels réglages il a été produit n'est pas
    vérifiable : le même quai filmé à 25 % de seuil de confiance et à 75 % ne
    donne pas les mêmes incidents.

    Args:
        source: Source analysée.
        settings: Réglages de la session.

    Returns:
        Le contexte, **sans identifiant de connexion** — `VideoSource.label`
        masque le mot de passe des URL RTSP, et un rapport se transmet.
    """
    modele = next(
        (
            label
            for label, chemin in config.AVAILABLE_MODELS.items()
            if str(chemin) == str(settings.get("weights"))
        ),
        "-",
    )
    return SessionContext(
        source_label=source.label,
        is_live=source.is_live,
        model_label=modele,
        confidence=float(settings["confidence"]),
        watched_classes=tuple(settings.get("classes") or ()),
        duration_factor=float(settings.get("min_duration_factor", 1.0)),
    )


def scaled_rules(settings: dict[str, object]) -> tuple[config.EventRule, ...]:
    """Règles ajustées au curseur de durées de la barre latérale.

    Simple adaptateur : il lit le réglage de l'interface et délègue le calcul à
    `config.scaled_rules()`. La règle — multiplier seuil **et** délai de garde
    ensemble pour préserver l'invariant `cooldown_s > min_duration_s` — est une
    décision métier ; elle vit dans la configuration, où un script batch peut
    l'appliquer sans importer Streamlit.

    Args:
        settings: Réglages issus de la barre latérale.

    Returns:
        Les règles ajustées, laissant `config.EVENT_RULES` intact.
    """
    return config.scaled_rules(float(settings.get("min_duration_factor", 1.0)))


# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------


def render_model_selector() -> str | None:
    """Affiche le menu déroulant de choix du modèle dans la barre latérale.

    Le menu liste les noms lisibles de `config.AVAILABLE_MODELS`. Les modèles
    absents du disque restent visibles mais signalés dans la liste : masquer
    une option laisserait l'utilisateur croire qu'elle n'existe pas, alors qu'il
    lui manque simplement une étape d'installation.

    Returns:
        Le chemin du modèle choisi, ou `None` si aucun modèle n'est disponible
        en local (un message d'installation est alors affiché).
    """
    st.sidebar.header("Modèle de détection")

    available = {
        label: path for label, path in config.AVAILABLE_MODELS.items() if path.is_file()
    }
    missing = config.missing_models()

    if not available:
        st.sidebar.error(
            "Aucun modèle trouvé dans `models/`.\n\n"
            "Lancez d'abord, une seule fois :\n\n"
            "```\npython download_models.py\n```"
        )
        return None

    labels = list(config.AVAILABLE_MODELS)
    default_label = (
        config.DEFAULT_MODEL_LABEL
        if config.DEFAULT_MODEL_LABEL in available
        else next(iter(available))
    )

    choice = st.sidebar.selectbox(
        "Modèle",
        options=labels,
        index=labels.index(default_label),
        format_func=lambda label: label if label in available else f"{label} (non téléchargé)",
        help="Les modèles sont chargés depuis `models/`, jamais téléchargés en cours d'exécution.",
    )

    if choice not in available:
        st.sidebar.warning(
            f"Le modèle « {choice} » n'est pas présent en local "
            f"(`{config.AVAILABLE_MODELS[choice]}`).\n\n"
            "Lancez `python download_models.py` puis rechargez la page. "
            f"Modèle utilisé en attendant : « {default_label} »."
        )
        choice = default_label

    st.sidebar.caption(config.MODEL_DESCRIPTIONS.get(choice, ""))
    if missing:
        st.sidebar.caption(
            f"{len(missing)} modèle(s) non téléchargé(s) : {', '.join(missing)}."
        )

    return str(config.AVAILABLE_MODELS[choice])


def render_sidebar() -> dict[str, object]:
    """Barre latérale : **uniquement les régulateurs**.

    Séparation voulue : à gauche ce qui se règle par curseur — sensibilité,
    cadence, affichage ; à droite ce qui se choisit — la source et les objets à
    surveiller. Un opérateur ajuste les premiers en cours d'analyse et ne touche
    aux seconds qu'au moment de préparer une session.

    Returns:
        Les réglages d'analyse (le modèle, les seuils, la cadence, l'affichage).
    """
    settings: dict[str, object] = {"weights": render_model_selector()}

    st.sidebar.divider()
    st.sidebar.markdown('<p class="sr-section">Sensibilité</p>', unsafe_allow_html=True)
    settings["confidence"] = st.sidebar.slider(
        "Seuil de confiance",
        min_value=0.05,
        max_value=0.95,
        value=float(config.MODEL.confidence),
        step=0.05,
        help="Plus bas = plus de détections, mais davantage de faux positifs.",
    )
    settings["min_duration_factor"] = st.sidebar.slider(
        "Durées de déclenchement",
        min_value=0.25,
        max_value=3.0,
        value=1.0,
        step=0.25,
        format="× %.2f",
        help=(
            "Multiplie les seuils de `config.EVENT_RULES`. Sous 1, les incidents "
            "se déclenchent plus tôt ; au-dessus, il faut stationner plus longtemps."
        ),
    )

    st.sidebar.divider()
    st.sidebar.markdown('<p class="sr-section">Cadence</p>', unsafe_allow_html=True)
    settings["realtime"] = st.sidebar.toggle(
        "Analyse en temps réel",
        value=config.REALTIME.enabled,
        help=(
            "Tient le rythme de la vidéo en écartant les images que la machine "
            "n'a pas le temps de traiter. Les **durées restent exactes** ; c'est "
            "l'exhaustivité de l'observation qui est échangée contre la cadence. "
            "Désactiver pour une analyse image par image, reproductible, mais "
            "trois à quatre fois plus longue que la vidéo."
        ),
    )
    settings["frame_stride"] = st.sidebar.slider(
        "Analyser une frame sur",
        min_value=1,
        max_value=10,
        value=int(config.VIDEO.frame_stride),
        help=(
            "Sous-échantillonnage **fixe**, appliqué avant toute chose. À "
            "distinguer du temps réel, qui écarte des images selon la charge "
            "réelle de la machine."
        ),
    )
    settings["max_seconds"] = st.sidebar.slider(
        "Durée maximale analysée",
        min_value=10,
        max_value=600,
        value=120,
        step=10,
        format="%d s",
        help="Garde-fou : interrompt l'analyse au-delà de cette durée de vidéo.",
    )

    st.sidebar.divider()
    st.sidebar.markdown('<p class="sr-section">Affichage</p>', unsafe_allow_html=True)
    # Pas d'interrupteur « zones » : la surveillance est globale, le périmètre
    # couvre tout le champ et n'est donc jamais tracé (`ZoneConfig.draw`). Un
    # réglage sans effet visible serait plus déroutant qu'utile.
    settings["show_confidence"] = st.sidebar.toggle(
        "Score de confiance", value=config.UI.show_confidence
    )
    settings["show_track_id"] = st.sidebar.toggle(
        "Identifiants de suivi", value=config.UI.show_track_id
    )

    return settings


# ---------------------------------------------------------------------------
# Panneau principal : source et objets surveillés
# ---------------------------------------------------------------------------


def render_source_panel(settings: dict[str, object]) -> None:
    """Panneau de droite : source vidéo et objets à surveiller.

    Complète `settings` en place avec `source_kind`, `upload` et `classes`.

    Args:
        settings: Réglages déjà collectés dans la barre latérale.
    """
    source_column, classes_column = st.columns([2, 3], gap="medium")

    with source_column:
        with st.container(border=True):
            st.markdown(
                '<p class="sr-section">1 · Source vidéo</p>', unsafe_allow_html=True
            )
            source_kind = st.radio(
                "Type de source",
                options=("Fichier vidéo", "Webcam"),
                horizontal=True,
                label_visibility="collapsed",
            )
            settings["source_kind"] = source_kind

            if source_kind == "Fichier vidéo":
                upload = st.file_uploader(
                    "Déposez une vidéo",
                    type=[ext.lstrip(".") for ext in config.VIDEO.allowed_extensions],
                    label_visibility="collapsed",
                )
                settings["upload"] = upload
                if upload is not None:
                    size_mb = upload.size / 1_048_576
                    st.success(f"{upload.name} · {size_mb:.1f} Mo", icon="\N{MOVIE CAMERA}")
                else:
                    st.caption(
                        "Formats acceptés : "
                        f"{', '.join(config.VIDEO.allowed_extensions)} · "
                        f"{config.VIDEO.max_width} px de large maximum à l'analyse."
                    )
            else:
                settings["upload"] = None
                st.info(
                    f"Webcam n°{config.VIDEO.webcam_index}. Un flux en direct n'a "
                    "pas de fin : la durée maximale réglée à gauche s'applique.",
                    icon="\N{MOVIE CAMERA}",
                )

    with classes_column:
        with st.container(border=True):
            st.markdown(
                '<p class="sr-section">2 · Objets surveillés</p>', unsafe_allow_html=True
            )
            settings["classes"] = _render_class_picker()


def render_status_strip(events: list[Event]) -> None:
    """Synthèse chiffrée des incidents d'une session terminée.

    Placée au-dessus du tableau plutôt qu'en tête de page : avant toute analyse,
    quatre compteurs à zéro n'apprennent rien et occupent l'espace où devrait
    se trouver la seule action utile — préparer la session.

    Args:
        events: Incidents accumulés depuis le début de la session.
    """
    # L'ordre des gravités est une donnée métier : `config.worst_severity()`
    # en est l'unique dépositaire. Le réimplémenter ici par une liste ordonnée
    # en dur aurait créé un second classement, à mettre à jour deux fois le jour
    # où un niveau s'ajoute.
    pire_niveau = config.worst_severity([event.severity for event in events])
    pire = pire_niveau.value.capitalize() if pire_niveau else "—"

    critiques = sum(
        1
        for event in events
        if event.priority
        and event.priority.level
        in (config.PriorityLevel.HIGH, config.PriorityLevel.CRITICAL)
    )
    derniere = max((event.video_time for event in events), default=0.0)

    colonnes = st.columns(4)
    colonnes[0].metric("Incidents", len(events))
    colonnes[1].metric("Priorité élevée ou plus", critiques or "—")
    colonnes[2].metric("Gravité maximale", pire)
    colonnes[3].metric("Dernier incident", f"{derniere:.0f} s" if events else "—")


def render_action_bar(settings: dict[str, object]) -> bool:
    """Récapitulatif de la session et commandes de lancement.

    Le récapitulatif tient en **une phrase** et non en cartes de métriques : ces
    valeurs sont déjà réglées dans la barre latérale, les réafficher en gros
    caractères créait deux rangées de chiffres concurrentes sur le même écran.
    Ici on veut seulement une relecture avant d'engager un traitement long.

    Args:
        settings: Réglages complets de la session.

    Returns:
        True si l'utilisateur demande le lancement de l'analyse.
    """
    modele = next(
        (
            label
            for label, path in config.AVAILABLE_MODELS.items()
            if str(path) == str(settings.get("weights"))
        ),
        "—",
    )
    intrusion = next(
        rule
        for rule in scaled_rules(settings)
        if rule.event_type is config.EventType.INTRUSION
    )
    objets = len(settings.get("classes") or [])
    source = settings.get("source_kind", "—")
    if settings.get("upload") is not None:
        source = settings["upload"].name

    st.caption(
        f"**{modele.split(' (')[0]}** · {objets} objet(s) surveillé(s) · "
        f"confiance {float(settings['confidence']):.0%} · "
        f"intrusion signalée après {intrusion.min_duration_s:.0f} s · "
        f"source : {source}"
    )

    lancer_col, reset_col, _ = st.columns([1, 1, 2])
    with lancer_col:
        launch = st.button("Lancer l'analyse", type="primary", **stretch())
    with reset_col:
        if st.button("Réinitialiser", **stretch()):
            st.session_state["events"] = []
            st.rerun()

    return launch


def _render_class_picker() -> list[str]:
    """Sélecteur d'objets surveillés : groupes métier + ajouts à la carte.

    Returns:
        Les classes COCO à transmettre au détecteur.
    """
    labels = [group.label for group in config.CLASS_GROUPS]
    icons = {group.label: group.icon for group in config.CLASS_GROUPS}

    chosen = st.multiselect(
        "Catégories",
        options=labels,
        default=list(config.DEFAULT_CLASS_GROUPS),
        format_func=lambda label: f"{icons.get(label, '')} {label}".strip(),
        label_visibility="collapsed",
    )

    # Les limites du modèle sont affichées au moment où elles deviennent
    # pertinentes, pas noyées dans une documentation que personne n'ouvre.
    for group in config.CLASS_GROUPS:
        if group.label in chosen and group.note:
            st.warning(group.note, icon="\N{WARNING SIGN}")

    classes = config.classes_for_groups(chosen)
    extra: list[str] = st.session_state.setdefault("custom_classes", [])

    # Volontairement sans `st.expander` : ce panneau est lui-même appelé depuis
    # un expander, et Streamlit interdit de les imbriquer
    # (« Expanders may not be nested inside other expanders »). Une ligne
    # toujours visible est de toute façon plus découvrable qu'un repli.
    remaining = [name for name in config.COCO_CLASSES if name not in classes + extra]
    picker_column, add_column = st.columns([3, 1], vertical_alignment="bottom")
    with picker_column:
        candidate = st.selectbox(
            "Ajouter un objet précis",
            options=remaining,
            index=None,
            placeholder="Ajouter un objet parmi les 80 classes du modèle...",
            label_visibility="collapsed",
        )
    with add_column:
        if st.button("Ajouter", **stretch()) and candidate:
            extra.append(candidate)
            st.rerun()

    # La liste est fermée à dessein : seul le modèle décide de ce qu'il sait
    # reconnaître. Une saisie libre laisserait espérer une classe « pistolet »
    # qui ne se déclencherait jamais.
    if extra and st.button("Retirer les ajouts", type="tertiary"):
        st.session_state["custom_classes"] = []
        st.rerun()

    effective = classes + [name for name in extra if name not in classes]
    if effective:
        # `escape` par principe : les noms viennent aujourd'hui de COCO et sont
        # alphanumériques, mais du HTML injecté sans échappement est une
        # habitude qui finit toujours par coûter cher.
        st.markdown(
            "".join(
                f'<span class="sr-chip">{escape(name)}</span>' for name in effective
            ),
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "Aucune catégorie sélectionnée : toutes les classes du modèle seront "
            "détectées, ce qui multiplie les faux positifs.",
            icon="\N{WARNING SIGN}",
        )

    return effective


# ---------------------------------------------------------------------------
# Traitement vidéo
# ---------------------------------------------------------------------------


def resolve_source(settings: dict[str, object]) -> str | int | None:
    """Désigne la source choisie par l'opérateur.

    Rend une **désignation**, pas une source ouverte : la nature du flux et sa
    gestion appartiennent à `sentinel.source`, pas à l'interface.

    Un fichier téléversé vit en mémoire ; toute capture vidéo exige un chemin sur
    disque. On l'écrit donc dans un fichier temporaire, conservé dans la session
    pour ne pas le réécrire à chaque rerun.

    Args:
        settings: Réglages issus de la barre latérale.

    Returns:
        Un chemin de fichier, un index de webcam, ou `None` si rien n'est prêt.
    """
    if settings["source_kind"] == "Webcam":
        return int(config.VIDEO.webcam_index)

    upload = settings.get("upload")
    if upload is None:
        return None

    cached = st.session_state.get("video_path")
    if cached and st.session_state.get("video_name") == upload.name:
        return cached

    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(upload.getbuffer())
        path = handle.name

    st.session_state["video_path"] = path
    st.session_state["video_name"] = upload.name
    return path


def annotate(frame, zone_manager: ZoneManager, objects, settings: dict[str, object]):
    """Dessine zones et boîtes sur une copie de la frame.

    Args:
        frame: Image BGR d'origine.
        zone_manager: Gestionnaire de zones déjà initialisé.
        objects: Objets suivis à représenter.
        settings: Réglages d'affichage.

    Returns:
        Une nouvelle image annotée.
    """
    # Les options d'affichage retombent sur `config.UI` quand elles sont absentes :
    # une case à cocher non renseignée ne doit pas interrompre l'analyse en cours.
    show_confidence = bool(settings.get("show_confidence", config.UI.show_confidence))
    show_track_id = bool(settings.get("show_track_id", config.UI.show_track_id))

    # `draw()` ne trace que les zones dont `ZoneConfig.draw` vaut True. En
    # surveillance globale, aucune ne l'est : l'image reste nette, seules les
    # boîtes de détection s'y ajoutent.
    canvas = zone_manager.draw(frame)

    for obj in objects:
        x1, y1, x2, y2 = obj.detection.as_int_box()
        # La couleur d'alerte est celle de la zone en infraction, pas un rouge
        # codé ici : deux zones de gravités différentes restent distinguables
        # sur la vidéo. Hors infraction, la couleur neutre vient de `config.UI`.
        color = zone_manager.alert_color_for(obj.zones) or config.UI.box_color

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, config.UI.box_thickness)
        label = obj.detection.label(
            show_confidence=show_confidence,
            show_track_id=show_track_id,
        )
        cv2.putText(
            canvas,
            label,
            (x1, max(15, y1 - config.UI.label_offset_px)),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.UI.label_scale,
            color,
            config.UI.box_thickness,
            cv2.LINE_AA,
        )

    return canvas


def open_source(target: str | int, settings: dict[str, object]) -> VideoSource:
    """Construit la source d'images correspondant au choix de l'opérateur.

    L'interface ne connaît que deux choses : une désignation (chemin ou index) et
    l'intention de l'opérateur. `detect_kind()` fait le reste — c'est le module
    `source.py` qui sait qu'un entier est une webcam et qu'un `://` est un flux
    réseau, pas `app.py`.

    Args:
        target: Chemin de fichier, index de webcam ou URL de flux.
        settings: Réglages de la session.

    Returns:
        La source, **non encore ouverte** : l'ouverture appartient au bloc
        `with` de `process_video`, qui garantit sa libération.
    """
    cadence = replace(config.REALTIME, enabled=bool(settings.get("realtime", True)))
    return VideoSource(target, kind=detect_kind(target), realtime=cadence)


def process_video(
    source: VideoSource,
    settings: dict[str, object],
    pipeline,
    boot_slot=None,
) -> None:
    """Boucle principale : lit la source, exécute le pipeline, affiche en direct.

    C'est ici que le flux de données prend forme, une frame à la fois :

        VideoSource.frames()       : image + les deux horodatages
          -> Detector.track()      : boîtes + classes + track_id
          -> Tracker.update()      : mémoire temporelle par objet
          -> ZoneManager.zones_for : zones occupées par chaque objet
          -> TrackedObject.update_zones()
          -> EventEngine.evaluate(): règles zone + durée + type -> Event
          -> ReportGenerator       : Event -> rapport

    **Pourquoi `VideoSource` et non `cv2.VideoCapture` directement.** Une capture
    brute ne sait ni se rouvrir après une coupure réseau, ni écarter les images
    accumulées pendant le traitement, ni surtout dire l'heure : sur un direct,
    `frame_index / fps` sous-estime toutes les durées dès qu'une image est
    sautée, et aucune règle temporelle n'est alors fiable. `Frame` porte le temps
    métier déjà calculé — reconstruit sur un fichier, observé à l'horloge en
    direct. L'interface n'a plus à en décider.

    Args:
        source: Source d'images, non encore ouverte.
        settings: Réglages issus de `render_sidebar()`.
        pipeline: Composants retournés par `build_pipeline()`.
        boot_slot: Emplacement portant l'écran d'initialisation affiché pendant
            le chargement du modèle. Vidé dès la première image affichée.

    Raises:
        VideoSourceError: Si la source ne peut pas être ouverte.
    """
    tracker = pipeline.tracker
    zone_manager = pipeline.zones
    event_engine = pipeline.events
    timeline = pipeline.timeline
    lines = pipeline.lines

    stride = max(1, int(settings["frame_stride"]))
    max_seconds = float(settings["max_seconds"])

    with st.container(border=True):
        st.markdown('<p class="sr-section">Analyse en cours</p>', unsafe_allow_html=True)
        image_slot = st.empty()
        status_slot = st.empty()
        progress = st.progress(0.0)

    # L'écran d'initialisation prend la place de la vidéo tant qu'aucune image
    # n'a été traitée. Une attente sans signe de vie se lit comme une panne.
    image_slot.markdown(
        _boot_screen_html(
            "INITIALISATION",
            "Chargement du modèle et calibration de la cadence...",
        ),
        unsafe_allow_html=True,
    )

    processed = 0
    lues = 0
    ecartees = 0

    try:
        # Le gestionnaire de contexte garantit la libération de la capture :
        # sans elle, un fichier reste verrouillé sous Windows et une webcam
        # reste allumée jusqu'à l'arrêt du processus.
        with source:
            for frame in source.frames(max_seconds=max_seconds):
                lues += 1
                ecartees += frame.dropped

                if frame.index % stride:
                    continue

                image = frame.image
                if config.VIDEO.max_width and image.shape[1] > config.VIDEO.max_width:
                    scale = config.VIDEO.max_width / image.shape[1]
                    image = cv2.resize(image, None, fx=scale, fy=scale)

                if not zone_manager.is_initialized:
                    zone_manager.initialize(image.shape)
                if not lines.is_initialized:
                    lines.initialize(image.shape)

                # `video_time` est **imposé** par la source : c'est le seul moyen
                # de rester juste en direct, où le compteur d'images ne mesure
                # plus le temps écoulé.
                objects = tracker.update(
                    image,
                    frame_index=frame.index,
                    fps=source.fps,
                    wall_time=frame.wall_time,
                    video_time=frame.video_time,
                )

                for obj in objects:
                    # `zones_for` rend des marges signées, relatives à la taille
                    # apparente de l'objet ; l'hystérésis les exploite dans
                    # `update_zones`. L'interface ne fait que transmettre.
                    obj.update_zones(
                        zone_manager.zones_for(obj.detection), tracker.video_time
                    )

                # `evaluate()` recompte l'occupation avant d'appliquer les
                # règles : les variations sont donc déjà à jour ici.
                variations = event_engine.occupancy.changes(notable_only=True)
                avant = len(variations)
                events = event_engine.evaluate(
                    objects, tracker, image, tracker.video_time, frame.wall_time
                )
                nouvelles = event_engine.occupancy.changes(notable_only=True)[avant:]
                passages = lines.update(tracker.active(), tracker.video_time)
                if events:
                    st.session_state["events"].extend(events)

                # La chronologie observe les objets **retenus** par le tracker,
                # pas ceux vus sur cette seule frame : la rétention absorbe déjà
                # les occlusions courtes. Lui passer les objets de la frame
                # produirait une fausse sortie puis une fausse entrée à chaque
                # fois qu'un objet passe derrière un poteau.
                timeline.observe(
                    tracker.video_time,
                    tracker.active(),
                    events,
                    frame.wall_time,
                    occupancy_changes=nouvelles,
                    crossings=passages,
                )

                if boot_slot is not None:
                    # La première image traitée est le seul signal fiable que le
                    # système est prêt : le modèle est chargé, la source lit, le
                    # pipeline tourne.
                    boot_slot.empty()
                    boot_slot = None

                image_slot.image(
                    annotate(image, zone_manager, objects, settings),
                    channels="BGR",
                    **stretch(),
                )
                status_slot.markdown(
                    _status_line(
                        source, tracker, ecartees, len(st.session_state["events"])
                    )
                )

                # Un direct n'a pas d'avancement, seulement une durée : la barre
                # suit alors la fraction de durée maximale déjà écoulée.
                avancement = source.progress()
                if avancement is None:
                    avancement = min(1.0, frame.video_time / max_seconds) if max_seconds else 0.0
                progress.progress(avancement)

                processed += 1
    except SourceDisconnectedError as exc:
        # Un flux perdu en cours de route n'annule pas l'analyse déjà produite :
        # les incidents relevés restent exploitables, seule la suite manque.
        st.warning(f"{exc} L'analyse effectuée jusque-là reste exploitable.", icon="⚠")
    finally:
        progress.empty()

    if ecartees:
        origine = "la caméra" if source.is_live else "le rythme de la vidéo"
        st.caption(
            f"{ecartees} image(s) écartée(s) pour suivre {origine} : le traitement "
            "est plus lent que la source. Les **durées restent exactes** — seule "
            "l'exhaustivité de l'observation est réduite. Pour tout examiner, "
            "désactivez « Analyse en temps réel » à gauche ; l'analyse durera "
            "alors plus longtemps que la vidéo."
        )

    logger.info(
        "Analyse terminée : %d frames traitées sur %d lues (%d écartées).",
        processed,
        lues,
        ecartees,
    )


def _status_line(
    source: VideoSource, tracker: Tracker, dropped: int, incidents: int
) -> str:
    """Ligne d'état affichée sous la vidéo pendant l'analyse.

    Le compte d'incidents est **passé en argument** plutôt que relu dans
    `st.session_state` : une fonction de mise en forme qui va chercher son état
    dans une variable globale n'est testable qu'en simulant tout Streamlit.

    Nommer l'horloge n'est pas décoratif. Sur un fichier, `t` est reconstruit
    depuis le numéro d'image et l'analyse est reproductible ; en direct, `t` est
    l'heure écoulée. L'opérateur doit savoir laquelle il lit avant de recopier
    une durée dans un rapport.

    Args:
        source: Source en cours de lecture.
        tracker: Tracker, pour le temps vidéo et les comptages.
        dropped: Nombre cumulé d'images écartées.
        incidents: Nombre d'incidents relevés depuis le début de la session.

    Returns:
        Le texte Markdown de la ligne d'état.
    """
    horloge = "temps réel" if source.is_live else "temps vidéo"
    ligne = (
        f"**t = {tracker.video_time:.1f} s** ({horloge}) · objets suivis : "
        f"{len(tracker.active())} · incidents : {incidents} · "
        f"{tracker.counts() or '—'}"
    )
    if source.is_warming_up:
        # Pendant la chauffe, aucune image n'est écartée : le dire évite de
        # prendre les premières secondes saccadées pour un dysfonctionnement.
        ligne += f" · initialisation ({source.warmup_remaining:.0f} s)"
    if dropped:
        ligne += f" · {dropped} image(s) écartée(s)"
    if source.lag_s > 1.0:
        # Le rattrapage ne suffit plus : la machine est durablement dépassée.
        ligne += f" · retard {source.lag_s:.0f} s"
    return ligne


# ---------------------------------------------------------------------------
# Restitution
# ---------------------------------------------------------------------------


NIVEAU_PASTILLE = {
    config.PriorityLevel.CRITICAL: "🔴 Critique",
    config.PriorityLevel.HIGH: "🟠 Élevé",
    config.PriorityLevel.MEDIUM: "🟡 Moyen",
    config.PriorityLevel.LOW: "🔵 Faible",
}


def _incident_rows(events: list[Event]) -> list[dict[str, object]]:
    """Met les incidents en forme pour l'affichage à l'écran.

    Distinct de `Event.to_dict()`, qui sert l'export CSV et doit rester complet
    et stable. Ici on choisit **ce qu'un opérateur lit en diagonale** : la
    priorité d'abord, les colonnes techniques ensuite, l'ordre des colonnes
    faisant partie du message.

    Args:
        events: Incidents à afficher.

    Returns:
        Une ligne par incident, triée par priorité décroissante.
    """
    tries = sorted(
        events,
        key=lambda event: event.priority.value if event.priority else 0.0,
        reverse=True,
    )
    return [
        {
            "Priorité": (
                NIVEAU_PASTILLE.get(event.priority.level, "—") if event.priority else "—"
            ),
            "Score": round(event.priority.value) if event.priority else 0,
            "Heure": event.timestamp.strftime("%H:%M:%S"),
            "Temps vidéo": f"{event.video_time:.0f} s",
            "Type": event.event_type.value.replace("_", " "),
            "Objet": f"#{event.track_id} {event.class_name}",
            "Zone": event.zone_name or "—",
            "Durée": f"{event.duration_s:.0f} s",
            "Confiance": f"{event.confidence:.0%}",
        }
        for event in tries
    ]


def render_incident_table(events: list[Event]) -> None:
    """Affiche le tableau des incidents détectés.

    Args:
        events: Incidents accumulés durant la session.
    """
    if not events:
        st.info(
            "Aucun incident détecté pour l'instant. C'est souvent normal : les "
            "règles exigent une **durée**. Traverser le champ ne déclenche rien ; "
            "y stationner au-delà du seuil déclenche.",
            icon="ℹ",
        )
        return

    render_status_strip(events)

    filtre_type, filtre_priorite = st.columns(2)
    types = sorted({event.event_type.value for event in events})
    niveaux = [
        niveau
        for niveau in (
            config.PriorityLevel.CRITICAL,
            config.PriorityLevel.HIGH,
            config.PriorityLevel.MEDIUM,
            config.PriorityLevel.LOW,
        )
        if any(e.priority and e.priority.level is niveau for e in events)
    ]

    with filtre_type:
        types_retenus = st.multiselect("Filtrer par type", types, default=types)
    with filtre_priorite:
        niveaux_retenus = st.multiselect(
            "Filtrer par priorité",
            [n.value for n in niveaux],
            default=[n.value for n in niveaux],
        )

    filtres = [
        event
        for event in events
        if event.event_type.value in types_retenus
        and (event.priority.level.value if event.priority else "") in niveaux_retenus
    ]

    if not filtres:
        st.warning("Aucun incident ne correspond aux filtres.", icon="⚠")
        return

    st.dataframe(_incident_rows(filtres), hide_index=True, **stretch())
    st.caption(
        f"{len(filtres)} incident(s) affiché(s) sur {len(events)} · triés par "
        "priorité décroissante. Le détail du calcul figure dans le rapport."
    )


def render_incident_detail(events: list[Event], generator: ReportGenerator) -> None:
    """Affiche le détail d'un incident : rapport, image de preuve, export.

    Args:
        events: Incidents disponibles.
        generator: Générateur de rapports.
    """
    if not events:
        return

    st.subheader("Rapport d'incident")

    labels = {
        f"{event.event_id} — {event.event_type.value} (#{event.track_id})": event
        for event in events
    }
    choice = st.selectbox("Incident", options=list(labels))
    event = labels[choice]

    text_column, evidence_column = st.columns([3, 2])

    with text_column:
        st.text_area(
            "Brouillon de rapport",
            value=generator.generate_full_report(event),
            height=380,
        )

    with evidence_column:
        if event.evidence_path and Path(event.evidence_path).is_file():
            st.image(str(event.evidence_path), caption="Preuve visuelle", **stretch())
        else:
            st.caption("Aucune preuve visuelle enregistrée pour cet incident.")

    _render_exports(event, events, generator)


@st.cache_data(show_spinner=False)
def _single_pdf(event_id: str, _event: Event, _generator: ReportGenerator) -> tuple[str, bytes]:
    """Produit le PDF d'un incident, une seule fois par incident.

    Streamlit réexécute tout le script à chaque interaction — un filtre déplacé,
    une case cochée. Sans mémoïsation, chaque clic réécrirait les PDF sur le
    disque, images de preuve comprises. Les arguments préfixés d'un `_` sont
    exclus du calcul de la clé de cache : `event_id` suffit à identifier le
    document, et un `Event` (qui porte un dictionnaire) n'est de toute façon pas
    hachable par Streamlit.

    Args:
        event_id: Identifiant de l'incident, seule clé de cache.
        _event: Incident à exporter.
        _generator: Générateur de rapports.

    Returns:
        Le couple `(nom_de_fichier, contenu)`.
    """
    path = _generator.export_pdf(_event)
    return path.name, path.read_bytes()


@st.cache_data(show_spinner=False)
def _batch_exports(
    event_ids: tuple[str, ...], _events: list[Event], _generator: ReportGenerator
) -> tuple[tuple[str, bytes], tuple[str, bytes]]:
    """Produit la synthèse PDF et le journal CSV, une fois par lot d'incidents.

    Args:
        event_ids: Identifiants du lot — la clé de cache.
        _events: Incidents à exporter.
        _generator: Générateur de rapports.

    Returns:
        Deux couples `(nom_de_fichier, contenu)` : la synthèse puis le CSV.
    """
    pdf = _generator.export_batch_pdf(_events)
    csv_file = _generator.export_csv(_events)
    return (pdf.name, pdf.read_bytes()), (csv_file.name, csv_file.read_bytes())


def _render_exports(event: Event, events: list[Event], generator: ReportGenerator) -> None:
    """Propose les exports PDF et CSV.

    Les fichiers sont produits sur disque puis relus pour alimenter les boutons
    de téléchargement : `st.download_button` exige des octets en mémoire, et
    écrire le fichier laisse en plus une trace dans `reports/`.

    Args:
        event: Incident sélectionné.
        events: Tous les incidents de la session.
        generator: Générateur de rapports.
    """
    pdf_column, batch_column, csv_column = st.columns(3)

    try:
        with pdf_column:
            name, data = _single_pdf(event.event_id, event, generator)
            st.download_button(
                "Rapport PDF de cet incident",
                data=data,
                file_name=name,
                mime="application/pdf",
            )

        (batch_name, batch_data), (csv_name, csv_data) = _batch_exports(
            tuple(item.event_id for item in events), events, generator
        )

        with batch_column:
            st.download_button(
                f"Synthèse PDF ({len(events)} incidents)",
                data=batch_data,
                file_name=batch_name,
                mime="application/pdf",
            )

        with csv_column:
            st.download_button(
                "Journal CSV",
                data=csv_data,
                file_name=csv_name,
                mime="text/csv",
            )
    except SentinelError as exc:
        st.error(f"Export impossible : {exc}")


def render_session_report(
    events: list[Event],
    timeline: Timeline,
    context: SessionContext | None,
    generator: SessionReportGenerator,
) -> None:
    """Affiche le rapport de session à deux niveaux et ses exports.

    **Pourquoi deux ordres de lecture.** La partie 1 trie les incidents par
    score décroissant : c'est la liste d'action, le premier de la liste est celui
    à regarder en premier. La partie 2 rejoue le déroulé par tranches de temps
    vidéo : c'est la mise en récit. Trier par priorité fait perdre la causalité,
    trier par temps fait perdre l'urgence — un sac abandonné se comprend en
    voyant qui l'a posé trois tranches plus tôt, savoir par où commencer demande
    l'ordre inverse.

    Rien ici ne suppose que la session est close : `Timeline` s'alimente au fil
    de l'eau et se consulte à tout moment. Le rapport reflète l'état à son
    instant d'édition, ce qui rendra possible l'export en pleine surveillance.

    Args:
        events: Incidents de la session.
        timeline: Chronologie observée pendant l'analyse.
        context: Réglages de la session, pour l'en-tête.
        generator: Générateur de rapports de session.
    """
    st.subheader("Rapport de session")
    st.caption(
        f"Deux lectures des mêmes faits : les incidents **par priorité**, puis le "
        f"déroulé **par tranches de {config.TIMELINE.slice_seconds:.0f} s** de temps "
        "vidéo. Les tranches sans changement ne sont pas listées."
    )

    try:
        texte = generator.generate_session_report(events, timeline, context)
    except SentinelError as exc:
        st.error(f"Rapport de session indisponible : {exc}")
        return

    st.text_area("Brouillon de rapport de session", value=texte, height=420)

    pdf_column, csv_column, info_column = st.columns([1, 1, 2])

    try:
        nom_pdf, donnees_pdf = _session_pdf(
            tuple(item.event_id for item in events), len(timeline), events, timeline, context, generator
        )
        with pdf_column:
            st.download_button(
                "Rapport de session (PDF)",
                data=donnees_pdf,
                file_name=nom_pdf,
                mime="application/pdf",
                type="primary",
            )
    except SentinelError as exc:
        with pdf_column:
            st.warning(f"PDF indisponible : {exc}", icon="⚠")

    try:
        nom_csv, donnees_csv = _timeline_csv(len(timeline), timeline, generator)
        with csv_column:
            st.download_button(
                "Chronologie (CSV)",
                data=donnees_csv,
                file_name=nom_csv,
                mime="text/csv",
            )
    except SentinelError:
        with csv_column:
            st.caption("Chronologie vide : rien à exporter.")

    with info_column:
        tranches = timeline.slices()
        st.caption(
            f"{len(timeline)} fait(s) marquant(s) sur {len(tranches)} tranche(s) "
            f"retenue(s) · {len(events)} incident(s)."
        )


@st.cache_data(show_spinner=False)
def _session_pdf(
    event_ids: tuple[str, ...],
    fact_count: int,
    _events: list[Event],
    _timeline: Timeline,
    _context: SessionContext | None,
    _generator: SessionReportGenerator,
) -> tuple[str, bytes]:
    """Produit le PDF de session, une fois par état de session.

    La clé de cache est le couple *(identifiants d'incidents, nombre de faits)* :
    ce sont les deux seules choses qui font changer le document. Sans cette
    mémoïsation, chaque filtre déplacé réécrirait le PDF sur le disque.

    Args:
        event_ids: Identifiants des incidents — première moitié de la clé.
        fact_count: Nombre de faits de la chronologie — seconde moitié.
        _events: Incidents à décrire.
        _timeline: Chronologie à rejouer.
        _context: Réglages de la session.
        _generator: Générateur de rapports de session.

    Returns:
        Le couple `(nom_de_fichier, contenu)`.
    """
    chemin = _generator.export_session_pdf(_events, _timeline, _context)
    return chemin.name, chemin.read_bytes()


@st.cache_data(show_spinner=False)
def _timeline_csv(
    fact_count: int, _timeline: Timeline, _generator: SessionReportGenerator
) -> tuple[str, bytes]:
    """Produit le CSV de la chronologie, une fois par état de chronologie.

    Args:
        fact_count: Nombre de faits — la clé de cache.
        _timeline: Chronologie à exporter.
        _generator: Générateur de rapports de session.

    Returns:
        Le couple `(nom_de_fichier, contenu)`.
    """
    chemin = _generator.export_timeline_csv(_timeline)
    return chemin.name, chemin.read_bytes()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def purge_expired_evidence() -> int:
    """Applique la politique de conservation des captures, une fois par session.

    `@st.cache_resource` sert ici de « une seule fois » : Streamlit réexécute le
    script à chaque interaction, et balayer le dossier des preuves à chaque clic
    serait aussi inutile que coûteux.

    Le démarrage est le bon moment : c'est le seul instant où l'on est certain
    qu'aucune analyse n'est en cours, donc qu'aucun fichier examiné n'est en
    train d'être écrit.

    Returns:
        Le nombre de fichiers supprimés.
    """
    supprimes = EvidenceWriter().purge_expired()
    if supprimes:
        logger.info("Purge au démarrage : %d preuve(s) expirée(s).", supprimes)
    return supprimes


def main() -> None:
    """Point d'entrée de l'application.

    La page suit les trois temps d'une session — **préparer, analyser,
    dépouiller** — et n'affiche que ce qui a du sens à l'instant présent. Le
    panneau de préparation se replie dès qu'une analyse a produit des résultats,
    et la section des incidents n'apparaît qu'une fois qu'il y en a. Tout montrer
    en permanence obligerait à faire défiler la page pour atteindre l'utile.
    """
    configure_page()  # doit rester la toute première commande Streamlit
    inject_styles()
    render_header()
    purge_expired_evidence()

    st.session_state.setdefault("events", [])
    events: list[Event] = st.session_state["events"]

    settings = render_sidebar()
    if settings["weights"] is None:
        st.stop()

    # ── 1. Préparer ──────────────────────────────────────────────────────────
    st.markdown(
        '<p class="sr-section">Étape 1 · Préparer la session</p>',
        unsafe_allow_html=True,
    )
    with st.expander(
        "Source vidéo et objets surveillés",
        expanded=not events,
        icon="⚙",
    ):
        render_source_panel(settings)
        st.divider()
        launch = render_action_bar(settings)

    # ── 2. Analyser ──────────────────────────────────────────────────────────
    if launch:
        source = resolve_source(settings)
        if source is None:
            st.warning(
                "Choisissez d'abord une vidéo à l'étape 1.",
                icon="⚠",
            )
        else:
            st.markdown(
                '<p class="sr-section">Étape 2 · Analyse</p>', unsafe_allow_html=True
            )
            # Toutes les erreurs métier remontent ici sous forme de SentinelError :
            # l'utilisateur voit un message actionnable, jamais une trace Python.
            # Peint avant `build_pipeline` : c'est lui qui charge les poids
            # YOLO, l'opération la plus longue de la session. Sans signe de vie
            # à ce moment-là, l'application paraît figée.
            boot = st.empty()
            boot.markdown(
                _boot_screen_html(
                    "INITIALISATION",
                    "Chargement du modèle de détection...",
                ),
                unsafe_allow_html=True,
            )
            try:
                pipeline = build_pipeline(str(settings["weights"]), settings)
                st.session_state["events"] = []
                flux = open_source(source, settings)
                process_video(flux, settings, pipeline, boot_slot=boot)
                # La chronologie et le contexte survivent à la boucle : c'est ce
                # qui permet de rééditer le rapport de session à chaque rerun
                # sans relancer l'analyse.
                st.session_state["generator"] = pipeline.generator
                st.session_state["timeline"] = pipeline.timeline
                st.session_state["context"] = session_context(flux, settings)
                events = st.session_state["events"]
                st.success(
                    f"Analyse terminée : {len(events)} incident(s) détecté(s).",
                    icon="✅",
                )
            except SentinelError as exc:
                st.error(str(exc), icon="❌")
            except Exception:  # pragma: no cover - filet de sécurité
                logger.exception("Erreur inattendue pendant l'analyse.")
                st.error(
                    "Erreur inattendue pendant l'analyse. Consultez le terminal "
                    "pour la trace complète.",
                    icon="❌",
                )
            finally:
                # Un modèle introuvable ou une source injoignable laisserait
                # sinon l'écran de chargement tourner indéfiniment au-dessus du
                # message d'erreur.
                boot.empty()

    # ── 3. Dépouiller ────────────────────────────────────────────────────────
    events = st.session_state["events"]
    st.markdown(
        '<p class="sr-section">Étape 3 · Incidents et rapports</p>',
        unsafe_allow_html=True,
    )
    render_incident_table(events)

    generator = st.session_state.get("generator") or SessionReportGenerator()
    if events:
        st.divider()
        render_incident_detail(events, generator)

    # Le rapport de session tient debout sans incident : une surveillance calme
    # est un résultat, et la chronologie le documente. Il ne s'affiche donc pas
    # « s'il y a des incidents » mais « si une analyse a eu lieu ».
    timeline = st.session_state.get("timeline")
    if timeline is not None and isinstance(generator, SessionReportGenerator):
        st.divider()
        render_session_report(
            events, timeline, st.session_state.get("context"), generator
        )


if __name__ == "__main__":
    main()
