"""Tests de l'interface Streamlit, exécutés **sans navigateur**.

`streamlit.testing.v1.AppTest` exécute le script comme le ferait le serveur et
donne accès aux éléments produits. C'est ce qui permet de vérifier en continu
qu'`app.py` démarre sans exception — la régression la plus coûteuse du projet,
puisqu'elle rend l'application entièrement inutilisable.

Ces tests ne chargent aucun modèle : au démarrage, `app.py` se contente de lire
`models/` pour peupler le menu déroulant ; les poids ne sont ouverts qu'au clic
sur « Lancer l'analyse ».
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import config
from sentinel.detection import Detection
from sentinel.tracker import TrackedObject
from sentinel.zones import ZoneManager

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def rendered() -> AppTest:
    """Application exécutée une fois pour tout le module."""
    return AppTest.from_file(str(APP), default_timeout=120).run()


# ---------------------------------------------------------------------------
# Démarrage
# ---------------------------------------------------------------------------


def test_app_starts_without_exception(rendered: AppTest) -> None:
    """Le script doit s'exécuter de bout en bout sans lever."""
    assert not rendered.exception, [str(e.value) for e in rendered.exception]


def test_startup_renders_the_expected_controls(rendered: AppTest) -> None:
    """Logo, sélecteur de modèle et boutons d'action sont présents."""
    # Filtrer sur la balise et non sur le nom de classe : le bloc `<style>` cite
    # lui aussi « sr-logo-card » dans ses commentaires.
    logo_blocks = [
        m.value for m in rendered.markdown if '<div class="sr-logo-card">' in m.value
    ]

    assert logo_blocks, "L'écrin du logo n'a pas été rendu."
    assert "data:image/svg+xml;base64," in logo_blocks[0]
    assert "Modèle" in [s.label for s in rendered.selectbox]
    assert "Lancer l'analyse" in [b.label for b in rendered.button]


def test_injected_html_is_always_balanced() -> None:
    """Chaque bloc `unsafe_allow_html` doit contenir du HTML complet.

    Ouvrir une balise dans un appel à `st.markdown` et la fermer dans un autre
    semble marcher, mais le navigateur referme automatiquement le premier bloc
    et laisse la fermante du second orpheline. Le DOM réel diverge alors de
    celui que React gère, et un réaffichage ultérieur échoue sur
    « NotFoundError: Failed to execute 'removeChild' on 'Node' » — une erreur
    qui se déclenche loin de sa cause, typiquement pendant la boucle vidéo.
    """
    import ast
    import re

    source = APP.read_text(encoding="utf-8")
    tag = re.compile(r"<(/?)([a-zA-Z][\w-]*)[^>]*?(/?)>")
    void_tags = {"img", "br", "hr", "input", "meta", "link"}

    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not any(
            keyword.arg == "unsafe_allow_html" and getattr(keyword.value, "value", False)
            for keyword in node.keywords
        ):
            continue

        argument = node.args[0] if node.args else None
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            html = argument.value
        elif isinstance(argument, ast.JoinedStr):
            html = "".join(
                part.value for part in argument.values if isinstance(part, ast.Constant)
            )
        else:
            continue  # fragment construit ailleurs, couvert par son propre test

        if "<style>" in html:
            assert html.count("<style>") == html.count("</style>"), node.lineno
            continue

        stack: list[str] = []
        for closing, name, self_closing in tag.findall(html):
            lowered = name.lower()
            if lowered in void_tags or self_closing:
                continue
            if closing:
                assert stack and stack[-1] == lowered, f"ligne {node.lineno} : </{lowered}>"
                stack.pop()
            else:
                stack.append(lowered)

        assert not stack, f"ligne {node.lineno} : balises non fermées {stack}"


def test_logo_card_is_a_single_self_contained_block() -> None:
    """Le fragment du logo se suffit à lui-même : ouverture, image, fermeture."""
    import app

    html = app._logo_card_html(config.BRAND.logo_on_dark)

    assert html.startswith('<div class="sr-logo-card">')
    assert html.endswith("</div>")
    assert html.count("<div") == 1 and html.count("</div>") == 1
    assert "data:image/svg+xml;base64," in html


def test_regulators_live_in_the_sidebar(rendered: AppTest) -> None:
    """Les réglages continus sont des curseurs, regroupés à gauche."""
    sliders = [s.label for s in rendered.slider]

    assert "Seuil de confiance" in sliders
    assert "Durées de déclenchement" in sliders
    assert "Analyser une frame sur" in sliders
    assert [t.label for t in rendered.toggle] == [
        "Score de confiance",
        "Identifiants de suivi",
    ]


def test_no_zone_overlay_control_is_offered() -> None:
    """La surveillance est globale : il n'y a aucune zone à afficher ou masquer.

    Proposer un interrupteur sans effet visible serait plus déroutant qu'utile.
    """
    app_test = AppTest.from_file(str(APP), default_timeout=120).run()

    assert not any("one" in t.label.lower() for t in app_test.toggle)


def test_source_and_classes_live_in_the_main_panel(rendered: AppTest) -> None:
    """Le choix de la source et des objets surveillés occupe la zone de droite."""
    assert [r.label for r in rendered.radio] == ["Type de source"]
    assert [m.label for m in rendered.multiselect] == ["Catégories"]


def test_weapon_group_warns_about_the_coco_limitation() -> None:
    """Sélectionner « Objets dangereux » doit avertir de l'absence d'armes à feu."""
    app_test = AppTest.from_file(str(APP), default_timeout=120).run()

    app_test.multiselect[0].select("Objets dangereux").run()

    assert any("pistolet" in w.value for w in app_test.warning)


def test_custom_classes_can_be_added_and_cleared() -> None:
    """L'ajout à la carte alimente la liste transmise au modèle, et se retire."""
    app_test = AppTest.from_file(str(APP), default_timeout=120).run()
    app_test.session_state["custom_classes"] = ["laptop"]
    app_test.run()

    assert "Retirer les ajouts" in [b.label for b in app_test.button]

    [b for b in app_test.button if b.label == "Retirer les ajouts"][0].click().run()

    assert app_test.session_state["custom_classes"] == []
    assert not app_test.exception


def test_startup_does_not_produce_error_banners(rendered: AppTest) -> None:
    """Aucun `st.error` ne doit s'afficher sur une installation correcte."""
    assert [e.value for e in rendered.error] == []


def test_empty_session_explains_why_nothing_is_detected(rendered: AppTest) -> None:
    """Sans incident, l'interface explique la règle de durée plutôt que de rester muette."""
    assert any("durée" in info.value for info in rendered.info)


# ---------------------------------------------------------------------------
# Garde-fous d'API
# ---------------------------------------------------------------------------


def test_stretch_targets_a_parameter_the_running_version_accepts() -> None:
    """`stretch()` doit produire l'argument compris par le Streamlit installé.

    Les deux conventions ne sont pas interchangeables : `width="stretch"` sur une
    version 1.44 lève `TypeError: '<=' not supported between instances of 'str'
    and 'int'`, et `use_container_width` déclenche un avertissement de
    dépréciation à partir de la 1.49.
    """
    import inspect

    import streamlit as st

    import app

    kwargs = app.stretch()

    assert len(kwargs) == 1
    assert next(iter(kwargs)) in inspect.signature(st.button).parameters


def test_stretch_is_accepted_by_every_widget_that_uses_it() -> None:
    """Vérification par l'exécution, sur les trois widgets concernés.

    C'est le test qui aurait évité le plantage en production : la signature de
    `st.image` accepte `width` dans les deux versions, mais avec une sémantique
    différente (entier avant la 1.49, jeton après). Seul un appel réel le montre.
    """
    def _script() -> None:
        # Tous les imports sont internes : AppTest réexécute le code source de la
        # fonction dans un contexte neuf, où les variables de la portée
        # englobante n'existent pas.
        import numpy as np
        import streamlit as st

        import app

        st.button("bouton", **app.stretch())
        st.image(np.zeros((20, 30, 3), dtype=np.uint8), **app.stretch())
        st.dataframe([{"colonne": 1}], **app.stretch())

    rendered = AppTest.from_function(_script, default_timeout=60).run()

    assert not rendered.exception, [str(e.value) for e in rendered.exception]


def test_no_expander_is_nested_inside_another() -> None:
    """Streamlit interdit d'imbriquer deux `st.expander`.

    Le panneau de configuration est lui-même un expander : les fonctions qu'il
    appelle ne doivent donc pas en créer. Ce test est indispensable parce que
    Streamlit 1.54 tolère l'imbrication en silence là où la 1.44 lève
    « Expanders may not be nested inside other expanders » — un défaut invisible
    sur une version, fatal sur l'autre.
    """
    import ast

    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    bodies = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    # Le contrôle porte sur l'**appel** `st.expander(` et non sur le mot seul :
    # les commentaires de ces fonctions expliquent précisément pourquoi elles
    # n'en créent pas.
    for name in ("render_source_panel", "_render_class_picker", "render_action_bar"):
        assert name in bodies, f"{name} introuvable dans app.py"
        assert "st.expander(" not in bodies[name], (
            f"{name} crée un expander alors qu'elle est appelée depuis un expander."
        )


def test_no_widget_hardcodes_a_width_convention() -> None:
    """Aucun appel de widget ne doit court-circuiter `stretch()`.

    Le contrôle porte sur les appels eux-mêmes, pas sur le texte du fichier : les
    commentaires de la couche de compatibilité citent forcément les deux
    conventions.
    """
    import re

    source = APP.read_text(encoding="utf-8")
    calls = re.findall(r"st\.(?:image|dataframe|button)\((?:[^()]|\([^()]*\))*\)", source)
    faulty = [
        call
        for call in calls
        if "use_container_width" in call or 'width="stretch"' in call
    ]

    assert faulty == [], faulty
    assert "use_column_width" not in source


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def _tracked(x: float, zones: set[str]) -> TrackedObject:
    """Objet suivi placé en `x`, occupant les zones indiquées."""
    from datetime import datetime

    detection = Detection(0, "person", 0.9, (x, 400.0, x + 40.0, 600.0), 1)
    obj = TrackedObject(
        track_id=1,
        class_name="person",
        first_seen=0.0,
        last_seen=1.0,
        first_seen_wall=datetime(2026, 8, 13, 12, 0),
        last_seen_wall=datetime(2026, 8, 13, 12, 0),
        detection=detection,
    )
    obj.zones = zones
    return obj


def test_annotate_distinguishes_restricted_zones() -> None:
    """Un objet en zone restreinte est encadré en rouge, les autres en vert.

    L'opérateur doit pouvoir juger la situation sur l'image, sans lire le tableau.
    """
    import app

    manager = ZoneManager()
    manager.initialize((720, 1280, 3))
    restricted = manager.restricted_zone_names()[0]

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    settings = {"show_zones": False, "show_confidence": True, "show_track_id": True}

    inside = app.annotate(frame, manager, [_tracked(100.0, {restricted})], settings)
    outside = app.annotate(frame, manager, [_tracked(100.0, set())], settings)

    # BGR : le rouge pur n'a que le canal 2, le vert que le canal 1.
    assert inside[:, :, 2].sum() > 0 and inside[:, :, 1].sum() == 0
    assert outside[:, :, 1].sum() > 0 and outside[:, :, 2].sum() == 0


def test_annotate_never_modifies_the_source_frame() -> None:
    """La frame d'origine sert de preuve : elle ne doit jamais être annotée en place."""
    import app

    manager = ZoneManager()
    manager.initialize((720, 1280, 3))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    app.annotate(frame, manager, [_tracked(100.0, set())], {"show_zones": True})

    assert frame.sum() == 0


# ---------------------------------------------------------------------------
# Source vidéo
# ---------------------------------------------------------------------------


def test_resolve_source_returns_the_webcam_index() -> None:
    """Le mode webcam s'appuie sur l'index de la configuration."""
    import app

    assert app.resolve_source({"source_kind": "Webcam"}) == config.VIDEO.webcam_index


def test_resolve_source_without_upload_returns_none() -> None:
    """Sans fichier choisi, aucune analyse ne doit démarrer."""
    import app

    assert app.resolve_source({"source_kind": "Fichier vidéo", "upload": None}) is None


# ---------------------------------------------------------------------------
# Catalogue de classes
# ---------------------------------------------------------------------------


def test_every_group_references_real_coco_classes() -> None:
    """Un intitulé métier ne doit jamais pointer vers une classe inexistante.

    Sans ce contrôle, une faute de frappe dans `CLASS_GROUPS` produirait une
    catégorie cochable qui ne détecterait jamais rien — le pire des défauts pour
    un système d'alerte, puisqu'il est silencieux.
    """
    for group in config.CLASS_GROUPS:
        assert group.classes, f"Le groupe '{group.label}' est vide."
        for name in group.classes:
            assert name in config.COCO_CLASSES, f"'{name}' n'existe pas dans COCO."


def test_default_selection_covers_every_class_used_by_the_rules() -> None:
    """Les catégories cochées par défaut doivent alimenter toutes les règles.

    Autrement, l'application démarrerait avec des règles d'incident incapables
    de se déclencher.
    """
    active = set(config.classes_for_groups(config.DEFAULT_CLASS_GROUPS))

    for rule in config.EVENT_RULES:
        for name in rule.classes:
            assert name in active, f"{rule.event_type.value} cible '{name}', non surveillé."


def test_group_resolution_deduplicates_and_keeps_order() -> None:
    """Deux groupes partageant une classe ne la transmettent qu'une fois."""
    resolved = config.classes_for_groups(["Personnes", "Véhicules"])

    assert resolved[0] == "person"
    assert len(resolved) == len(set(resolved))


def test_no_group_promises_firearm_detection() -> None:
    """Aucune catégorie ne doit laisser croire qu'un pistolet est détectable.

    COCO n'a pas de classe d'arme à feu ; toute catégorie qui en suggérerait une
    serait une promesse intenable.
    """
    for group in config.CLASS_GROUPS:
        for name in group.classes:
            assert not any(k in name for k in ("gun", "pistol", "rifle", "firearm"))


# ---------------------------------------------------------------------------
# Curseur de durées
# ---------------------------------------------------------------------------


def test_duration_slider_scales_the_rules() -> None:
    """Le facteur multiplie réellement les seuils : ce n'est pas un bouton décoratif."""
    import app

    doubled = app.scaled_rules({"min_duration_factor": 2.0})

    for original, scaled in zip(config.EVENT_RULES, doubled):
        assert scaled.min_duration_s == original.min_duration_s * 2


def test_scaling_preserves_the_anti_bounce_invariant() -> None:
    """`cooldown_s` doit rester supérieur à `min_duration_s`, quel que soit le facteur.

    Sinon un même objet redéclencherait le même incident en boucle dès que le
    curseur dépasse une certaine valeur.
    """
    import app

    for factor in (0.25, 1.0, 3.0):
        for rule in app.scaled_rules({"min_duration_factor": factor}):
            assert rule.cooldown_s > rule.min_duration_s, f"facteur {factor}"


def test_neutral_factor_returns_the_configuration_untouched() -> None:
    """À facteur 1, on doit retrouver les règles d'origine, sans copie inutile."""
    import app

    assert app.scaled_rules({"min_duration_factor": 1.0}) is config.EVENT_RULES
