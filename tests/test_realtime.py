"""Analyse au rythme réel — le mode par défaut, y compris sur un fichier soumis.

Le problème
------------
Sur un processeur, une inférence YOLO coûte 100 à 150 ms. Analyser **chaque**
image d'une vidéo à 25 images/s demande trois à quatre fois sa durée : dix
minutes de vidéo occupent une demi-heure, pendant laquelle l'opérateur regarde
une image saccadée avancer au ralenti.

Le compromis, et pourquoi il est acceptable
---------------------------------------------
Tenir le rythme suppose de renoncer à voir toutes les images. Ce que ces tests
verrouillent, c'est la frontière exacte de ce renoncement :

* le **temps métier reste exact** — `video_time` vaut `index / fps`, et `grab()`
  fait avancer l'index comme `read()`. Toutes les durées restent justes, donc
  toutes les règles aussi. C'est ce qui rend le mode acceptable par défaut ;
* la **reproductibilité stricte est perdue** — deux machines n'examinent pas les
  mêmes images. D'où le mode exhaustif, qui reste disponible.

La fenêtre d'initialisation
-----------------------------
Les premières secondes ne sont pas représentatives : la première inférence est
souvent dix fois plus lente que les suivantes. Les compter comme du retard ferait
jeter le début de la vidéo — précisément le moment où la scène s'établit.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from sentinel.source import SourceKind, VideoSource

TEMPS_REEL = config.RealtimeConfig(enabled=True, warmup_s=15.0, max_lag_s=0.4)
EXHAUSTIF = config.RealtimeConfig(enabled=False)


class _Capture:
    """Capture OpenCV réduite à ce que `VideoSource` lui demande."""

    def __init__(self, frames: int = 10_000, fps: float = 25.0) -> None:
        self._restantes = frames
        self._fps = fps
        self.grabs = 0
        self.reads = 0
        self._ouverte = True

    def isOpened(self) -> bool:  # noqa: N802 - API OpenCV
        return self._ouverte

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 10_000.0
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def grab(self) -> bool:
        if self._restantes <= 0:
            return False
        self._restantes -= 1
        self.grabs += 1
        return True

    def read(self):
        if self._restantes <= 0:
            return False, None
        self._restantes -= 1
        self.reads += 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        self._ouverte = False


class _Horloge:
    """Horloge pilotée par le test : le temps avance quand on le décide."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock(monkeypatch) -> _Horloge:
    """Remplace l'horloge monotone du module par une horloge contrôlée."""
    fausse = _Horloge()
    monkeypatch.setattr("sentinel.source.time.monotonic", fausse)
    return fausse


def _source(capture: _Capture, *, realtime=TEMPS_REEL, kind=SourceKind.FILE) -> VideoSource:
    """Source ouverte sur une capture donnée."""
    return VideoSource(
        "video.mp4" if kind is SourceKind.FILE else 0,
        kind=kind,
        capture_factory=lambda *_: capture,
        realtime=realtime,
    ).open()


# ---------------------------------------------------------------------------
# Le mode par défaut
# ---------------------------------------------------------------------------


def test_realtime_is_the_shipped_default() -> None:
    """C'est le mode du travail courant : une analyse qui suit la vidéo."""
    assert config.REALTIME.enabled is True


def test_the_warm_up_window_is_fifteen_seconds() -> None:
    """Le temps laissé au chargement du modèle avant de mesurer le rythme."""
    assert config.REALTIME.warmup_s == 15.0


def test_a_live_source_paces_itself_whatever_the_setting() -> None:
    """Un direct n'a pas le choix : les images arrivent qu'on les traite ou non.

    Accumuler du retard sur un flux revient à regarder le passé en croyant voir
    le présent.
    """
    direct = VideoSource(0, kind=SourceKind.WEBCAM, realtime=EXHAUSTIF)

    assert direct.paces_itself


def test_a_file_paces_itself_only_in_realtime_mode() -> None:
    """Un fichier attend : c'est le réglage qui décide."""
    assert VideoSource("v.mp4", realtime=TEMPS_REEL).paces_itself
    assert not VideoSource("v.mp4", realtime=EXHAUSTIF).paces_itself


# ---------------------------------------------------------------------------
# La fenêtre d'initialisation
# ---------------------------------------------------------------------------


def test_the_source_reports_that_it_is_warming_up(clock) -> None:
    """L'interface a besoin de le savoir pour afficher son écran de chargement."""
    source = _source(_Capture())

    assert source.is_warming_up
    assert source.warmup_remaining == pytest.approx(15.0)

    clock.advance(16.0)
    assert not source.is_warming_up
    assert source.warmup_remaining == 0.0


def test_nothing_is_dropped_during_the_warm_up(clock) -> None:
    """Le cas que la fenêtre existe pour couvrir.

    Quatorze secondes de retard pendant le chargement du modèle : sans la
    fenêtre, ce serait 350 images jetées d'un coup, et l'analyse commencerait en
    sautant le début de la vidéo.
    """
    capture = _Capture()
    source = _source(capture)

    source.read()
    clock.advance(14.0)
    seconde = source.read()

    assert seconde.dropped == 0
    assert capture.grabs == 0


def test_an_exhaustive_source_has_no_warm_up_window() -> None:
    """Rien à calibrer quand rien n'est jamais écarté."""
    source = _source(_Capture(), realtime=EXHAUSTIF)

    assert not source.is_warming_up


# ---------------------------------------------------------------------------
# Le rattrapage
# ---------------------------------------------------------------------------


def test_a_slow_machine_catches_up_after_the_warm_up(clock) -> None:
    """Le cœur du mode : quatre secondes de retard, cent images rattrapées.

    À 25 images/s, quatre secondes de traitement pendant lesquelles une seule
    image a été lue laissent cent images de retard.
    """
    capture = _Capture()
    source = _source(capture)

    clock.advance(16.0)  # sortie de la fenêtre d'initialisation
    source.read()  # pose le repère de rythme
    clock.advance(4.0)  # traitement très lent
    rattrapage = source.read()

    assert rattrapage.dropped > 90
    assert capture.grabs == rattrapage.dropped


def test_a_machine_that_keeps_up_drops_nothing(clock) -> None:
    """Une machine assez rapide ne doit rien perdre.

    40 ms de traitement sur une vidéo à 25 images/s : c'est exactement le rythme.
    """
    capture = _Capture()
    source = _source(capture)
    clock.advance(16.0)
    source.read()

    for _ in range(20):
        clock.advance(0.04)
        frame = source.read()
        assert frame.dropped == 0

    assert capture.grabs == 0


def test_a_small_hiccup_is_tolerated(clock) -> None:
    """Une tolérance nulle ferait sauter une image au moindre à-coup système."""
    capture = _Capture()
    source = _source(capture)
    clock.advance(16.0)
    source.read()

    clock.advance(0.3)  # sous `max_lag_s` = 0.4
    assert source.read().dropped == 0
    assert capture.grabs == 0


def test_catching_up_is_capped(clock) -> None:
    """Après un long décrochage, on rattrape progressivement.

    La machine s'est mise en veille, ou un autre logiciel a saturé le
    processeur : sauter dix mille images d'un coup bloquerait la boucle.
    """
    reglages = config.RealtimeConfig(enabled=True, warmup_s=15.0, max_dropped_frames=30)
    capture = _Capture()
    source = _source(capture, realtime=reglages)

    clock.advance(16.0)
    source.read()
    clock.advance(600.0)  # dix minutes de décrochage
    rattrapage = source.read()

    assert rattrapage.dropped == 30


def test_unrecovered_lag_is_reported(clock) -> None:
    """Un retard qui persiste n'est pas un à-coup : l'opérateur doit le savoir."""
    reglages = config.RealtimeConfig(enabled=True, warmup_s=15.0, max_dropped_frames=30)
    source = _source(_Capture(), realtime=reglages)

    clock.advance(16.0)
    source.read()
    clock.advance(60.0)
    source.read()

    assert source.lag_s > 1.0


def test_recovery_is_progressive_and_the_lag_eventually_clears(clock) -> None:
    """Le rattrapage est **progressif**, et le compteur finit par retomber à zéro.

    Après une minute de décrochage, le plafond `max_dropped_frames` interdit de
    tout reprendre d'un coup : chaque lecture regagne au plus 120 images, soit
    4,8 s à 25 images/s. Une dizaine de lectures suffisent donc, à condition que
    la machine tienne enfin le rythme.

    Le compteur de retard ne doit pas rester bloqué sur un incident passé : une
    valeur figée ferait croire à une machine durablement dépassée.
    """
    source = _source(_Capture())
    clock.advance(16.0)
    source.read()
    clock.advance(60.0)
    source.read()

    assert source.lag_s > 1.0, "Une minute de décrochage doit se voir."

    for _ in range(40):
        clock.advance(0.04)  # la machine tient désormais le rythme
        source.read()
        if source.lag_s == 0.0:
            break

    assert source.lag_s == 0.0, "Le retard n'a jamais été résorbé."


def test_an_exhaustive_source_never_catches_up(clock) -> None:
    """Le mode exhaustif examine tout, même si l'analyse dure plus que la vidéo."""
    capture = _Capture()
    source = _source(capture, realtime=EXHAUSTIF)

    source.read()
    clock.advance(300.0)
    seconde = source.read()

    assert seconde.dropped == 0
    assert capture.grabs == 0


# ---------------------------------------------------------------------------
# Ce que le rattrapage ne casse pas
# ---------------------------------------------------------------------------


def test_dropped_frames_do_not_falsify_the_clock(clock) -> None:
    """**Le point qui rend ce mode acceptable.**

    `grab()` fait avancer le compteur d'images comme `read()`. Sur un fichier,
    `video_time = index / fps` : une image lue après cent images sautées porte
    donc bien l'horodatage de la 101ᵉ, pas celui de la 2ᵉ.

    Sans cette propriété, toutes les durées seraient sous-estimées et aucune
    règle temporelle ne serait fiable — c'est exactement le défaut que le mode
    direct avait corrigé.
    """
    source = _source(_Capture())

    clock.advance(16.0)
    premiere = source.read()
    clock.advance(4.0)
    seconde = source.read()

    assert premiere.video_time == pytest.approx(0.0, abs=0.05)
    # Quatre secondes de temps réel se sont écoulées : le temps vidéo doit avoir
    # avancé d'autant, à la tolérance de rattrapage près.
    assert seconde.video_time == pytest.approx(4.0, abs=0.5)


def test_the_video_time_keeps_matching_the_wall_clock(clock) -> None:
    """Sur la durée, l'analyse ne dérive pas : c'est la définition du temps réel."""
    source = _source(_Capture())
    clock.advance(16.0)
    source.read()

    derniere = None
    for _ in range(30):
        clock.advance(0.5)  # deux fois trop lent pour du 25 im/s
        derniere = source.read()

    # 15 s de temps réel écoulées après le repère : le temps vidéo doit suivre.
    assert derniere.video_time == pytest.approx(15.0, abs=1.0)


def test_the_frame_index_counts_dropped_frames_too(clock) -> None:
    """L'index reste un index dans la vidéo, pas un compteur d'images traitées."""
    source = _source(_Capture())

    clock.advance(16.0)
    source.read()
    clock.advance(4.0)
    seconde = source.read()

    assert seconde.index > 90


def test_a_live_source_keeps_its_own_catch_up_logic(clock) -> None:
    """Deux problèmes distincts, deux calculs distincts.

    En direct, ce qui est en retard est ce qui est arrivé depuis la dernière
    lecture — le tampon du pilote se vide. Sur un fichier, le retard est cumulé
    et se rattrape en avançant dans le fichier. Les confondre serait une fausse
    économie.
    """
    capture = _Capture()
    source = _source(capture, kind=SourceKind.WEBCAM)

    source.read()
    clock.advance(0.4)
    seconde = source.read()

    # Le direct rattrape immédiatement, sans attendre la fin de l'initialisation.
    assert seconde.dropped == 9


# ---------------------------------------------------------------------------
# Le mode exhaustif reste accessible
# ---------------------------------------------------------------------------


def test_the_exhaustive_mode_is_reachable_from_the_interface() -> None:
    """C'est le mode à employer pour produire une pièce contradictoire."""
    import app

    source = app.open_source("video.mp4", {"realtime": False})

    assert source.realtime.enabled is False


def test_the_interface_defaults_to_realtime() -> None:
    """Un réglage absent ne doit pas silencieusement changer de mode."""
    import app

    assert app.open_source("video.mp4", {}).realtime.enabled is True


def test_the_interface_choice_overrides_the_configuration() -> None:
    """Le reste des réglages de cadence est conservé."""
    import app

    source = app.open_source("video.mp4", {"realtime": False})

    assert source.realtime.warmup_s == config.REALTIME.warmup_s
    assert source.realtime.max_lag_s == config.REALTIME.max_lag_s


# ---------------------------------------------------------------------------
# Validation des réglages
# ---------------------------------------------------------------------------


def test_a_zero_tolerance_is_refused(monkeypatch) -> None:
    """Une tolérance nulle ferait écarter une image au moindre à-coup."""
    monkeypatch.setattr(config, "REALTIME", config.RealtimeConfig(max_lag_s=0.0))

    with pytest.raises(ValueError, match="max_lag_s"):
        config.validate()


def test_a_negative_warm_up_is_refused(monkeypatch) -> None:
    """Une fenêtre négative n'a pas de sens."""
    monkeypatch.setattr(config, "REALTIME", config.RealtimeConfig(warmup_s=-1.0))

    with pytest.raises(ValueError, match="warmup_s"):
        config.validate()


def test_a_cap_of_zero_is_refused(monkeypatch) -> None:
    """Sans rattrapage possible, l'analyse dérive sans fin."""
    monkeypatch.setattr(config, "REALTIME", config.RealtimeConfig(max_dropped_frames=0))

    with pytest.raises(ValueError, match="max_dropped_frames"):
        config.validate()


# ---------------------------------------------------------------------------
# L'écran d'initialisation
# ---------------------------------------------------------------------------


def test_the_boot_screen_is_a_single_self_contained_block() -> None:
    """Même exigence que l'écrin du logo, et le risque est ici maximal.

    Ce bloc est précisément celui que la première image de la vidéo remplace.
    Ouvrir une balise dans un appel à `st.markdown` et la fermer dans un autre
    laisse le DOM réel diverger de celui que React croit gérer, et le
    remplacement échoue sur `NotFoundError: removeChild`.
    """
    import app

    html = app._boot_screen_html("INITIALISATION", "Chargement...")

    assert html.startswith('<div class="sr-boot">')
    assert html.endswith("</div>")
    assert html.count("<div") == html.count("</div>")
    assert html.count("<p") == html.count("</p>")


def test_the_boot_screen_escapes_what_it_displays() -> None:
    """Les textes viennent du code aujourd'hui ; l'échappement est une habitude.

    Du HTML injecté sans échappement finit toujours par coûter cher.
    """
    import app

    html = app._boot_screen_html("<script>alert(1)</script>", "a & b")

    assert "<script>" not in html
    assert "a &amp; b" in html


def test_the_boot_screen_works_without_a_hint() -> None:
    """La ligne d'explication est optionnelle et ne laisse pas de balise vide."""
    import app

    html = app._boot_screen_html("INITIALISATION")

    assert "sr-boot-hint" not in html
    assert html.count("<div") == html.count("</div>")


def test_the_dots_are_animated_in_css_not_in_python() -> None:
    """Réécrire le bloc dix fois par seconde ferait clignoter la page.

    Cela multiplierait aussi les manipulations du DOM pendant la phase la plus
    fragile de la session — celle où le bloc va être remplacé par la vidéo.
    """
    import app

    html = app._boot_screen_html("INITIALISATION")
    source = (__import__("pathlib").Path(app.__file__)).read_text(encoding="utf-8")

    assert "....." not in html, "Les points doivent venir du CSS, pas du fragment."
    assert "sr-dots" in source


def test_the_boot_screen_is_shown_before_the_model_loads() -> None:
    """Le chargement des poids est l'opération la plus longue de la session.

    Sans signe de vie à ce moment-là, l'application paraît figée.
    """
    import ast
    from pathlib import Path

    source = Path(app_module().__file__).read_text(encoding="utf-8")
    arbre = ast.parse(source)
    principal = next(
        n for n in arbre.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    corps = ast.unparse(principal)

    assert corps.index("_boot_screen_html") < corps.index("build_pipeline")


def test_the_boot_screen_is_cleared_whatever_happens() -> None:
    """Un modèle introuvable laisserait sinon l'écran tourner sur l'erreur."""
    import ast
    from pathlib import Path

    source = Path(app_module().__file__).read_text(encoding="utf-8")
    arbre = ast.parse(source)
    principal = next(
        n for n in arbre.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )

    for noeud in ast.walk(principal):
        if isinstance(noeud, ast.Try) and noeud.finalbody:
            if "boot.empty()" in ast.unparse(ast.Module(body=noeud.finalbody, type_ignores=[])):
                return
    raise AssertionError("L'écran d'initialisation n'est pas vidé dans un `finally`.")


def app_module():
    """Module `app`, importé paresseusement pour ne pas ralentir la collecte."""
    import app

    return app
