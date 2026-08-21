"""Tests de `sentinel.source` — sans caméra, sans réseau, sans fichier vidéo.

`VideoSource` accepte une fabrique de capture en paramètre, ce qui permet de
substituer une doublure à `cv2.VideoCapture`. On peut ainsi éprouver ce qui est
justement impossible à reproduire à la demande : une caméra injoignable, un flux
qui se coupe puis revient, un traitement trop lent qui prend du retard sur le
direct.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from sentinel.exceptions import SourceDisconnectedError, VideoSourceError
from sentinel.source import Frame, SourceKind, VideoSource, detect_kind

IMAGE = np.zeros((48, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Doublure de capture
# ---------------------------------------------------------------------------


class _FakeCapture:
    """Imite le minimum de `cv2.VideoCapture` utilisé par `VideoSource`."""

    def __init__(
        self,
        target,
        *,
        opened: bool = True,
        frames: int | None = None,
        fps: float = 25.0,
        fail_after: int | None = None,
        recover: bool = False,
    ) -> None:
        self.target = target
        self._opened = opened
        self._remaining = frames
        self._fps = fps
        self._fail_after = fail_after
        self._recover = recover
        self.reads = 0
        self.grabs = 0
        self.released = False
        self.properties: dict[int, float] = {}

    # -- API OpenCV --------------------------------------------------------

    def isOpened(self) -> bool:  # noqa: N802 - nom imposé par OpenCV
        return self._opened

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._remaining or 0)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        self.properties[prop] = value
        return True

    def grab(self) -> bool:
        self.grabs += 1
        return True

    def read(self):
        self.reads += 1
        if self._fail_after is not None and self.reads > self._fail_after:
            return False, None
        if self._remaining is not None:
            if self._remaining <= 0:
                return False, None
            self._remaining -= 1
        return True, IMAGE.copy()

    def release(self) -> None:
        self.released = True


def _factory(**kwargs):
    """Fabrique de captures identiques, mémorisant celles qui ont été créées."""
    created: list[_FakeCapture] = []

    def build(target):
        capture = _FakeCapture(target, **kwargs)
        created.append(capture)
        return capture

    build.created = created  # type: ignore[attr-defined]
    return build


# ---------------------------------------------------------------------------
# Reconnaissance du type de source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target, expected",
    [
        (0, SourceKind.WEBCAM),
        ("1", SourceKind.WEBCAM),
        ("rtsp://192.168.1.10:554/stream1", SourceKind.STREAM),
        ("http://camera.local/video.mjpg", SourceKind.STREAM),
        ("data/videos/quai.mp4", SourceKind.FILE),
        (r"C:\videos\quai.mp4", SourceKind.FILE),
    ],
)
def test_source_kind_is_deduced_from_the_target(target, expected) -> None:
    """Une URL, un index et un chemin se distinguent sans que l'utilisateur ait à le dire."""
    assert detect_kind(target) is expected


def test_only_files_are_finite() -> None:
    """C'est la terminaison, pas le protocole, qui change le comportement."""
    assert SourceKind.FILE.is_live is False
    assert SourceKind.WEBCAM.is_live is True
    assert SourceKind.STREAM.is_live is True


# ---------------------------------------------------------------------------
# Ouverture et erreurs
# ---------------------------------------------------------------------------


def test_unreachable_stream_raises_a_business_error() -> None:
    """Une caméra injoignable produit un message actionnable, pas une trace."""
    source = VideoSource.from_stream(
        "rtsp://192.168.1.10:554/stream1", capture_factory=_factory(opened=False)
    )

    with pytest.raises(VideoSourceError, match="Flux injoignable"):
        source.open()


def test_unavailable_webcam_explains_the_usual_cause() -> None:
    """La cause la plus fréquente est une autre application qui occupe la caméra."""
    source = VideoSource.from_webcam(0, capture_factory=_factory(opened=False))

    with pytest.raises(VideoSourceError, match="autre application"):
        source.open()


def test_credentials_never_leak_into_messages() -> None:
    """Une URL RTSP porte souvent un mot de passe : il ne doit ni s'afficher ni se journaliser."""
    url = "rtsp://admin:s3cr3t@192.168.1.10:554/stream1"
    source = VideoSource.from_stream(url, capture_factory=_factory(opened=False))

    with pytest.raises(VideoSourceError) as raised:
        source.open()

    assert "s3cr3t" not in str(raised.value)
    assert "admin" not in str(raised.value)
    assert "192.168.1.10" in str(raised.value)


def test_reading_before_opening_is_rejected() -> None:
    """Lire une source non ouverte est une erreur de programmation, signalée comme telle."""
    source = VideoSource.from_file("video.mp4", capture_factory=_factory())

    with pytest.raises(VideoSourceError, match="open"):
        source.read()


def test_context_manager_always_releases_the_capture() -> None:
    """Sous Windows, une capture non libérée verrouille le fichier ou laisse la webcam allumée."""
    factory = _factory(frames=2)

    with VideoSource.from_file("video.mp4", capture_factory=factory) as source:
        assert source.is_open

    assert factory.created[0].released is True


# ---------------------------------------------------------------------------
# Source finie : le fichier
# ---------------------------------------------------------------------------


def test_file_source_ends_on_its_own() -> None:
    """Un fichier se termine : l'itération s'arrête sans intervention extérieure."""
    with VideoSource.from_file("video.mp4", capture_factory=_factory(frames=5)) as source:
        frames = list(source.frames())

    assert len(frames) == 5
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]


def test_file_video_time_is_reconstructed_from_the_frame_index() -> None:
    """Sur un fichier, le temps métier ne dépend pas de la vitesse de la machine.

    C'est ce qui rend l'analyse reproductible : deux exécutions, mêmes durées,
    donc mêmes incidents.
    """
    with VideoSource.from_file(
        "video.mp4", capture_factory=_factory(frames=4, fps=20.0)
    ) as source:
        frames = list(source.frames())

    assert [round(f.video_time, 3) for f in frames] == [0.0, 0.05, 0.10, 0.15]


def test_file_source_never_drops_frames() -> None:
    """Sur un fichier, aucune image n'est sautée : la fidélité prime sur la fraîcheur."""
    factory = _factory(frames=6)

    with VideoSource.from_file("video.mp4", capture_factory=factory) as source:
        frames = list(source.frames())

    assert all(f.dropped == 0 for f in frames)
    assert factory.created[0].grabs == 0


def test_progress_exists_only_for_finite_sources() -> None:
    """Un direct n'a pas d'avancement : il n'a qu'une durée."""
    with VideoSource.from_file("video.mp4", capture_factory=_factory(frames=10)) as fichier:
        fichier.read()
        assert 0.0 < fichier.progress() <= 1.0

    with VideoSource.from_webcam(0, capture_factory=_factory()) as direct:
        assert direct.progress() is None
        assert direct.total_frames == 0


def test_absurd_fps_falls_back_to_the_configuration() -> None:
    """Webcams et flux RTSP annoncent souvent 0 ou 1000 images par seconde."""
    with VideoSource.from_file(
        "video.mp4", capture_factory=_factory(frames=2, fps=0.0)
    ) as source:
        assert source.fps == config.VIDEO.default_fps


# ---------------------------------------------------------------------------
# Source infinie : le direct
# ---------------------------------------------------------------------------


def test_live_source_only_stops_when_asked() -> None:
    """Une caméra ne se termine jamais : l'arrêt vient forcément de l'extérieur."""
    seen: list[Frame] = []

    with VideoSource.from_webcam(0, capture_factory=_factory()) as source:
        for frame in source.frames(should_stop=lambda: len(seen) >= 7):
            seen.append(frame)

    assert len(seen) == 7


def test_stop_interrupts_the_iteration_immediately() -> None:
    """Le bouton « Arrêter » de l'interface passe par là."""
    with VideoSource.from_webcam(0, capture_factory=_factory()) as source:
        collected = []
        for frame in source.frames():
            collected.append(frame)
            if len(collected) == 3:
                source.stop()

    assert len(collected) == 3


class _Clock:
    """Horloge pilotée par le test, plus lisible qu'une suite de valeurs figées.

    Compter les appels à `monotonic()` rendrait les tests fragiles : la moindre
    réorganisation du code de lecture les casserait sans que le comportement ait
    changé. Ici, le test avance le temps explicitement.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock(monkeypatch) -> _Clock:
    """Remplace l'horloge monotone du module par une horloge contrôlée."""
    fake = _Clock()
    monkeypatch.setattr("sentinel.source.time.monotonic", fake)
    return fake


def test_live_video_time_follows_the_wall_clock(clock) -> None:
    """En direct, le temps métier est **observé**, pas reconstruit.

    Si le traitement prend du retard, des images sont sautées ; le compteur
    d'images cesse alors de mesurer le temps, et `index / fps` sous-estimerait
    toutes les durées. Un objet présent depuis 30 s paraîtrait n'en avoir que 12.
    """
    settings = config.SourceConfig(drop_late_frames=False)

    with VideoSource.from_webcam(
        0, capture_factory=_factory(), settings=settings
    ) as source:
        premiere = source.read()
        clock.advance(30.0)
        seconde = source.read()

    # Deux images seulement, mais 30 secondes écoulées : c'est l'horloge qui
    # fait foi, pas le compteur d'images.
    assert premiere.video_time == pytest.approx(0.0)
    assert seconde.video_time == pytest.approx(30.0)
    assert seconde.index == 1


def test_late_frames_are_discarded_to_stay_on_the_live_edge(clock) -> None:
    """Le piège classique du RTSP : le tampon se remplit et on regarde le passé.

    Traitement de 400 ms sur un flux à 25 images/s : dix images arrivent pendant
    ce temps, neuf doivent être écartées pour que la dixième — la plus récente —
    soit celle qu'on analyse.
    """
    factory = _factory(fps=25.0)

    with VideoSource.from_stream(
        "rtsp://camera/stream", capture_factory=factory
    ) as source:
        source.read()
        clock.advance(0.4)
        deuxieme = source.read()

    assert deuxieme.dropped == 9
    assert factory.created[0].grabs == 9


def test_dropping_is_capped_to_avoid_blocking_the_loop(clock) -> None:
    """Après un long décrochage, on ne saute pas des milliers d'images d'un coup."""
    settings = config.SourceConfig(max_dropped_frames=30)
    factory = _factory(fps=25.0)

    with VideoSource.from_stream(
        "rtsp://camera/stream", capture_factory=factory, settings=settings
    ) as source:
        source.read()
        clock.advance(600.0)  # dix minutes de décrochage
        rattrapage = source.read()

    assert rattrapage.dropped == 30


def test_file_playback_is_never_accelerated_by_dropping(clock) -> None:
    """Même avec un traitement lent, un fichier garde toutes ses images."""
    factory = _factory(frames=3)

    with VideoSource.from_file("video.mp4", capture_factory=factory) as source:
        frames = []
        for frame in source.frames():
            clock.advance(5.0)  # traitement très lent
            frames.append(frame)

    assert len(frames) == 3
    assert factory.created[0].grabs == 0
    assert all(f.dropped == 0 for f in frames)


# ---------------------------------------------------------------------------
# Reconnexion
# ---------------------------------------------------------------------------


def test_brief_outage_does_not_stop_the_surveillance(monkeypatch) -> None:
    """Une coupure réseau brève doit être absorbée, pas fatale."""
    monkeypatch.setattr("sentinel.source.time.sleep", lambda _: None)

    captures: list[_FakeCapture] = []

    def factory(target):
        # La première capture lâche après deux images, la suivante fonctionne.
        capture = _FakeCapture(target, fail_after=2 if not captures else None)
        captures.append(capture)
        return capture

    settings = config.SourceConfig(reconnect_attempts=3, reconnect_delay_s=0.0)

    with VideoSource.from_stream(
        "rtsp://camera/stream", capture_factory=factory, settings=settings
    ) as source:
        obtenues = [source.read() for _ in range(4)]

    assert all(frame is not None for frame in obtenues)
    assert len(captures) == 2, "Le flux aurait dû être rouvert une fois."
    assert captures[0].released is True


def test_permanent_outage_raises_a_dedicated_error(monkeypatch) -> None:
    """Après épuisement des tentatives, l'erreur dit quoi vérifier.

    Le type est distinct de `VideoSourceError` : une source qui n'a jamais
    démarré est une erreur de configuration, un flux perdu est un incident
    réseau — et l'analyse déjà produite reste valide.
    """
    monkeypatch.setattr("sentinel.source.time.sleep", lambda _: None)
    settings = config.SourceConfig(reconnect_attempts=2, reconnect_delay_s=0.0)

    def factory(target):
        return _FakeCapture(target, fail_after=1, opened=len(created) == 0)

    created: list[int] = []

    def counting_factory(target):
        capture = factory(target)
        created.append(1)
        return capture

    source = VideoSource.from_stream(
        "rtsp://camera/stream", capture_factory=counting_factory, settings=settings
    )
    source.open()
    source.read()  # première image : correcte

    with pytest.raises(SourceDisconnectedError, match="non rétabli"):
        source.read()


def test_reconnection_gives_up_when_stop_was_requested(monkeypatch) -> None:
    """Demander l'arrêt pendant une coupure doit terminer proprement, sans erreur."""
    monkeypatch.setattr("sentinel.source.time.sleep", lambda _: None)
    settings = config.SourceConfig(reconnect_attempts=5, reconnect_delay_s=0.0)

    source = VideoSource.from_stream(
        "rtsp://camera/stream",
        capture_factory=_factory(fail_after=1),
        settings=settings,
    )
    source.open()
    source.read()
    source.stop()

    assert source.read() is None


# ---------------------------------------------------------------------------
# Construction depuis la configuration
# ---------------------------------------------------------------------------


def test_source_can_be_built_from_the_configuration() -> None:
    """Le type de source et sa désignation vivent dans `config.SOURCE`."""
    settings = config.SourceConfig(
        kind=SourceKind.STREAM, stream_url="rtsp://camera/stream"
    )

    source = VideoSource.from_settings(settings, capture_factory=_factory())

    assert source.kind is SourceKind.STREAM
    assert source.is_live is True


@pytest.mark.parametrize(
    "settings, message",
    [
        (config.SourceConfig(kind=SourceKind.STREAM, stream_url=""), "URL"),
        (config.SourceConfig(kind=SourceKind.FILE, file_path=None), "fichier"),
    ],
)
def test_incomplete_configuration_is_reported_clearly(settings, message) -> None:
    """Une configuration incomplète est signalée avant toute tentative d'ouverture."""
    with pytest.raises(VideoSourceError, match=message):
        VideoSource.from_settings(settings)
