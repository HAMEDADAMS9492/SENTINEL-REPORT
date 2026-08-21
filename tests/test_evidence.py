"""Capture des preuves : écrire, conserver, et minimiser.

Pourquoi ce module a été extrait de `events.py`
-------------------------------------------------
Le moteur de règles faisait deux choses. Son docstring affirmait « le moteur ne
détecte rien et ne dessine rien », et cent vingt lignes plus bas il ouvrait des
fichiers, dessinait des rectangles et écrivait du JPEG.

Ce n'est pas un défaut d'esthétique : les deux responsabilités ont des raisons de
changer différentes. Le moteur change quand une règle métier évolue ; la capture
change quand la conservation, le format ou l'anonymisation évoluent. Les mélanger
obligeait à relire du code de règles pour ajouter un floutage.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import config
from sentinel.detection import Detection
from sentinel.events import Event, EventEngine
from sentinel.evidence import EvidenceWriter, blur_bystanders
from sentinel.tracker import TrackedObject

T0 = datetime(2026, 8, 21, 22, 15, 0)


def _obj(track_id: int = 1, *, box=(100.0, 100.0, 140.0, 200.0), class_name="person"):
    """Objet suivi minimal."""
    detection = Detection(0, class_name, 0.9, box, track_id)
    return TrackedObject(track_id, class_name, 0.0, 10.0, T0, T0, detection)


def _event(track_id: int = 1) -> Event:
    """Incident minimal."""
    return Event(
        event_id="SR-0001",
        event_type=config.EventType.INTRUSION,
        severity=config.Severity.HIGH,
        timestamp=T0,
        video_time=12.0,
        track_id=track_id,
        class_name="person",
        zone_name="Quai",
        duration_s=30.0,
        confidence=0.9,
        bbox=(100.0, 100.0, 140.0, 200.0),
    )


def _frame() -> np.ndarray:
    """Image de test, uniformément grise pour que le flou soit mesurable."""
    image = np.full((400, 400, 3), 128, dtype=np.uint8)
    # Un damier dans la région à flouter : une zone unie resterait unie après flou.
    image[200:300, 200:300] = np.indices((100, 100)).sum(axis=0)[:, :, None] % 2 * 255
    return image


def _writer(tmp_path: Path, **kwargs) -> EvidenceWriter:
    """Écrivain écrivant dans un dossier temporaire."""
    reglages = config.EvidenceConfig(directory=tmp_path, **kwargs)
    return EvidenceWriter(reglages)


# ---------------------------------------------------------------------------
# La séparation elle-même
# ---------------------------------------------------------------------------


def test_the_rules_engine_no_longer_writes_files() -> None:
    """`events.py` ne doit plus contenir d'écriture disque.

    Le vérifier structurellement évite que la capture ne revienne s'y loger à la
    faveur d'un remaniement — elle n'y casserait aucun test.
    """
    source = (Path(__file__).resolve().parents[1] / "sentinel" / "events.py").read_text(
        encoding="utf-8"
    )

    assert "cv2.imwrite" not in source
    assert "mkdir" not in source


def test_the_engine_delegates_to_an_evidence_writer() -> None:
    """Le moteur *utilise* un écrivain ; il n'en porte pas la logique."""
    from zone_doubles import FakeZones

    engine = EventEngine(FakeZones.restricted("Quai"))

    assert isinstance(engine._evidence, EvidenceWriter)


def test_a_failed_capture_does_not_cancel_the_incident(tmp_path: Path) -> None:
    """Un disque plein ne doit pas faire disparaître un incident.

    Le rapport existera, simplement sans image — ce qui est infiniment
    préférable à un incident perdu.
    """
    ecrivain = _writer(tmp_path)

    inchange = ecrivain.attach(_event(), None, _obj())

    assert inchange.evidence_path is None


# ---------------------------------------------------------------------------
# Écriture
# ---------------------------------------------------------------------------


def test_a_capture_is_written_and_attached(tmp_path: Path) -> None:
    """Le cas nominal : le fichier existe et l'incident le référence."""
    ecrivain = _writer(tmp_path)

    enrichi = ecrivain.attach(_event(), _frame(), _obj())

    assert enrichi.evidence_path is not None
    assert Path(enrichi.evidence_path).is_file()


def test_the_original_incident_is_never_mutated(tmp_path: Path) -> None:
    """`Event` est immuable : un incident publié ne change jamais."""
    ecrivain = _writer(tmp_path)
    origine = _event()

    ecrivain.attach(origine, _frame(), _obj())

    assert origine.evidence_path is None


def test_the_filename_carries_the_incident_context(tmp_path: Path) -> None:
    """Un dossier de preuves doit se lire sans ouvrir les fichiers."""
    ecrivain = _writer(tmp_path)

    enrichi = ecrivain.attach(_event(track_id=42), _frame(), _obj(42))

    nom = Path(enrichi.evidence_path).name
    assert "intrusion" in nom
    assert "id42" in nom


def test_cropping_produces_a_smaller_image(tmp_path: Path) -> None:
    """Le recadrage est optionnel et fonctionne."""
    complet = _writer(tmp_path / "a", crop_to_object=False)
    recadre = _writer(tmp_path / "b", crop_to_object=True, margin_px=5)

    grand = complet.attach(_event(), _frame(), _obj()).evidence_path
    petit = recadre.attach(_event(), _frame(), _obj()).evidence_path

    assert Path(grand).stat().st_size > Path(petit).stat().st_size


# ---------------------------------------------------------------------------
# Floutage — l'accroche
# ---------------------------------------------------------------------------


def test_blurring_is_off_by_default() -> None:
    """Le floutage dégrade une pièce destinée à être relue par un humain.

    C'est à l'exploitant de trancher, pas au logiciel de décider pour lui.
    """
    assert config.EVIDENCE.blur_bystanders is False


def test_the_subject_is_never_blurred() -> None:
    """L'objet en cause est la pièce même : le flouter viderait le rapport."""
    image = _frame()
    sujet = _obj(1, box=(200.0, 200.0, 300.0, 300.0))
    avant = image[200:300, 200:300].copy()

    blur_bystanders(image, sujet, [sujet])

    assert np.array_equal(image[200:300, 200:300], avant)


def test_a_bystander_is_blurred() -> None:
    """Le passant, lui, est traité."""
    image = _frame()
    sujet = _obj(1, box=(0.0, 0.0, 50.0, 50.0))
    passant = _obj(2, box=(200.0, 200.0, 300.0, 300.0))
    avant = image[200:300, 200:300].copy()

    blur_bystanders(image, sujet, [sujet, passant])

    assert not np.array_equal(image[200:300, 200:300], avant)


def test_only_the_configured_classes_are_blurred() -> None:
    """Flouter une valise n'a aucun intérêt et abîmerait la pièce."""
    image = _frame()
    sujet = _obj(1, box=(0.0, 0.0, 50.0, 50.0))
    valise = _obj(2, box=(200.0, 200.0, 300.0, 300.0), class_name="suitcase")
    avant = image[200:300, 200:300].copy()

    blur_bystanders(image, sujet, [sujet, valise])

    assert np.array_equal(image[200:300, 200:300], avant)


def test_a_degenerate_box_is_skipped() -> None:
    """Une boîte plate ne doit pas faire échouer la capture."""
    image = _frame()
    sujet = _obj(1, box=(0.0, 0.0, 50.0, 50.0))
    plat = _obj(2, box=(200.0, 200.0, 200.0, 200.0))

    blur_bystanders(image, sujet, [sujet, plat])  # ne doit pas lever


def test_an_injected_anonymiser_is_used(tmp_path: Path) -> None:
    """L'accroche est injectable : une autre politique se branche sans modifier ce module."""
    appels: list[int] = []

    def _noircir(image, sujet, autres):
        appels.append(sujet.track_id)
        return image

    ecrivain = EvidenceWriter(config.EvidenceConfig(directory=tmp_path), anonymiser=_noircir)
    ecrivain.attach(_event(), _frame(), _obj(7))

    assert appels == [7]


def test_anonymisation_happens_before_annotation(tmp_path: Path) -> None:
    """L'ordre compte : flouter après effacerait le rectangle qui désigne l'objet."""
    ordre: list[str] = []

    def _tracer(image, sujet, autres):
        ordre.append("flou")
        return image

    ecrivain = EvidenceWriter(
        config.EvidenceConfig(directory=tmp_path, annotate=True), anonymiser=_tracer
    )
    original = EvidenceWriter.annotate

    def _annoter(image, obj, event):
        ordre.append("annotation")
        return original(image, obj, event)

    ecrivain.annotate = staticmethod(_annoter)
    ecrivain.attach(_event(), _frame(), _obj())

    assert ordre == ["flou", "annotation"]


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------


def test_recent_captures_survive_the_purge(tmp_path: Path) -> None:
    """Une preuve d'hier reste une preuve."""
    ecrivain = _writer(tmp_path, retention_days=30)
    ecrivain.attach(_event(), _frame(), _obj())

    assert ecrivain.purge_expired() == 0
    assert list(tmp_path.iterdir())


def test_expired_captures_are_deleted(tmp_path: Path) -> None:
    """Ne jamais effacer transformerait un outil d'analyse en archive permanente.

    Une capture est une image de personnes prise sans leur accord, conservée pour
    un besoin précis et borné dans le temps.
    """
    ecrivain = _writer(tmp_path, retention_days=30)
    chemin = Path(ecrivain.attach(_event(), _frame(), _obj()).evidence_path)

    # 31 jours plus tard, du point de vue de la purge.
    supprimes = ecrivain.purge_expired(now=time.time() + 31 * 86_400)

    assert supprimes == 1
    assert not chemin.exists()


def test_indefinite_retention_is_possible_but_explicit(tmp_path: Path) -> None:
    """`None` conserve tout — un choix légitime, jamais le défaut."""
    ecrivain = _writer(tmp_path, retention_days=None)
    ecrivain.attach(_event(), _frame(), _obj())

    assert ecrivain.purge_expired(now=time.time() + 3650 * 86_400) == 0
    assert list(tmp_path.iterdir())


def test_the_shipped_configuration_bounds_retention() -> None:
    """Le défaut livré efface : c'est la position par défaut défendable."""
    assert config.EVIDENCE.retention_days is not None
    assert config.EVIDENCE.retention_days > 0


def test_a_missing_directory_does_not_break_startup(tmp_path: Path) -> None:
    """La purge tourne au démarrage : elle ne doit jamais empêcher de démarrer."""
    ecrivain = _writer(tmp_path / "inexistant", retention_days=30)

    assert ecrivain.purge_expired() == 0


def test_hidden_files_are_left_alone(tmp_path: Path) -> None:
    """`.gitkeep` maintient le dossier dans le dépôt : le purger le ferait disparaître."""
    garde = tmp_path / ".gitkeep"
    garde.touch()
    import os

    os.utime(garde, (0, 0))  # très ancien

    ecrivain = _writer(tmp_path, retention_days=1)

    assert ecrivain.purge_expired() == 0
    assert garde.exists()


def test_the_interface_purges_once_per_session() -> None:
    """Balayer le dossier à chaque clic serait aussi inutile que coûteux."""
    import ast

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    arbre = ast.parse(source)

    for noeud in arbre.body:
        if isinstance(noeud, ast.FunctionDef) and noeud.name == "purge_expired_evidence":
            decorateurs = [ast.unparse(d) for d in noeud.decorator_list]
            assert any("cache_resource" in d for d in decorateurs), decorateurs
            return
    raise AssertionError("purge_expired_evidence() est absente d'app.py.")
