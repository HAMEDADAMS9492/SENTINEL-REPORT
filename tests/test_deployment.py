"""Configuration de déploiement — vérifier ce qui ne se voit qu'en production.

Une erreur de déploiement ne se manifeste pas sur la machine de développement :
elle apparaît sur l'hébergement, dix minutes après un `git push`, sous la forme
d'un conteneur qui démarre et reste injoignable. Ces tests attrapent les fautes
classiques avant.

Ils ne construisent aucune image — ce serait plusieurs gigaoctets et plusieurs
minutes. Ils lisent les fichiers de configuration et vérifient que les décisions
qui comptent y figurent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
DOCKERFILE = (RACINE / "Dockerfile").read_text(encoding="utf-8")
VERCEL = json.loads((RACINE / "vercel.json").read_text(encoding="utf-8"))
LANDING = (RACINE / "public" / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# L'image d'exécution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paquet", ["libgl1", "libglib2.0-0", "ffmpeg"])
def test_opencv_system_libraries_are_installed(paquet: str) -> None:
    """Sans elles, `import cv2` échoue sur une erreur qui ne dit rien de sa cause.

    `libGL.so.1: cannot open shared object file` est la panne la plus fréquente
    d'OpenCV en conteneur, et la plus déroutante : rien dans le message ne
    suggère qu'il manque un paquet système.
    """
    assert paquet in DOCKERFILE


def test_the_image_listens_on_all_interfaces() -> None:
    """Sans `--server.address=0.0.0.0`, le conteneur paraît démarré mais est injoignable.

    Streamlit journalise « You can now view your app » tout en n'écoutant que la
    boucle locale : la panne est silencieuse côté serveur et totale côté client.
    """
    assert "--server.address=0.0.0.0" in DOCKERFILE


def test_the_image_honours_the_port_variable() -> None:
    """Render, Fly et Cloud Run imposent le port par variable d'environnement."""
    assert "${PORT}" in DOCKERFILE


def test_the_weights_are_baked_in_not_downloaded_at_runtime() -> None:
    """Ultralytics télécharge tout seul un modèle manquant, au premier appel.

    Sur un hébergement, cela se traduirait par une première analyse qui se fige
    plusieurs minutes — ou qui échoue si le réseau sortant est fermé. C'est le
    comportement que `download_models.py` existe pour empêcher.
    """
    assert "download_models.py" in DOCKERFILE


def test_only_the_needed_weights_are_embedded() -> None:
    """`--only nano` évite d'embarquer des poids qu'une démonstration n'ouvrira pas."""
    assert "--only" in DOCKERFILE


def test_torch_is_installed_in_its_cpu_flavour() -> None:
    """La variante CUDA pèse des gigaoctets pour un matériel que l'hôte n'a pas."""
    assert "download.pytorch.org/whl/cpu" in DOCKERFILE


def test_the_build_fails_rather_than_the_first_user() -> None:
    """Une dépendance manquante doit casser `docker build`, pas la première analyse."""
    assert 'python -c "import cv2, config, app' in DOCKERFILE


def test_runtime_outputs_live_outside_the_application_code() -> None:
    """Certains hébergements présentent un système de fichiers en lecture seule."""
    assert "SENTINEL_STATE_DIR" in DOCKERFILE


# ---------------------------------------------------------------------------
# Le dossier d'état, surchargeable
# ---------------------------------------------------------------------------


def _config_dans_un_sous_processus(tmp_path: Path, expression: str) -> str:
    """Évalue une expression de `config` dans un interpréteur neuf.

    Pourquoi un sous-processus plutôt que `importlib.reload`
    --------------------------------------------------------
    `config` est lu **à l'import** par tous les modules du paquet, qui gardent
    une référence directe aux objets qu'il expose. Le recharger au milieu de la
    suite ferait cohabiter deux configurations : `sentinel.zones` continuerait de
    pointer vers l'ancienne, les tests suivants sur une nouvelle. Le symptôme
    n'est pas un échec franc, c'est une suite qui se met à dépendre de son ordre
    d'exécution.

    Un interpréteur neuf est la seule façon honnête de tester une variable
    d'environnement lue au chargement.

    Args:
        tmp_path: Dossier d'état à imposer.
        expression: Expression Python évaluée après `import config`.

    Returns:
        La valeur affichée par le sous-processus.
    """
    environnement = {**os.environ, "SENTINEL_STATE_DIR": str(tmp_path)}
    resultat = subprocess.run(
        [sys.executable, "-c", f"import config; print({expression})"],
        cwd=RACINE,
        env=environnement,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert resultat.returncode == 0, resultat.stderr
    return resultat.stdout.strip()


def test_the_state_directory_can_be_relocated(tmp_path: Path) -> None:
    """`SENTINEL_STATE_DIR` déplace preuves et rapports sur un volume monté.

    Sans cela, les deux vivraient dans le dossier applicatif — éphémère sur la
    plupart des hébergements, donc effacé à chaque redéploiement.
    """
    sortie = _config_dans_un_sous_processus(
        tmp_path, "config.EVIDENCE_DIR, config.REPORTS_DIR, config.EVIDENCE_DIR.is_dir()"
    )

    assert str(tmp_path) in sortie
    assert "evidence" in sortie and "reports" in sortie
    assert "True" in sortie, "Le dossier doit être créé à l'import."


def test_the_weights_stay_with_the_repository(tmp_path: Path) -> None:
    """Les poids sont livrés avec l'image, pas écrits sur le volume d'état.

    Les déplacer avec l'état obligerait à les retélécharger à chaque nouveau
    volume — exactement ce que le téléchargement à la construction évite.
    """
    sortie = _config_dans_un_sous_processus(
        tmp_path, "config.MODELS_DIR == config.BASE_DIR / 'models'"
    )

    assert sortie == "True"


def test_the_default_state_directory_is_the_repository() -> None:
    """Sans variable, rien ne change pour un développement local."""
    import config

    assert config.EVIDENCE_DIR == config.BASE_DIR / "evidence"


# ---------------------------------------------------------------------------
# La page vitrine
# ---------------------------------------------------------------------------


def test_vercel_publishes_the_static_directory_only() -> None:
    """Vercel sert la vitrine ; l'application tourne dans un conteneur ailleurs."""
    assert VERCEL["outputDirectory"] == "public"


def test_the_brand_files_are_not_duplicated_in_the_repository() -> None:
    """Le logo affiché sur la page est **le même fichier** que celui de l'interface.

    Le dupliquer dans `public/` garantirait qu'un jour les deux divergent : la
    construction Vercel les copie depuis `assets/` à la place.
    """
    assert "cp assets/*.svg" in VERCEL["buildCommand"]
    assert not (RACINE / "public" / "assets").exists(), (
        "Les fichiers de marque sont copiés à la construction, pas versionnés deux fois."
    )


def test_the_landing_page_is_self_contained() -> None:
    """Aucune requête vers un tiers : ni police distante, ni script externe.

    Une page vitrine qui charge une police Google se met à dépendre de la
    disponibilité — et de la politique de confidentialité — d'un tiers.
    """
    for interdit in ("http://", "cdn.", "fonts.googleapis", "<script"):
        assert interdit not in LANDING, f"Ressource externe : {interdit}"


def test_the_landing_page_states_the_limits() -> None:
    """La vitrine ne doit pas promettre plus que le logiciel ne tient.

    Un portfolio qui tait ses angles morts est un portfolio qu'on ne peut pas
    défendre en entretien.
    """
    for mention in ("BROUILLON", "arme à feu", "n'identifie personne"):
        assert mention in LANDING, mention


def test_the_landing_page_explains_why_it_is_static() -> None:
    """La question se posera : autant y répondre sur la page elle-même."""
    assert "WebSocket" in LANDING


def test_the_landing_page_declares_a_favicon() -> None:
    """Le favicon vient des fichiers de marque, comme le reste."""
    assert "/assets/favicon.svg" in LANDING


# ---------------------------------------------------------------------------
# Ce qui ne doit pas partir dans l'image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", [".venv/", "models/*.pt", "evidence/*", ".env"])
def test_heavy_or_sensitive_paths_are_excluded_from_the_build(motif: str) -> None:
    """Le contexte de construction ne doit contenir ni gigaoctets ni secrets."""
    ignore = (RACINE / ".dockerignore").read_text(encoding="utf-8")

    assert motif in ignore


def test_the_documentation_names_a_working_target() -> None:
    """Dire que Vercel ne convient pas sans dire ce qui convient serait inutile."""
    doc = (RACINE / "DEPLOIEMENT.md").read_text(encoding="utf-8")

    assert "Render" in doc
    assert "Hugging Face" in doc
    assert "Fly.io" in doc
