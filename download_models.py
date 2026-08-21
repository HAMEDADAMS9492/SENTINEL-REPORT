"""Téléchargement préalable des poids YOLOv8 utilisés par SentinelReport.

Ce script est un **outil d'installation**, pas un module de l'application : il
s'exécute une seule fois, à la main, avant le premier lancement de l'interface.

    python download_models.py

Pourquoi séparer le téléchargement de l'utilisation
---------------------------------------------------
`YOLO("yolov8m.pt")` sait télécharger ses poids tout seul. C'est pratique pour un
notebook, mais mauvais pour une application : le téléchargement (50 Mo) se
produirait alors *pendant* que l'utilisateur attend, à un moment imprévisible,
sans barre de progression, et échouerait sans réseau. En le sortant ici, on
obtient une frontière nette : après ce script, `models/` est un dossier de
ressources locales, et l'application ne fait plus jamais d'accès réseau.

Le script est **idempotent** : un fichier déjà présent n'est pas retéléchargé,
on peut donc le relancer sans crainte (par exemple après avoir ajouté un modèle
dans `config.AVAILABLE_MODELS`).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import config


def _locate_downloaded_file(model, filename: str) -> Path | None:
    """Retrouve le fichier de poids qu'Ultralytics vient de télécharger.

    Ultralytics choisit lui-même où déposer le fichier (répertoire courant ou
    dossier de poids de sa configuration, selon les versions). On interroge donc
    l'objet modèle plutôt que de supposer un emplacement, avec un repli sur le
    répertoire courant si l'attribut interne venait à changer de nom.

    Args:
        model: Instance `ultralytics.YOLO` fraîchement construite.
        filename: Nom du fichier de poids (ex. `"yolov8s.pt"`).

    Returns:
        Le chemin du fichier téléchargé, ou `None` s'il reste introuvable.
    """
    candidates = [
        getattr(model, "ckpt_path", None),
        getattr(getattr(model, "model", None), "pt_path", None),
        Path.cwd() / filename,
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.is_file():
                return path
    return None


def download_model(label: str, destination: Path) -> bool:
    """Télécharge un modèle s'il n'est pas déjà présent dans `models/`.

    Args:
        label: Nom lisible du modèle, tel qu'affiché dans l'interface.
        destination: Chemin local attendu (`models/yolov8n.pt`, ...).

    Returns:
        True si le fichier est disponible à la fin de l'appel (déjà présent ou
        téléchargé avec succès), False en cas d'échec.
    """
    if destination.is_file():
        size_mb = destination.stat().st_size / 1_048_576
        print(f"  [OK]      {label:<20} deja present ({size_mb:.1f} Mo) -> {destination}")
        return True

    filename = destination.name
    print(f"  [...]     {label:<20} telechargement de {filename} en cours...")

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "  [ERREUR]  Le paquet 'ultralytics' est introuvable.\n"
            "            Installez les dependances : pip install -r requirements.txt"
        )
        return False

    try:
        # Passer le nom nu (et non le chemin) déclenche le téléchargement
        # automatique depuis les « assets » d'Ultralytics.
        model = YOLO(filename)
    except Exception as exc:
        print(f"  [ERREUR]  {label:<20} telechargement impossible : {exc}")
        return False

    source = _locate_downloaded_file(model, filename)
    if source is None:
        print(
            f"  [ERREUR]  {label:<20} fichier telecharge introuvable sur le disque. "
            f"Copiez manuellement {filename} dans {destination.parent}."
        )
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        # Ultralytics dépose souvent le fichier dans le répertoire courant : on le
        # *déplace* pour ne pas laisser un doublon de plusieurs dizaines de Mo à
        # la racine du projet. S'il vient du cache d'Ultralytics (ailleurs sur le
        # disque), on se contente de le copier pour ne pas vider ce cache.
        if source.parent.resolve() == Path.cwd().resolve():
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)

    size_mb = destination.stat().st_size / 1_048_576
    print(f"  [TELECHARGE] {label:<17} {size_mb:.1f} Mo -> {destination}")
    return True


def main() -> int:
    """Télécharge tous les modèles déclarés dans `config.AVAILABLE_MODELS`.

    Returns:
        Code de sortie du processus : 0 si tous les modèles sont disponibles,
        1 si au moins un a échoué (utile en intégration continue).
    """
    print("SentinelReport - preparation des modeles de detection")
    print(f"Dossier de destination : {config.MODELS_DIR}\n")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        label: download_model(label, path)
        for label, path in config.AVAILABLE_MODELS.items()
    }

    failed = [label for label, ok in results.items() if not ok]
    print()
    if failed:
        print(f"Termine avec des erreurs. Modeles manquants : {', '.join(failed)}")
        return 1

    print(f"Termine : {len(results)} modele(s) disponible(s) en local.")
    print("Vous pouvez lancer l'application : streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
