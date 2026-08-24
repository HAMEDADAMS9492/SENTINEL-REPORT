# SentinelReport — image d'exécution.
#
# Une image plutôt qu'un `pip install` sur la machine cible : le projet dépend
# d'OpenCV, qui réclame des bibliothèques système absentes des images Python
# minimales. Les diagnostiquer sur un hébergement distant coûte des heures ;
# les figer ici coûte quatre lignes.
#
# Cette image tourne telle quelle sur Render, Fly.io, Hugging Face Spaces,
# Cloud Run et n'importe quel hôte Docker. Elle ne tourne PAS sur Vercel, qui
# n'exécute que des fonctions sans état — voir DEPLOIEMENT.md.

FROM python:3.11-slim

# Dépendances système d'OpenCV. `libgl1` et `libglib2.0-0` sont les deux
# manquantes classiques : sans elles, `import cv2` échoue sur
# « libGL.so.1: cannot open shared object file », une erreur qui ne dit rien de
# sa cause. `ffmpeg` fournit les décodeurs vidéo au-delà des formats intégrés.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# `torch` en version CPU : la variante CUDA pèse plusieurs gigaoctets de plus
# pour un matériel que la plupart des hébergements n'ont pas. Installée avant le
# reste pour que la couche soit mise en cache indépendamment du code.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Les poids sont téléchargés **à la construction**, jamais au premier clic.
# Ultralytics télécharge tout seul un modèle manquant : sur un hébergement, cela
# se traduirait par une première analyse qui se fige plusieurs minutes, ou qui
# échoue si le réseau sortant est fermé. C'est exactement le comportement que ce
# projet refuse, et `download_models.py` existe pour cela.
RUN python download_models.py --only nano

# Sorties d'exécution hors du code applicatif : certains hébergements montent
# un volume, d'autres présentent un système de fichiers en lecture seule.
ENV SENTINEL_STATE_DIR=/data
RUN mkdir -p /data/evidence /data/reports

# Vérification à la construction : une configuration fautive ou une dépendance
# système manquante doit faire échouer le `docker build`, pas le premier
# utilisateur. `config.validate()` tourne à l'import.
RUN python -c "import cv2, config, app; print('image verifiee')"

EXPOSE 8501

# `--server.address=0.0.0.0` : sans lui, Streamlit n'écoute que la boucle locale
# et le conteneur paraît démarré mais injoignable.
# `$PORT` : Render, Fly et Cloud Run imposent le port par variable d'environnement.
ENV PORT=8501
CMD streamlit run app.py \
      --server.port=${PORT} \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false
