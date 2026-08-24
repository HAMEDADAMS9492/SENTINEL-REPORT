# Déploiement

## En une phrase

**Vercel héberge la page vitrine ; l'application tourne dans un conteneur
ailleurs.** Ce n'est pas un contournement, c'est la seule répartition possible —
la raison est détaillée plus bas, et elle est bonne à savoir défendre.

---

## 1. Pourquoi Vercel ne peut pas héberger l'application

Quatre obstacles, dont aucun ne se contourne par de la configuration.

| Obstacle | Ce que Vercel offre | Ce que SentinelReport demande |
|---|---|---|
| **Modèle d'exécution** | fonctions sans état, réveillées par requête | un serveur durable, avec une connexion WebSocket maintenue pendant toute la session (Streamlit repose sur Tornado) |
| **Poids des dépendances** | 250 Mo décompressés par fonction | PyTorch + Ultralytics + OpenCV : plus de 2 Go |
| **Durée d'exécution** | 10 s en Hobby, 300 s au maximum | l'analyse d'une vidéo dure des minutes |
| **Écriture disque** | système de fichiers éphémère | `evidence/` et `reports/` doivent survivre à la session |

Le premier point suffirait à lui seul : une fonction sans état ne peut pas tenir
la connexion qu'une application Streamlit maintient d'un bout à l'autre d'une
session. Les trois autres confirment que le problème n'est pas de dimensionnement
mais de nature.

**Ce qui est donc déployé sur Vercel** : `public/index.html`, une page statique
autonome qui présente le projet, son architecture et ses décisions de conception,
et qui renvoie vers le code et vers l'instance en fonctionnement.

```bash
npm i -g vercel
vercel                # aperçu
vercel --prod         # production
```

`vercel.json` copie les fichiers de marque depuis `assets/` au moment de la
construction. Ils ne sont donc **pas dupliqués** dans le dépôt : le logo affiché
sur la page est le même fichier que celui de l'interface et des rapports PDF.

---

## 2. Où l'application tourne réellement

Le `Dockerfile` est la référence : les quatre cibles ci-dessous l'utilisent tel
quel, et il tourne aussi bien sur n'importe quel hôte Docker.

### Render — recommandé

Le plus proche d'un « Vercel pour Python durable » : un conteneur qui tourne en
continu, un disque persistant, un déploiement à chaque `git push`.

```bash
# render.yaml est déjà dans le dépôt : « New > Blueprint » sur render.com,
# pointer sur le dépôt, et c'est tout.
```

Le plan `starter` est délibéré : le plan gratuit s'endort après quinze minutes
d'inactivité — le réveil rechargerait les poids YOLO à froid — et manque de
mémoire pour PyTorch.

### Hugging Face Spaces — le plus simple pour une démonstration

Le `Dockerfile` fonctionne sans modification. Choisir « Docker » comme SDK à la
création du Space. Gratuit, avec une mise en veille acceptable pour une
démonstration : le contexte est celui d'un portfolio, pas d'une surveillance
réelle.

### Fly.io — le plus proche des utilisateurs

```bash
fly launch --dockerfile Dockerfile --no-deploy
fly volumes create sentinel_state --size 5
fly deploy
```

Ajouter le montage dans `fly.toml` :

```toml
[[mounts]]
  source = "sentinel_state"
  destination = "/data"
```

### Docker, en local ou sur un serveur

```bash
docker build -t sentinelreport .
docker run -p 8501:8501 -v sentinel_state:/data sentinelreport
```

---

## 3. Ce que le `Dockerfile` règle, et pourquoi

**Les bibliothèques système d'OpenCV.** `libgl1` et `libglib2.0-0` sont les deux
manquantes classiques des images Python minimales. Sans elles, `import cv2`
échoue sur `libGL.so.1: cannot open shared object file` — une erreur qui ne dit
rien de sa cause et qui coûte des heures à diagnostiquer sur une machine
distante.

**PyTorch en version CPU.** La variante CUDA pèse plusieurs gigaoctets de plus,
pour un matériel que la plupart des hébergements n'ont pas. Elle est installée
avant le reste pour que la couche Docker soit mise en cache indépendamment du
code applicatif.

**Les poids téléchargés à la construction.** Ultralytics télécharge tout seul un
modèle manquant, au premier appel. Sur un hébergement, cela se traduirait par une
première analyse qui se fige plusieurs minutes — ou qui échoue si le réseau
sortant est fermé. C'est exactement le comportement que ce projet refuse ;
`download_models.py --only nano` existe pour cela, et le `--only` évite
d'embarquer cinquante mégaoctets de poids qu'une démonstration n'utilisera pas.

**La vérification à la construction.** `python -c "import cv2, config, app"`
échoue le `docker build` plutôt que le premier utilisateur. `config.validate()`
tournant à l'import, une configuration fautive est refusée là aussi.

**`--server.address=0.0.0.0`.** Sans lui, Streamlit n'écoute que la boucle locale
et le conteneur paraît démarré mais reste injoignable — panne fréquente et
déroutante, puisque les journaux affichent « You can now view your app ».

---

## 4. Variables d'environnement

| Variable | Rôle | Défaut |
|---|---|---|
| `SENTINEL_STATE_DIR` | Racine des preuves et des rapports. À pointer vers un volume persistant, sinon la purge par `EVIDENCE.retention_days` n'a plus rien à purger et les rapports disparaissent au redéploiement. | le dossier du dépôt |
| `SENTINEL_LOG_LEVEL` | Verbosité (`DEBUG`, `INFO`, `WARNING`). | `INFO` |
| `PORT` | Port d'écoute. Imposé par Render, Fly et Cloud Run. | `8501` |
| `YOLO_CONFIG_DIR` | Où Ultralytics écrit ses réglages. Sans elle, un avertissement à chaque démarrage sur les systèmes de fichiers restreints. | dossier personnel |

---

## 5. Ce qui change une fois hébergé

**Pas de webcam.** Le serveur n'a pas de caméra : seul le téléversement de
fichier a du sens. L'option webcam reste affichée et échouera proprement sur un
message clair (`VideoSourceError`) plutôt que sur une trace Python — mais il vaut
mieux le savoir avant de cliquer.

**Un flux RTSP suppose que le serveur atteigne la caméra.** Une caméra sur un
réseau local n'est pas joignable depuis un hébergement public. C'est une des
raisons pour lesquelles le multi-caméra et l'ONVIF restent hors périmètre
(§ 8.3 du README).

**Le processeur est partagé.** Le mode temps réel — celui par défaut — écarte
d'autant plus d'images que la machine est lente. Les durées restent exactes, mais
l'exhaustivité de l'observation se dégrade. Pour une démonstration, préférer des
vidéos courtes ; pour une analyse sérieuse, le mode exhaustif sur une machine
dédiée.

**Le téléversement est plafonné à 500 Mo** (`.streamlit/config.toml`). Certains
hébergements imposent une limite plus basse sur la taille des requêtes ; la
vérifier avant de promettre l'analyse d'un fichier volumineux.

---

## 6. Ce qu'il ne faut pas déployer publiquement sans y réfléchir

Ce logiciel écrit sur disque des images de personnes filmées sans leur accord.
Trois points à trancher avant d'exposer une instance :

1. **La rétention.** `EVIDENCE.retention_days` vaut 30 jours et la purge tourne
   au démarrage. Sur un hébergement qui redémarre rarement, la purge s'exécute
   rarement : envisager une tâche planifiée.
2. **L'accès.** Le projet n'a **aucune authentification** — c'est documenté au
   § 7 du README. Une instance publique est lisible par quiconque a l'URL.
   Mettre l'hébergement derrière une authentification si des vidéos réelles y
   sont déposées.
3. **Le floutage.** `EVIDENCE.blur_bystanders` est désactivé par défaut. Sur une
   instance de démonstration accessible à des tiers, l'activer est le choix
   raisonnable.
