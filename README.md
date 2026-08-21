# SentinelReport

Détection d'incidents de sécurité par vision par ordinateur, et rédaction
automatique d'un brouillon de rapport d'incident horodaté avec image de preuve.

> **État : ÉTAPE 9 / 9 — le cœur fonctionnel est complet.** La chaîne vidéo →
> détection → suivi → zones → règles → rapport PDF/CSV tourne de bout en bout,
> couverte par 366 tests, validés sur Python 3.11 / Streamlit 1.44 et Python 3.14 / Streamlit 1.54. Voir « Avancement » et « Feuille de route » plus bas.

---

## 1. Cas d'usage

Un agent de sécurité surveillant un campus ou un site industriel fait face à deux
problèmes concrets :

1. **Il ne peut pas regarder 12 écrans en continu.** Les incidents pertinents sont
   rares et noyés dans des heures de vidéo sans intérêt.
2. **Rédiger un rapport d'incident prend du temps** et se fait souvent après coup,
   de mémoire, avec des horodatages approximatifs.

SentinelReport analyse un flux vidéo (fichier ou webcam) et répond aux deux :

- il signale uniquement les situations qui correspondent à une **règle explicite**
  — intrusion en zone restreinte, présence prolongée, objet abandonné, présence
  hors horaires ;
- pour chacune, il produit un **brouillon de rapport** horodaté à la seconde,
  accompagné de la capture d'écran correspondante.

**Ce que le logiciel ne fait pas, volontairement :** il ne décide pas s'il y a
faute, il n'identifie personne, il ne déclenche aucune alarme automatique. Il
produit un *brouillon à valider par un opérateur humain*. C'est une limite
assumée et affichée dans chaque rapport généré.

---

## 2. Architecture

### 2.1 Principe

Une classe = une responsabilité. Chaque module doit pouvoir être testé, remplacé
ou expliqué isolément.

```
SENTINEL REPORT/
├── config.py              # Réglages centralisés (SEUL endroit avec des valeurs métier)
├── app.py                 # Interface Streamlit — orchestration et affichage uniquement
├── requirements.txt
├── README.md
│
├── sentinel/              # Paquet applicatif
│   ├── __init__.py
│   ├── exceptions.py      # Hiérarchie d'erreurs métier (SentinelError et filles)
│   ├── detection.py       # dataclass Detection : le contrat entre les modules
│   ├── source.py          # VideoSource   — fichier / webcam / flux RTSP unifiés
│   ├── detector.py        # Detector      — YOLOv8 : frame -> détections
│   ├── tracker.py         # Tracker       — mémoire temporelle des objets suivis
│   ├── zones.py           # ZoneManager   — zones polygonales, appartenance, dessin
│   ├── events.py          # EventEngine   — règles + score de priorité -> Event
│   ├── timeline.py        # Timeline      — faits marquants par tranche de temps
│   ├── report.py          # ReportGenerator — Event -> texte + PDF
│   └── session_report.py  # Rapport de session à deux niveaux
│
├── evidence/              # Captures horodatées (générées, non versionnées)
├── reports/               # Rapports PDF/CSV exportés
├── data/videos/           # Vidéos de test
├── assets/                # Identité visuelle (logos, favicon)
├── models/                # Poids YOLO (déposés par download_models.py)
└── tests/                 # 366 tests, sans modèle ni vidéo : logique métier,
                           # invariants d'architecture, hygiène des dépendances
```

### 2.2 Pourquoi un paquet `sentinel/` plutôt que des fichiers à la racine

`tracker.py`, `events.py` et `report.py` sont des noms très courants ; les
regrouper dans un paquet évite toute collision avec un module tiers et rend les
imports explicites (`from sentinel.tracker import Tracker`). `config.py` et
`app.py` restent à la racine : ce sont respectivement le fichier qu'on édite le
plus souvent et le point d'entrée.

### 2.3 Rôle de chaque fichier, en une phrase

| Fichier | Rôle |
|---|---|
| `config.py` | Contient toutes les valeurs de réglage — aucune autre ligne du projet ne code un seuil en dur. |
| `sentinel/exceptions.py` | Définit des erreurs métier lisibles, pour que l'interface affiche un message clair plutôt qu'une trace Ultralytics. |
| `sentinel/detection.py` | Définit `Detection`, la structure de données qui circule entre tous les modules. |
| `sentinel/source.py` | Fournit les images horodatées, qu'elles viennent d'un fichier, d'une webcam ou d'un flux RTSP — le reste du pipeline ignore leur origine. |
| `sentinel/detector.py` | Charge YOLOv8 et transforme une frame en liste de détections. |
| `sentinel/tracker.py` | Se souvient de chaque objet suivi : depuis quand il est là, où il est passé, dans quelle zone et depuis combien de temps. |
| `sentinel/zones.py` | Répond à « cet objet est-il dans cette zone ? » et dessine les zones. |
| `sentinel/events.py` | Applique les règles de la configuration, produit des `Event` structurés et calcule leur score de priorité. |
| `sentinel/timeline.py` | Observe le déroulé et retient les faits marquants, tranche par tranche de temps vidéo. |
| `sentinel/report.py` | Transforme un `Event` en rapport français, exportable en PDF. |
| `sentinel/session_report.py` | Assemble le rapport de session : incidents triés par priorité, puis chronologie. |
| `app.py` | Assemble le tout dans une interface Streamlit et affiche les résultats. |

### 2.4 Flux de données — de la frame au rapport

```
   Fichier · webcam · flux RTSP
        │
        ▼
┌───────────────────┐
│ VideoSource       │  ouverture, cadence, reconnexion, deux horodatages
│ .frames()         │
└───────────────────┘
        │  Frame (image BGR + video_time + wall_time)
┌───────────────────┐
│  Detector.track() │  YOLOv8 + ByteTrack
└───────────────────┘
        │  list[Detection]  (boîte, classe, confiance, track_id)
        ▼
┌───────────────────┐
│ Tracker.update()  │  mémoire temporelle par track_id
└───────────────────┘
        │  list[TrackedObject]  (+ âge, historique, chronos de zone)
        ▼
┌───────────────────┐
│ ZoneManager       │  test point-dans-polygone sur le point d'appui au sol
│ .zones_for()      │
└───────────────────┘
        │  {objet -> zones occupées}     → TrackedObject.update_zones()
        ▼
┌───────────────────┐
│ EventEngine       │  règles : classe + zone + durée + anti-rebond
│ .evaluate()       │
└───────────────────┘
        │  list[Event]  (+ capture de preuve écrite dans /evidence)
        ▼
┌───────────────────┐        ┌───────────────────┐
│ ReportGenerator   │        │ Timeline.observe()│  transitions : entrées,
│  gabarit → PDF    │        │                   │  sorties, incidents
└───────────────────┘        └───────────────────┘
        │                            │
        ▼                            ▼
   Rapport d'incident        ┌───────────────────┐
   + image de preuve         │ SessionReport     │  1. incidents par priorité
                             │  Generator        │  2. chronologie par tranches
                             └───────────────────┘
```

`Timeline` observe **les objets retenus par le tracker**, pas ceux visibles sur
la frame : la rétention absorbe déjà les occlusions courtes, et lui passer la
liste de la frame produirait une fausse sortie suivie d'une fausse entrée chaque
fois qu'un objet passe derrière un poteau.

Le sens de lecture est important : **l'information ne remonte jamais**. Le
détecteur ignore l'existence des zones, les zones ignorent les événements, les
événements ignorent le format du rapport. Chaque étage enrichit la donnée et la
passe au suivant.

### 2.5 Deux horloges, et pourquoi

- **Temps vidéo** (`frame_index / fps`) : pilote **toutes** les décisions métier.
  Une vidéo analysée en accéléré ou au ralenti produit exactement les mêmes
  incidents — l'analyse est donc reproductible, ce qui est indispensable pour un
  document à valeur de rapport.
- **Heure réelle** (`datetime`) : sert uniquement à dater le rapport et à
  évaluer la règle « hors horaires ».

---

## 3. Algorithmes utilisés

Cette section explique les quatre briques algorithmiques du projet. Elle est
écrite pour être comprise sans connaissance préalable en vision par ordinateur,
tout en employant les termes techniques exacts.

### 3.0 Sources finies et sources infinies (`source.py`)

Le pipeline accepte trois origines d'images — fichier, webcam locale, flux réseau
RTSP/HTTP — derrière une abstraction unique, `VideoSource`. `Detector` et
`Tracker` reçoivent une image et un temps ; d'où elle vient ne les regarde pas.

**La distinction qui compte n'est pas le protocole, c'est la terminaison.**

| | Fichier | Webcam / RTSP |
|---|---|---|
| Fin | naturelle, longueur connue | **jamais** — l'arrêt vient de l'extérieur |
| Rejouable | oui, à l'identique | non, ce qui n'est pas traité est perdu |
| Images sautées | aucune | oui, volontairement |
| Horloge métier | `frame_index / fps` | temps écoulé à l'horloge |

**Pourquoi l'heure réelle devient l'horloge naturelle en direct.** Sur un
fichier, `frame_index / fps` est la bonne mesure : elle rend l'analyse
indépendante de la vitesse de la machine, donc reproductible. En direct, cette
formule devient **fausse** dès qu'on saute des images — et il faut en sauter. Si
le traitement d'une image prend 140 ms alors que la caméra en produit une toutes
les 40 ms, trois images arrivent pendant qu'on travaille ; les écarter fait
diverger le compteur d'images du temps qui passe. Un objet présent depuis
30 secondes paraîtrait n'en avoir que 12, et aucune règle de durée ne serait
fiable.

En direct, le temps vidéo est donc **observé** à l'horloge plutôt que
*reconstruit* depuis un compteur. Ce n'est pas un renoncement au principe des
deux horloges : c'est son application correcte à une source qui, par nature, ne
peut pas être rejouée.

**Stratégie de cadence.** Accumuler les images en retard est le piège classique
du RTSP : le tampon du pilote se remplit et l'opérateur finit par regarder une
scène vieille de plusieurs minutes en croyant voir le direct. Deux mesures se
cumulent : `CAP_PROP_BUFFERSIZE = 1` demande au pilote de ne garder que la
dernière image (tous les backends ne l'honorent pas) ; et avant chaque lecture on
estime le nombre d'images arrivées pendant le traitement précédent
(`temps écoulé × fps`) pour les écarter avec `grab()`, qui récupère l'image sans
la décoder. La règle est donc : **en direct on préfère perdre des images plutôt
que du temps** ; sur un fichier l'arbitrage s'inverse.

**Robustesse réseau.** Une coupure brève ne doit pas arrêter une surveillance :
la source se rouvre jusqu'à `reconnect_attempts` fois avant d'abandonner, et
lève alors `SourceDisconnectedError` — un type distinct de `VideoSourceError`,
parce que la conduite à tenir diffère : une source qui n'a jamais démarré est une
erreur de configuration à corriger, un flux perdu en cours de route est un
incident réseau, et l'analyse déjà produite reste valide. Les URL RTSP portant
souvent des identifiants, ni les journaux ni les messages d'erreur ne recopient
le mot de passe.

> `app.py` consomme `VideoSource.frames()` : c'est le seul chemin de lecture
> du projet. Un test structurel (`tests/test_app_source.py`) vérifie qu'aucun
> `cv2.VideoCapture` ne réapparaît dans l'interface — la régression serait
> silencieuse, puisque l'analyse d'un fichier continuerait de marcher et que
> seul le direct deviendrait faux.

### 3.1 YOLO — détection d'objets en une passe (*one-stage*)

**Le problème.** Sur une image, trouver *où* sont les objets et *ce qu'ils sont*.

**Les approches historiques (*two-stage*, comme R-CNN)** procédaient en deux
temps : d'abord proposer quelques milliers de régions potentiellement
intéressantes, puis classer chacune séparément. C'est précis mais lent — on fait
tourner un classifieur des milliers de fois par image.

**L'idée de YOLO** (*You Only Look Once*) est de tout faire en **une seule
passe** du réseau de neurones. Concrètement :

1. L'image est redimensionnée à une taille fixe (ici 640×640, `config.MODEL.image_size`)
   et passée dans un réseau convolutif.
2. Le réseau produit une **grille** de cellules. Chaque cellule est responsable
   de la portion d'image qui lui correspond et prédit directement :
   - les coordonnées d'une ou plusieurs boîtes englobantes,
   - un score d'objectness (« y a-t-il quelque chose ici ? »),
   - une distribution de probabilité sur les 80 classes du jeu de données COCO
     (personne, voiture, sac à dos…).
3. Comme plusieurs cellules voisines détectent souvent le même objet, on obtient
   des boîtes en double. On les filtre par **suppression des non-maxima (NMS)** :
   on garde la boîte la plus confiante, on supprime toutes celles qui la
   recouvrent au-delà d'un seuil d'**IoU** (*Intersection over Union* : aire de
   l'intersection divisée par aire de l'union de deux boîtes, entre 0 et 1). Ce
   seuil est `config.MODEL.iou`.

**Le compromis.** Une seule passe = temps d'inférence constant et prévisible,
d'où le temps réel. En contrepartie, YOLO est historiquement un peu moins précis
que les approches en deux temps sur les très petits objets.

**Pourquoi `yolov8n.pt`.** Le « n » signifie *nano* : environ 3,2 millions de
paramètres, contre ~68 millions pour la variante `x`. Il tourne à plusieurs
images par seconde sur un simple CPU. Pour un projet de portfolio qui doit
pouvoir être démontré sur n'importe quel portable, c'est le bon choix ; le modèle
est interchangeable en une ligne de `config.py`.

**Deux paramètres à savoir défendre :**
- `confidence` (0.35) : sous ce score, la détection est ignorée. Le baisser
  augmente le rappel et les faux positifs ; le monter fait l'inverse.
- `iou` (0.50) : seuil de la NMS. Le baisser supprime plus agressivement les
  doublons, au risque de fusionner deux objets réellement proches.

### 3.2 ByteTrack — suivi multi-objets et association

**Le problème.** YOLO analyse chaque frame indépendamment. Sur la frame 100 il
voit « une personne », sur la frame 101 il voit « une personne ». Rien ne dit que
c'est *la même*. Or toutes les règles de SentinelReport sont temporelles :
« présent depuis 3 secondes », « immobile depuis 30 secondes ». **Sans identité
persistante, aucune durée n'est mesurable.**

**Le suivi multi-objets (MOT)** consiste à attribuer à chaque objet un
identifiant stable d'une frame à l'autre. Le schéma classique, dit
*tracking-by-detection*, enchaîne trois opérations à chaque frame :

1. **Prédiction.** Pour chaque piste existante, on prédit où l'objet *devrait*
   se trouver, à l'aide d'un **filtre de Kalman** — un estimateur qui modélise la
   position et la vitesse de l'objet et projette son état à l'instant suivant.
2. **Association.** On met en correspondance les prédictions et les détections
   réelles de la frame. Le coût d'appariement est fondé sur l'IoU entre boîte
   prédite et boîte détectée, et le couplage optimal est résolu par l'**algorithme
   hongrois** (méthode classique d'affectation à coût minimal).
3. **Mise à jour.** Les pistes appariées sont corrigées avec la mesure réelle,
   les détections orphelines créent de nouvelles pistes, et les pistes sans
   détection sont conservées « perdues » pendant quelques frames avant d'être
   supprimées — c'est ce qui permet de traverser une occlusion courte.

**L'apport spécifique de ByteTrack** tient en une idée simple et efficace. Les
trackers antérieurs jetaient les détections de faible confiance, considérées
comme du bruit. Or une détection de faible confiance correspond souvent à un
objet réel **partiellement masqué**. ByteTrack fait donc l'association en **deux
passes** :

- **passe 1** : associer les pistes aux détections de **forte** confiance ;
- **passe 2** : tenter d'associer les pistes restées orphelines aux détections
  de **faible** confiance.

Résultat : un objet momentanément occulté conserve son identifiant au lieu de
disparaître puis de réapparaître sous un nouveau numéro.

**Pourquoi cela remplace avantageusement une déduplication par proximité des
centres** (l'approche de mon prototype initial, qui comparait les centres des
boîtes avec une tolérance en pixels) :

| | Proximité des centres | ByteTrack |
|---|---|---|
| Identité persistante | ✗ non | ✓ oui, `track_id` stable |
| Occlusion courte | ✗ nouvel objet, chronomètre remis à zéro | ✓ piste conservée puis réassociée |
| Croisement de deux personnes | ✗ échange d'identité fréquent | ✓ le mouvement prédit lève l'ambiguïté |
| Mesure d'une durée | ✗ impossible | ✓ c'est le fondement du projet |

Ce n'est donc pas une optimisation : sans identité persistante, la fonctionnalité
principale du logiciel est irréalisable.

**Répartition dans le code.** Ultralytics implémente ByteTrack à l'intérieur de
l'objet modèle : c'est `model.track()` qui attribue les identifiants. L'association
reste donc dans `Detector`, qui possède le modèle. `Tracker` construit par-dessus
la **mémoire temporelle** — première et dernière apparition, historique des
positions, chronomètre par zone — qui est notre logique métier et dont dépend
`events.py`.

### 3.3 Appartenance à un polygone

**Le problème.** Une zone de surveillance n'est presque jamais un rectangle : un
quai de chargement vu en perspective est un quadrilatère quelconque. Il faut
donc tester l'appartenance d'un point à un **polygone arbitraire**.

**L'algorithme du lancer de rayon** (*ray casting*) répond à cette question de
façon élégante : depuis le point testé, on trace une demi-droite dans une
direction quelconque et on compte combien de côtés du polygone elle traverse.
**Nombre impair → le point est à l'intérieur ; nombre pair → à l'extérieur.**
L'intuition : chaque traversée fait passer de dehors à dedans ou l'inverse ; en
partant de l'infini (forcément dehors), la parité indique le côté.

En pratique on utilise **`cv2.pointPolygonTest`**, qui implémente exactement ce
test. Le choix mérite d'être justifié, car la conception initiale prévoyait
`supervision.PolygonZone` (Roboflow) : mesure faite, cette dépendance tire ~30 Mo
de paquets — dont PyAV, un décodeur vidéo complet — pour un service qu'OpenCV,
déjà obligatoire pour la lecture vidéo, rend avec une API stable depuis dix ans.
L'argument de la vectorisation ne tient pas à cette échelle : quelques dizaines
de tests par frame représentent quelques microsecondes, face aux ~140 ms
d'inférence YOLO qui les précèdent. Optimiser le maillon 10 000 fois moins cher
que le goulot d'étranglement n'a pas de sens.

La dépendance a été **retirée du projet**, pas seulement écartée du code : la
laisser déclarée dans `requirements.txt` faisait payer les 30 Mo sans rien
rendre en échange, et rendait ce paragraphe faux en pratique.
`tests/test_dependencies.py` verrouille le refus.

**Deux décisions de conception à savoir justifier :**

1. **Quel point tester ?** Pas le centre de la boîte, mais son **point d'appui au
   sol** — le milieu du bord inférieur (`Detection.anchor`). Pour une caméra en
   plongée, le centre de la boîte d'une personne debout se situe à mi-hauteur du
   corps et peut tomber hors de la zone alors que ses pieds y sont clairement.
   C'est réglable via `config.GEOMETRY.anchor`.
2. **Coordonnées normalisées.** Les polygones de `config.ZONES` sont exprimés en
   fractions de l'image (0.0–1.0), pas en pixels. La même définition de zone
   fonctionne donc en 720p, en 1080p et sur webcam ; `ZoneManager.initialize()`
   les convertit en pixels quand la première frame arrive.

### 3.4 La mémoire temporelle (`tracker.py`)

ByteTrack fournit des identifiants ; il ne fournit aucune **histoire**. C'est le
rôle de `TrackedObject` : pour chaque identifiant, accumuler ce qu'une frame
isolée ne peut pas dire. Quatre décisions y méritent d'être justifiées.

**1. Une fenêtre glissante, pas un journal complet.** L'historique des positions
est un `deque` purgé à `config.TRACKING.history_seconds` (30 s). Une vidéo de
10 minutes à 25 fps avec 10 objets, c'est 150 000 positions si l'on garde tout —
pour un besoin qui ne remonte jamais au-delà de la dernière minute.

**2. Déplacement maximal, jamais cumulé.** Pour juger de l'immobilité d'un objet,
on mesure l'écart **maximal à la position la plus ancienne** de la fenêtre. La
distance cumulée serait un piège : une boîte de détection « tremble » de 2-3 px
d'une frame à l'autre même sur un objet parfaitement immobile, ce qui accumule
des centaines de pixels en quelques secondes et rendrait la règle « objet
abandonné » incapable de déclencher.

**3. L'immobilité exige de couvrir la fenêtre.** `is_stationary()` refuse de
répondre `True` si l'objet n'a pas été observé pendant **toute** la durée
demandée. Sans ce contrôle, un sac qui vient d'apparaître serait trivialement
« immobile depuis 30 secondes » — et signalé abandonné à l'instant où on le voit.

**4. La rétention se compte en secondes, pas en frames.** Un objet non revu reste
en mémoire `config.TRACKING.max_age_s` (3 s) avant d'être purgé. Cette tolérance
absorbe les occlusions courtes : quelqu'un qui passe derrière un poteau garde son
identifiant, donc son chronomètre d'intrusion. Raisonner en secondes plutôt qu'en
nombre de frames manquées conserve le même comportement que la source tourne à 25
ou à 10 images par seconde.

**5. Hystérésis sur les frontières de zone.** Entrer dans une zone demande
`min_overlap_frames` frames consécutives ; en sortir en demande
`exit_tolerance_frames`, volontairement plus élevé. Cette asymétrie corrige le
défaut le plus coûteux du suivi géométrique : une boîte de détection tremble de
quelques pixels sur la frontière, et sans tolérance **chaque oscillation
remettrait le chronomètre d'intrusion à zéro**. Une présence de 50 secondes
entrecoupée d'une frame perdue toutes les cinq frames ne franchirait jamais un
seuil de 3 secondes. Perdre brièvement un objet est plus fréquent que le
détecter à tort, d'où la sortie plus tolérante que l'entrée.

Le chronomètre part du **premier contact**, pas de l'instant de confirmation :
les frames d'observation ont bien été passées dans la zone, les décompter
introduirait un biais systématique sur toutes les durées rapportées.

**6. Vote de classe majoritaire.** YOLO peut changer d'avis d'une frame à
l'autre sur un même objet suivi — « truck » puis « bus ». Retenir la classe de
la première frame serait le pire choix : c'est celle où l'objet est le plus
petit, le plus flou ou le plus partiellement visible. `TrackedObject` accumule
donc les votes et retient la classe majoritaire, l'ordre alphabétique tranchant
les égalités pour que deux analyses de la même vidéo produisent le même rapport.

S'y ajoutent deux garde-fous contre les fausses alertes, exploités par
`events.py` : `min_hits` (un objet vu moins de 3 fois n'est pas « confirmé » et
n'atteint jamais les règles) et `last_event_time` par type d'événement, qui
implémente le délai de garde décrit ci-dessous.

### 3.5 Seuils relatifs — corriger la perspective sans calibration

Un seuil exprimé en pixels n'a pas le même sens partout dans l'image. Un
déplacement de 25 px, c'est un frémissement pour un sac au premier plan et une
traversée complète pour le même sac au fond du champ. Un rayon de 150 px autour
d'un bagage couvre environ un mètre devant la caméra et huit mètres au loin.
Deux règles sur quatre en dépendaient directement.

**La correction employée** rapporte chaque distance à la **hauteur apparente de
l'objet** (`TrackedObject.scale_px`). Un objet deux fois plus éloigné apparaît
deux fois plus petit : le rapport `distance / hauteur` est donc à peu près
invariant par la profondeur. La règle « objet abandonné » s'énonce alors :

```
immobile          : déplacement < 0,35 × sa propre hauteur
sans propriétaire : aucune personne à moins de 3 × sa hauteur
```

C'est une approximation — elle suppose des objets de taille physique comparable —
mais elle supprime l'essentiel de l'erreur, **sans calibration ni homographie**,
donc sans rien demander à l'utilisateur. Les seuils en pixels restent déclarés
dans `config.EVENT_RULES` et servent de repli si le ratio n'est pas renseigné.

### 3.6 Le moteur d'événements temporel

Ce n'est pas un algorithme publié, mais c'est la partie proprement « métier » du
projet — celle qui transforme de la géométrie en information de sécurité. Le
principe est une **machine à états par objet suivi**.

**Une règle** (`config.EventRule`) est un ensemble de conditions :

```
classe de l'objet  +  zone occupée  +  durée écoulée  [+ immobilité]  →  Event
```

**Le mécanisme, frame par frame**, pour chaque objet suivi :

1. `ZoneManager` détermine les zones qu'il occupe.
2. `TrackedObject.update_zones()` compare avec les zones de la frame précédente.
   Une zone nouvellement occupée déclenche un **chronomètre** (`zone_entry_time`) ;
   une zone quittée l'efface.
3. `EventEngine` évalue chaque règle : la classe correspond-elle ? la zone est-elle
   concernée ? `dwell_time()` dépasse-t-il `min_duration_s` ?
4. Si oui, un `Event` est produit, une capture est écrite dans `/evidence/`, et
   l'instant du déclenchement est mémorisé.

**Trois garde-fous, chacun résolvant un problème réel :**

- **Le seuil de durée** (`min_duration_s`) distingue un incident d'un simple
  passage. Traverser une zone restreinte en marchant n'est pas une intrusion ; y
  rester trois secondes l'est.
- **L'anti-rebond** (`cooldown_s`) : sans lui, une personne restant deux minutes
  dans une zone déclencherait un événement *par frame*, soit environ 3 000
  rapports pour un seul incident. Chaque objet mémorise la date de son dernier
  événement par type et ne peut pas redéclencher avant expiration du délai.
- **La confirmation de piste** (`min_hits`) : un objet vu sur une seule frame est
  presque toujours un faux positif de YOLO. On attend quelques apparitions avant
  de le considérer comme réel.

**Cas particulier de l'objet abandonné** (étape 8) : la durée ne suffit pas, il
faut aussi de l'**immobilité** et l'**absence de propriétaire**. L'immobilité est
mesurée comme l'écart maximal à la position la plus ancienne de la fenêtre
d'observation — et non comme la distance cumulée, qui gonflerait avec le bruit
de détection : une boîte qui « tremble » de 2 pixels par frame accumulerait des
centaines de pixels alors que l'objet ne bouge pas. L'absence de propriétaire est
testée en cherchant la personne suivie la plus proche dans un rayon donné.

### 3.7 Le score de priorité — ordonner sans juger

Quand trente incidents tombent pendant la nuit, l'opérateur a besoin de savoir
**par lequel commencer**. C'est le seul rôle du score : il n'ajoute aucune
information, il ordonne celle que les règles ont déjà établie.

**Ce qu'il n'est pas.** Pas de modèle appris, pas de détection d'anomalie, pas de
pondération découverte dans les données, pas de comparaison à des cas passés.
Chaque point provient d'un signal déjà mesuré ailleurs dans le pipeline :

| Signal | Origine | Contribution |
|---|---|---|
| Type de règle déclenchée | `EventEngine` | points de base (15 à 35) |
| Durée constatée | `TrackedObject.dwell_time()` | 10 pts/minute, plafonné à 30 |
| Immobilité | `TrackedObject.is_stationary()` | 8 pts/minute, plafonné à 20 |
| Cumul de règles sur l'objet | `TrackedObject.last_event_time` | 15 pts par règle, plafonné à 30 |
| Hors horaires | `config.SCHEDULE` + heure réelle | multiplicateur × 1,5 |

Le barème entier vit dans `config.ScoringConfig`. Le modifier ne demande aucune
ligne de logique — c'est une **convention**, discutable et ajustable, pas une
vérité.

**Trois décisions à savoir défendre :**

1. **Le cumul est le signal le plus fort.** Un objet qui a rôdé, puis pénétré en
   zone restreinte, puis abandonné un sac raconte une histoire qu'aucune des
   trois règles ne dit seule. Le bonus s'appuie sur `last_event_time`, qui
   n'existe que grâce à l'identité persistante de ByteTrack.
2. **Le hors-horaires est un multiplicateur, pas un bonus fixe.** Le même
   comportement est plus grave à 3 h du matin qu'à midi — *proportionnellement*.
   Un bonus fixe écraserait la hiérarchie entre une intrusion et un rôdage
   nocturnes.
3. **Tout est plafonné.** Sans plafond sur la durée, un objet oublié depuis une
   heure paraîtrait plus urgent qu'une intrusion en cours.

**Traçabilité.** Chaque `Event` porte un `PriorityScore` qui conserve le détail
terme à terme. La justification est lisible telle quelle :

```
Moyen (46 pts) : intrusion_zone_restreinte (30) + présence 3 s (+0)
                 + hors horaires (× 1.5)
```

L'addition se refait à la main, et tout écart entre deux exécutions s'expliquerait
par une ligne de `config.py`. C'est ce qui distingue une aide à la décision d'un
oracle : le score est **opposable**. Un opérateur qui conteste un classement voit
d'où viennent les points.

**Où vit le calcul.** Dans `events.py`, avec les autres appréciations métier —
jamais dans `report.py`, qui reste purement présentation : il trie et affiche des
scores déjà calculés. Le flux de données reste unidirectionnel.

### 3.8 Le rapport à deux niveaux

Le rapport d'incident isolé répond à « que s'est-il passé à 21 h 47 ? ». Le
rapport de **session** répond à la question d'un chef de poste en fin de
service : « que s'est-il passé pendant la surveillance, et par quoi dois-je
commencer ? ». D'où deux parties, et pas une de plus.

**Partie 1 — Incidents prioritaires.** Les incidents triés par score
décroissant, chacun avec son horodatage, son type, sa zone, la **justification
de son score** et sa preuve. C'est la liste d'action.

**Partie 2 — Chronologie.** Le déroulé par tranches de 30 s de temps vidéo
(`config.TIMELINE.slice_seconds`), ne retenant que les faits marquants. C'est la
mise en récit.

Les deux parties disent la même chose dans deux ordres différents, et c'est
délibéré : **trier par priorité fait perdre la causalité, trier par temps fait
perdre l'urgence.** Un sac abandonné se comprend en voyant qui l'a posé trois
tranches plus tôt ; savoir par où commencer demande l'ordre inverse.

**Ne dire que ce qui change.** Une chronologie qui répète « 3 personnes
présentes » toutes les 30 secondes pendant une heure ne se lit pas. Chaque
tranche ne retient que les **transitions** — arrivées, départs, incidents levés,
passages en priorité élevée — et une tranche sans transition n'est pas listée
du tout.

**Pourquoi un module `timeline.py` distinct.** Décider qu'une personne « entre
dans le champ » demande de comparer l'état présent à l'état précédent, donc une
observation continue. Un générateur de rapport reçoit des faits déjà établis ;
lui confier la chronologie l'obligerait à reconstituer après coup une histoire
qu'il n'a pas vue. Le flux reste unidirectionnel :

```
tracker  ->  timeline (observe et retient les transitions)  ->  report (met en forme)
events   ->  score de priorité                              ->  report (trie)
```

`session_report.py` ne calcule **aucun** score et n'établit **aucun** fait : un
test vérifie que le module ne contient ni `PriorityScore(`, ni `points_for`, ni
`level_for`. S'il se mettait à en produire, la même donnée serait calculée à
deux endroits — et divergerait tôt ou tard.

**Échantillonnage en temps vidéo.** Le découpage suit `video_time`, jamais
l'horloge de traitement : la même vidéo analysée sur un portable lent ou sur un
GPU produit exactement la même chronologie. En direct, `video_time` **est**
l'horloge murale (§ 3.0), donc les tranches correspondent à des minutes réelles
— le comportement attendu d'une surveillance continue.

**Mode fini et mode continu.** Rien dans la génération ne suppose que la session
est close : `Timeline` s'alimente au fil de l'eau et se consulte à tout moment.
Un rapport peut donc être exporté en pleine surveillance, puis un autre plus
tard, chacun reflétant l'état à son instant d'édition.

---

## 4. Installation

Prérequis : **Python 3.10 ou supérieur** (le code utilise la syntaxe `X | None`).

```bash
git clone <url-du-depot>
cd "SENTINEL REPORT"

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

python download_models.py     # une seule fois : dépose les poids dans models/
```

**Téléchargement des modèles.** `download_models.py` récupère les trois modèles
de `config.AVAILABLE_MODELS` (~78 Mo au total) et les place dans `models/`. Le
script est idempotent : un fichier déjà présent n'est pas retéléchargé.

Ce téléchargement est volontairement **séparé de l'application**. Ultralytics
sait charger `yolov8m.pt` à la volée, mais cela ferait attendre l'utilisateur
50 Mo en pleine analyse, sans progression visible, et échouerait sans réseau. En
l'isolant dans un script d'installation, `models/` devient un dossier de
ressources locales et le processus Streamlit n'a plus jamais besoin d'accès
Internet — ce que `load_model()` vérifie explicitement en refusant tout fichier
de poids absent.

| Modèle | Fichier | Paramètres | mAP50-95 | Poids |
|---|---|---|---|---|
| Rapide (nano) — défaut | `yolov8n.pt` | 3,2 M | ~37,3 | 6 Mo |
| Équilibré (small) | `yolov8s.pt` | 11,2 M | ~44,9 | 22 Mo |
| Précis (medium) | `yolov8m.pt` | 25,9 M | ~50,2 | 50 Mo |

Le choix se fait dans la barre latérale de l'interface. Le nano est le défaut
parce que les règles portent sur des **durées** : mieux vaut une chronologie
dense à 7 images/s qu'une détection plus fine à 1,7 image/s, d'autant que
ByteTrack a besoin d'un flux serré pour associer correctement les identifiants.

**GPU (optionnel).** L'installation par défaut de PyTorch est en version CPU, ce
qui suffit pour `yolov8n.pt`. Pour utiliser une carte NVIDIA, installer d'abord
PyTorch avec CUDA depuis [pytorch.org](https://pytorch.org), puis
`requirements.txt`. `config.MODEL.device = "auto"` détecte le matériel disponible
et retombe silencieusement sur le CPU s'il n'y a rien.

## 5. Utilisation

Dans l'ordre, depuis la racine du projet, environnement virtuel activé :

```bash
pip install -r requirements.txt   # 1. dépendances (une fois)
python download_models.py         # 2. poids YOLO dans models/ (une fois)
pytest -q                         # 3. vérification : 366 tests, < 5 s
streamlit run app.py              # 4. interface (complète à partir de l'étape 7)
```

**Vérifier l'installation sans interface**, utile pour diagnostiquer :

```bash
python -c "import cv2, ultralytics, streamlit, fpdf, lap; print('dependances OK')"
python -c "import config; print(config.missing_models() or 'modeles OK')"
python -c "import sys, streamlit; print(sys.executable, streamlit.__version__)"
```

La troisième commande vaut le détour : **c'est l'interpréteur qui exécutera
l'application**. Si plusieurs Python sont installés, `pip install` et
`streamlit run` peuvent viser des environnements différents — les dépendances
paraissent installées et l'application plante quand même. En cas de doute,
lancer explicitement `python -m streamlit run app.py` plutôt que `streamlit run
app.py` garantit que les deux commandes parlent du même environnement.

**Compatibilité Streamlit.** Le projet fonctionne de la **1.44 à la 1.54**. Ces
versions ont changé de convention pour « occuper toute la largeur » —
`use_container_width=True` avant la 1.49, `width="stretch"` ensuite, les deux
étant mutuellement incompatibles. `app.py` détecte la version installée une fois
à l'import (fonction `stretch()`) et emploie la bonne : passer le mauvais
paramètre lève `TypeError: '<=' not supported between instances of 'str' and
'int'` sur les versions anciennes, et inonde d'avertissements les récentes.

**Organisation de l'interface.** Elle sépare deux gestes différents :

| Barre latérale — *ce qui se règle* | Panneau principal — *ce qui se choisit* |
|---|---|
| Modèle (nano / small / medium) | Source : fichier vidéo ou webcam |
| Seuil de confiance | Catégories surveillées : personnes, véhicules, objets dangereux, sacs et bagages |
| Facteur de durées de déclenchement | Ajout d'un objet précis parmi les 80 classes du modèle |
| Cadence (une frame sur N, durée maximale) | |
| Affichage (confiance, identifiants de suivi) | |

Un opérateur ajuste les curseurs de gauche **pendant** une analyse ; il ne
touche à la colonne de droite qu'en préparant une session. Le facteur de durées
multiplie à la fois les seuils de déclenchement et les délais de garde de
`config.EVENT_RULES` — les deux ensemble, pour préserver l'invariant
`cooldown_s > min_duration_s` qui empêche un même objet de redéclencher en boucle.

Puis « Lancer l'analyse » : la vidéo annotée défile en direct, les incidents
s'accumulent dans le tableau, chacun avec son brouillon de rapport, sa preuve
visuelle et les exports PDF et CSV.

**Deux documents, deux questions.** Le rapport d'**incident** répond à « que
s'est-il passé à 21 h 47 ? » : un incident, son constat, ses éléments techniques,
son score de priorité et sa justification, sa preuve visuelle. Le rapport de
**session** répond à celle d'un chef de poste en fin de service : « que s'est-il
passé pendant la surveillance, et par quoi dois-je commencer ? » — les incidents
triés par priorité, puis la chronologie par tranches. Les deux s'exportent en PDF,
la chronologie aussi en CSV. Le rapport de session s'affiche **même sans
incident** : une surveillance calme est un résultat, et la chronologie le
documente.

**Webcam.** Un flux en direct n'a pas de fin : l'arrêt vient du garde-fou
« Durée maximale analysée » de la barre latérale. Le temps affiché sous la vidéo
indique alors « temps réel » et non « temps vidéo » — sur un fichier, `t` est
reconstruit depuis le numéro d'image et l'analyse est reproductible ; en direct,
`t` est l'heure écoulée. Savoir laquelle on lit importe avant de recopier une
durée dans un rapport.

**Thème.** Les couleurs de l'interface sont définies dans
`.streamlit/config.toml` et reprises **telles quelles** des fichiers SVG de
`assets/` : anthracite `#0F172A`, bleu acier `#0EA5E9`, ardoise `#64748B`.
L'interface et le logo partagent donc exactement les mêmes valeurs. Passer par
le fichier de thème officiel plutôt que par du CSS injecté garantit que les
composants internes de Streamlit — curseurs, cases, menus — suivent la palette
sans sélecteurs fragiles.

**Aucune détection ?** C'est souvent normal : les règles exigent une **durée**.
Traverser une zone ne déclenche rien ; il faut y stationner au-delà de
`min_duration_s` (3 s pour une intrusion, 60 s pour du rôdage, 30 s pour un objet
abandonné). Les polygones de `config.ZONES` doivent par ailleurs correspondre à
la scène filmée — voir « Adapter les zones à sa propre vidéo » ci-dessus.

**Surveillance globale par défaut.** `config.ZONES` ne contient qu'une seule
entrée, qui couvre l'intégralité de l'image : **tout ce qui est visible est
surveillé**. Aucun polygone n'est tracé sur la vidéo — l'opérateur voit l'image
nette, avec seulement les boîtes de détection.

Pourquoi conserver un périmètre nommé plutôt que supprimer le mécanisme : toutes
les règles reposent sur un chronomètre « depuis quand cet objet est-il **ici** ».
Sans périmètre, il n'y a plus de « ici », donc plus de durée mesurable, donc plus
d'incident. La zone plein cadre est le périmètre le plus simple possible.

**Restreindre à une portion de l'image**, si le besoin s'en présente : remplacer
l'entrée par un ou plusieurs polygones en coordonnées normalisées, `(0, 0)` en
haut à gauche et `(1, 1)` en bas à droite. Le mécanisme est inchangé et
pleinement testé.

```python
ZONES = (
    SurveillanceZone(
        name="Quai de chargement",
        polygon=((0.05, 0.45), (0.45, 0.45), (0.45, 0.95), (0.05, 0.95)),
        zone_type=ZoneType.FORBIDDEN,   # toute présence y est une infraction
        draw=True,                      # tracé, contrairement au plein cadre
    ),
    SurveillanceZone(
        name="Hall d'accueil",
        polygon=((0.55, 0.40), (0.98, 0.40), (0.98, 0.98), (0.55, 0.98)),
        zone_type=ZoneType.TRANSIT,     # on y passe : seul le rôdage compte
        min_duration_s=90.0,            # plus tolérant qu'ailleurs
        cooldown_s=300.0,
    ),
)
```

**Le type de zone décide des règles applicables.** Une zone `TRANSIT` ne lève
jamais d'intrusion — traverser un hall est normal — mais signale le rôdage. Une
zone `COUNTING` ne lève **rien** : compter les entrées d'un magasin ne doit pas
produire un rapport par client.

| Type | Ce qu'il attend | Règles levées |
|---|---|---|
| `FORBIDDEN` | Toute présence y est une infraction | les quatre |
| `SCHEDULED` | Présence normale aux heures d'ouverture | rôdage, objet abandonné, hors horaires |
| `TRANSIT` | On y passe, on n'y reste pas | rôdage, objet abandonné |
| `SENSITIVE` | Le risque est l'objet déposé | objet abandonné, hors horaires |
| `COUNTING` | Mesure de flux | aucune |

**Les seuils se surchargent par zone.** `min_duration_s`, `cooldown_s` et
`max_movement_ratio` valent `None` par défaut, ce qui signifie « suivre la valeur
globale de `EVENT_RULES` ». Les renseigner permet d'être plus strict dans une
réserve que dans un hall sans dupliquer le jeu de règles. `None` et une valeur
identique à la globale ne sont pas la même chose : la première suit les
ajustements futurs, la seconde les ignore.

Une surcharge qui casserait l'invariant `cooldown_s > min_duration_s` est
**refusée au démarrage** par `ZoneManager`, avec le nom de la zone et de la règle
en cause. La découvrir au milieu de l'analyse d'une vidéo d'une heure coûterait
l'analyse entière.

**Un seul mécanisme de ciblage.** `EventRule.zones` a été supprimé : c'est la
zone qui déclare les règles qu'elle accepte, jamais l'inverse. Deux mécanismes
concurrents auraient exigé une règle de résolution de conflit — donc un
paragraphe de plus ici, et une source d'erreur de plus dans la configuration.

**Ajuster les seuils.** Tout se règle dans `config.py` :
`MODEL.confidence` (sensibilité de la détection), `EVENT_RULES[*].min_duration_s`
(durée avant déclenchement), `EVENT_RULES[*].cooldown_s` (anti-rebond),
`SCHEDULE` (horaires d'ouverture).

---

## 6. Avancement

| Étape | Module | État |
|---|---|---|
| 1 | Squelette, `config.py`, `requirements.txt`, README | ✅ fait |
| 2 | `detector.py` — YOLOv8 + sélecteur de modèle | ✅ fait |
| 3 | `tracker.py` — ByteTrack + mémoire temporelle | ✅ fait |
| 4 | `zones.py` — zones polygonales | ✅ fait |
| 5 | `events.py` — les 4 règles d'incident + anti-rebond | ✅ fait |
| 6 | `report.py` — gabarits français + PDF + CSV | ✅ fait |
| 7 | `app.py` — interface Streamlit | ✅ fait |
| 8 | Bonus : règle OBJET ABANDONNÉ | ✅ fait |
| 9 | Bonus : `generate_with_llm()` avec repli | ✅ fait |
| 10 | `source.py` — fichier / webcam / RTSP unifiés | ✅ fait |
| 11 | Score de priorité — barème transparent et traçable | ✅ fait |
| 12 | `timeline.py` + `session_report.py` — rapport à deux niveaux | ✅ fait |

### 6.1 Assainissement — aligner le code sur ce que le README annonçait

Les trois modules ci-dessus ont longtemps été **écrits, testés et jamais
appelés** : 1 321 lignes que l'exécutable n'importait pas. Un module qu'aucun
chemin d'exécution ne touche n'existe pas pour l'utilisateur, et un README qui
le décrit annonce un produit qui n'est pas livré. Cette campagne l'a corrigé.

| Phase | Correction | Pourquoi c'était un problème |
|---|---|---|
| A | Retrait de `supervision` | Le README argumentait son refus tout en la déclarant en dépendance dure : 30 Mo installés pour une fonction sans appelant. |
| B | `VideoSource` branché dans `app.py` | L'interface refaisait la lecture vidéo à la main, en plus faible. En direct, `frame_index / fps` sous-estimait toutes les durées dès qu'une image était sautée — donc toutes les règles. |
| C | Logique métier sortie d'`app.py` | Le fichier réécrivait les règles, validait la cadence contre deux constantes locales et rangeait les gravités dans son propre ordre, contre son propre docstring. |
| D | `timeline.py` et `session_report.py` branchés | Le rapport à deux niveaux était documenté mais inaccessible. |
| E | Score affiché dans le rapport d'incident | Le même incident portait « critique » à l'écran et aucune priorité sur le papier. |
| F | Surface publique et documentation | `__all__` décrivait le projet tel qu'il était à l'étape 3. |

**Le cœur du projet est fonctionnel** : 366 tests, et la chaîne source → suivi →
zones → incidents → chronologie → rapports PDF/CSV tourne de bout en bout, sur
fichier comme sur webcam.

---

## 7. Limites connues

Les énoncer explicitement fait partie du travail : un système de sécurité dont on
ignore les angles morts est dangereux.

**Limites de la détection**
- `yolov8n` est le plus petit modèle de la famille. Il rate des objets petits,
  très éloignés ou fortement occultés. Passer à `yolov8s`/`yolov8m` améliore le
  rappel au prix du temps de calcul.
- Les classes sont celles du jeu de données COCO. Un objet hors de ces 80 classes
  n'est pas détectable sans réentraînement.
- **Aucune arme à feu n'est détectable.** COCO ne contient ni pistolet, ni fusil,
  ni arme de poing : la catégorie « Objets dangereux » de l'interface se limite
  à `knife`, `scissors` et `baseball bat`, les trois seuls objets contondants du
  jeu de données. Un pistolet sera au mieux ignoré, au pire classé « cell phone ».
  Le signaler explicitement dans l'interface est un choix assumé : une case à
  cocher « armes à feu » qui ne se déclencherait jamais serait le pire défaut
  possible pour un système d'alerte, parce qu'il est silencieux. Détecter les
  armes à feu suppose un modèle réentraîné sur un jeu de données dédié.
- Performance dégradée de nuit, sous forte pluie, à contre-jour ou avec une
  caméra très inclinée.

**Limites du suivi**
- Le suivi est purement géométrique : ByteTrack n'utilise pas l'apparence
  visuelle. Dans une foule dense, des échanges d'identité restent possibles, et
  un objet occulté longtemps réapparaît sous un nouvel identifiant — son
  chronomètre repart à zéro.
- Aucune ré-identification entre caméras ou entre sessions.

**Limites du raisonnement**
- Les zones sont définies en 2D dans le plan image. Un objet loin derrière la
  zone mais dont le point d'appui se projette dedans produira un faux positif.
  Une homographie vers un plan de sol corrigerait cela. Les **seuils de distance**
  (immobilité, proximité du propriétaire) sont en revanche déjà corrigés de la
  perspective, en les rapportant à la taille apparente de l'objet — voir § 3.5.
- Les règles sont des seuils fixes, sans apprentissage du comportement normal
  du site.
- Le système ne distingue pas une personne autorisée d'une personne non
  autorisée : toute intrusion en zone restreinte est signalée de la même façon.

**Limites d'ingénierie**
- **Une source à la fois.** Fichier, webcam ou flux RTSP, mais un seul. Ni
  multi-caméras, ni ONVIF, ni base de données, ni authentification, ni alertes
  poussées — voir § 8.3 pour les raisons de cette exclusion.
- En direct, la qualité dépend de la caméra et du réseau autant que du logiciel.
  Une coupure brève est absorbée (`reconnect_attempts`), une coupure longue
  interrompt la surveillance en conservant l'analyse déjà produite.
- Si le traitement est plus lent que la caméra, des images sont **volontairement
  écartées** pour rester sur le direct. L'interface le signale : c'est un
  arbitrage assumé, pas une panne.
- L'analyse d'une longue vidéo est plus lente que le temps réel sur CPU.
- Le rapport est un **brouillon** ; sa validation par un humain est obligatoire.

## 8. Feuille de route et extensions

Le périmètre est découpé en trois niveaux. Ce découpage est un choix assumé : un
projet solo se juge sur sa finition, et quatre règles mesurées et documentées
valent mieux que huit règles approximatives.

### 8.1 Cœur — le produit livrable

Détection multi-classes, suivi ByteTrack, mémoire temporelle, zones polygonales,
les quatre règles d'incident (intrusion, présence prolongée, objet abandonné,
hors horaires), anti-rebond, journal horodaté avec image de preuve, rapports
PDF/CSV, interface Streamlit avec tableau de bord et réglages ajustables,
comptage d'objets distincts par zone. Suivi détaillé au § 6.

### 8.2 Avancé — ambitieux mais atteignable

| Extension | Apport | Librairies |
|---|---|---|
| Enrichissement LLM du rapport | Rédaction en langage naturel **par-dessus** des faits déjà établis, avec repli automatique sur le gabarit | `anthropic` |
| Franchissement de ligne virtuelle | Comptage entrée/sortie : raisonner en flux et plus seulement en présence | aucune — signe du produit vectoriel |
| Jeu de validation + métriques | Chiffrer le taux de détection et les fausses alertes par heure de vidéo | `csv`, `pandas` |
| Éditeur de zones dans l'interface | Redéfinir les polygones sans éditer `config.py` | `streamlit-drawable-canvas` |
| Persistance SQLite | Historique des incidents entre deux sessions | `sqlite3` |
| Conteneurisation | Reproductibilité de l'environnement | Docker |

### 8.3 Extensions futures — hors périmètre, et pourquoi

| Extension | Raison de l'exclusion |
|---|---|
| **Détection d'anomalies apprise** (modèle du comportement normal plutôt que seuils fixes) | Exige des semaines de vidéo du **même site** et, surtout, des incidents réels annotés pour évaluer le résultat. Effort non borné, valeur non démontrable dans le cadre du projet. |
| **Multi-caméra avec ré-identification** | Nécessite un modèle d'apparence (`torchreid`) et un jeu de données multi-vues ; change l'architecture (association inter-caméras, synchronisation d'horloges). Projet à part entière. |
| **Alertes temps réel** (courriel, webhook) | Techniquement peu coûteux, mais n'ajoute rien à la démonstration technique et introduit des secrets à gérer. |
| **Déploiement edge (Jetson)** | Bloqué par le matériel : suppose un export TensorRT et une quantification INT8 invérifiables sans la carte. |
| **Calibration par homographie** | Corrigerait les deux limites géométriques connues (rayon de proximité en pixels, projection au sol). Abordable techniquement, mais demande une calibration manuelle par caméra. |
| **Estimation de pose / détection de chute** | `yolov8n-pose` rend la détection facile, mais la règle « au sol » est fragile et aucune donnée ne permet de la valider ici. |

**Éthique et vie privée.** Toute mise en service réelle relèverait de la
réglementation sur la vidéosurveillance et la protection des données
(information des personnes, durée de conservation, base légale). Le projet ne
fait aucune reconnaissance faciale ni identification de personnes, et n'est pas
prévu pour cela.

---

## 9. Licence

Projet réalisé dans un cadre pédagogique.
