# Notes de migration — SentinelReport

Campagne en quatorze phases, un commit par phase. Ce document dit, pour chacune :
**ce qui a changé**, **ce que ça débloque**, et **ce qu'il faut vérifier à l'œil**.

| | |
|---|---|
| Tests avant | **238** |
| Tests après | **594** |
| Environnements | Python 3.11.5 / Streamlit 1.44.1 **et** Python 3.14.0 / Streamlit 1.54 |
| Phases revertées | **aucune** |
| Dépendances ajoutées | **aucune** |
| Dépendances retirées | **une** (`supervision`) |

La suite est verte sur les deux interpréteurs à la fin de chaque phase. Aucune
phase n'a nécessité de `git revert`.

---

## Vue d'ensemble

```
238  état de référence
266  A  retire supervision
281  B  branche VideoSource dans app.py
311  C  sort la logique métier d'app.py
327  D  branche timeline.py et session_report.py
341  E  le score de priorité dans le rapport d'incident
366  F  surface publique et documentation
383  1  ouvre EventEngine aux règles multi-objets
412  2  SurveillanceZone remplace ZoneConfig
433  3  frontière à marge signée
468  4  occupation des zones et surdensité
500  5  franchissement de ligne
517  6  objet abandonné relationnel
565  8  garde-fous config, rétention, capture extraite
594  7  ré-association des pistes perdues (optionnelle)
```

> **Rectification.** Le message du commit de la phase 6 annonce « 500 → 516 » ;
> le compte réel était **517**. L'écart vient d'un test parametrizé ajouté au
> passage dans `test_package_surface.py`. Les chiffres de ce tableau font foi.

---

# Partie I — Assainissement

Ces six phases n'ajoutent aucune fonctionnalité. Elles rendent vrai ce que le
README affirmait déjà, pour que les extensions de la partie II se posent sur
l'architecture réelle et non sur celle du diagramme.

## Phase A — Retirer `supervision`

**Ce qui a changé.** `detection.to_supervision()` supprimée, `supervision`
retirée de `requirements.txt`, en-tête de `zones.py` corrigé.

**Le problème.** Le README consacrait un paragraphe à *refuser* `supervision`
(~30 Mo, dont PyAV, pour ce que `cv2.pointPolygonTest` rend déjà) tout en la
déclarant en dépendance **dure**. Sa seule porte d'entrée n'avait aucun appelant,
tests compris. L'argumentaire se retournait contre le projet : coût payé,
bénéfice nul.

**Ce que ça débloque.** Une installation plus légère, et un argumentaire qui
tient. `tests/test_dependencies.py` verrouille le refus — une dépendance en trop
ne casse rien, elle coûte seulement, donc rien ne la signalerait.

**À vérifier à l'œil.** `pip list | grep supervision` doit être vide après une
réinstallation propre.

## Phase B — Brancher `VideoSource` dans `app.py`

**Ce qui a changé.** `process_video()` itère sur `VideoSource.frames()`.
`open_source()` déduit la nature du flux. La source est consommée dans un `with`.
`tracker.update()` reçoit `video_time=` et `wall_time=` portés par la `Frame`.
`SourceDisconnectedError` est rattrapée.

**Le problème.** `app.py` refaisait à la main, en plus faible, ce que
`sentinel/source.py` faisait proprement : `cv2.VideoCapture` + boucle `read()`.
Ni reconnexion, ni rattrapage des images accumulées, et surtout un temps métier
calculé en `frame_index / fps` **alors qu'en direct des images sont
volontairement sautées**. Toutes les durées étaient sous-estimées, donc toutes
les règles fausses.

**Ce que ça débloque.** Le direct devient exploitable : webcam et RTSP passent
par le même chemin qu'un fichier, avec reconnexion et horloge murale.

**À vérifier à l'œil.**
1. Analyser un fichier : le comportement doit être **identique** à avant.
2. Choisir « Webcam » et lancer : la ligne d'état doit afficher **« temps réel »**
   et non « temps vidéo ».
3. Si le traitement est plus lent que la caméra, un message doit apparaître sous
   la vidéo : *« N image(s) écartée(s) pour suivre le direct »*.

## Phase C — Sortir la logique métier d'`app.py`

**Ce qui a changé.** `config.scaled_rules()`, `Severity.rank`,
`worst_severity()`, `VIDEO.credible_fps()`, `UI.box_color`. `zones.py` expose
`color_for()` et `alert_color_for()`.

**Le problème.** Contre son propre docstring, `app.py` réécrivait les règles,
validait la cadence contre deux constantes locales, rangeait les gravités dans
son propre ordre et peignait les boîtes en dur.

**Ce que ça débloque.** Un futur script batch peut appliquer le facteur de durées
sans importer Streamlit. Et `source.py` comme `tracker.py` partagent désormais un
seul contrôle de vraisemblance du FPS, au lieu de deux replis locaux qui
pouvaient diverger.

**À vérifier à l'œil.** Le curseur « Durées de déclenchement » doit toujours
modifier le seuil annoncé dans la phrase de récapitulatif, avant le lancement.

## Phase D — Brancher `timeline.py` et `session_report.py`

**Ce qui a changé.** `Pipeline` devient un `NamedTuple` portant une `Timeline` et
un `SessionReportGenerator`. `process_video()` alimente la chronologie.
Chronologie et contexte survivent dans `st.session_state`.
`render_session_report()` affiche les deux parties et les exports.

**Le problème.** 1 321 lignes finies et testées n'étaient jamais importées par
`app.py`. Un module que l'exécutable n'appelle pas n'existe pas pour
l'utilisateur : le README annonçait un rapport à deux niveaux que personne ne
pouvait obtenir.

**Ce que ça débloque.** Le rapport de session, réédité à chaque rerun sans
relancer l'analyse.

**À vérifier à l'œil.** Après une analyse, une section **« Rapport de session »**
doit apparaître sous les incidents, avec deux parties (priorités, puis
chronologie) et deux boutons de téléchargement. **Elle doit s'afficher même si
aucun incident n'a été détecté** — une surveillance calme est un résultat.

## Phase E — Le score de priorité dans le rapport d'incident

**Ce qui a changé.** `_template_context()` expose `priorite`, `score` et
`justification`, **lus** sur `event.priority`. Le rapport texte et le PDF les
affichent.

**Le problème.** Le score était visible dans trois sorties sur quatre. Le rapport
d'incident isolé — celui qu'on imprime et qu'on joint à une main courante — n'en
disait rien. Un même incident portait « critique » à l'écran et aucune priorité
sur le papier.

**Ce que ça débloque.** Rien de nouveau : c'est une correction de cohérence. Mais
`tests/test_report_priority.py` pose le même verrou que celui de
`session_report.py` — `report.py` ne doit contenir ni `PriorityScore(`, ni
`points_for`, ni `level_for`, ni `SCORING`.

**À vérifier à l'œil.** Ouvrir le PDF d'un incident : la ligne *« Priorité :
Critique (150 pts) : … »* doit y figurer, avec le détail du calcul.

## Phase F — Surface publique et documentation

**Ce qui a changé.** `sentinel/__init__.py` décrit le pipeline entier **dans son
sens de lecture**. README remis en cohérence : diagramme, tableau d'avancement,
limites, compteurs.

**À vérifier à l'œil.** `from sentinel import Timeline, VideoSource` doit
fonctionner.

---

# Partie II — Extensions

## Phase 1 — Ouvrir `EventEngine` aux règles multi-objets

**Ce qui a changé.** `FrameContext` (objets de la frame, tracker, zones, deux
horloges) et `RuleOutcome`. Les quatre prédicats deviennent des générateurs. Une
table d'aiguillage remplace la cascade de `if`.

**Le problème.** La boucle `for objet: for règle:` interdisait
**structurellement** toute règle relationnelle. Le moteur détenait pourtant déjà
l'ensemble — il reçoit le `Tracker` et possède le `ZoneManager`. Ce qui manquait
n'était pas l'accès, mais le contrat.

**La distinction à retenir.** `context.in_zone()` **compte**, toute la scène.
`context.candidates()` **désigne**, en écartant les objets sous délai de garde.
C'est elle qui rend la surdensité possible : sept personnes forment un
attroupement même si six viennent d'être signalées.

**Ce que ça débloque.** Les phases 4, 5 et 6. Aucune n'était exprimable avant.

**À vérifier à l'œil.** Rien : comportement strictement identique. Les 366 tests
existants sont passés sans modification — c'est le contrôle.

## Phase 2 — `SurveillanceZone` remplace `ZoneConfig`

**Ce qui a changé.** `ZoneType` (interdite / horaires / transit / sensible /
comptage) conditionne les règles applicables. Trois seuils surchargeables par
zone. `restricted` devient une propriété **dérivée** du type. **`EventRule.zones`
est supprimé.**

**Arbitrage ① appliqué : la zone gagne.** Un seul mécanisme de ciblage, donc
aucune résolution de conflit à documenter.

**Ce que ça débloque.** Un site réel se décrit enfin : rôder dans un hall est
banal, rôder dans une réserve ne l'est pas, et compter les passages à une porte
ne doit lever aucun incident.

**Zéro régression.** La zone plein cadre livrée est `FORBIDDEN`, le seul type qui
déclare toutes les règles — « surveiller tout ce qui s'affiche ».

**À vérifier à l'œil.** Le comportement par défaut est inchangé. Pour éprouver la
nouveauté, remplacer `config.ZONES` par deux zones de types différents (exemple
au § 5 du README) et vérifier qu'une zone `TRANSIT` ne lève jamais d'intrusion.

**Effet de bord traité.** Un `conftest.py` a été ajouté à la racine : `tests` est
un nom si banal qu'il existait déjà comme paquet installé sur l'environnement
3.14, et `from tests.zone_doubles import ...` résolvait ailleurs.

## Phase 3 — Frontière à marge signée

**Ce qui a changé.** `zones_for()` rend `dict[str, float]` — la distance
**signée** au bord, rapportée à la hauteur apparente. `update_zones()` applique
une bande d'incertitude.

**Le problème.** Une boîte qui tremble sur la frontière fait osciller
l'appartenance, et chaque oscillation remet le chronomètre à zéro. Une intrusion
de cinquante secondes le long d'une clôture n'était jamais signalée.

**Deux hystérésis, deux défauts.** La bande *spatiale* absorbe l'imprécision de
la boîte ; les compteurs de frames *temporels* absorbent les détections
erratiques. La première ne remplace pas la seconde.

**L'exemption à retenir.** Une zone qui épouse le cadre est dispensée de marge :
un objet ne peut pas sortir latéralement de l'image, il disparaît. Lui appliquer
une bande créerait un **anneau aveugle** tout autour du champ — une personne dont
les pieds touchent le bas du cadre resterait « en cours d'entrée ».

**À vérifier à l'œil.** Sur une vidéo avec une sous-zone, un objet qui longe la
frontière ne doit plus faire clignoter sa boîte entre rouge et vert.

## Phase 4 — Occupation et surdensité

**Ce qui a changé.** `sentinel/occupancy.py` (`ZoneOccupancy`), nouvelle règle
`OVERCROWDING`, seuil surchargeable par `SurveillanceZone.min_occupancy`. La
chronologie reçoit les variations.

**Arbitrage ③ appliqué : une classe distincte.** `ZoneManager` reste **sans
mémoire**, donc géométrique. L'occupation est de nature opposée — une durée ne se
constate que dans le temps. Aucun test de `test_occupancy.py` n'appelle
`initialize()` : ni polygone, ni image, ni OpenCV.

**Ce que ça débloque.** « Plus de N personnes en zone X pendant T secondes », et
les transitions « Hall : 2 → 7 personnes » dans la chronologie — un fait
qu'aucune règle par objet ne saurait formuler.

**À vérifier à l'œil.** Sur une vidéo de foule, la partie 2 du rapport de session
doit contenir des lignes du type *« Champ de la caméra : 2 → 7 person »*.

## Phase 5 — Franchissement de ligne

**Ce qui a changé.** `config.CrossingLine` et `sentinel/crossing.py`.

**Arbitrage ④ appliqué : une structure séparée.** Une ligne a deux points, pas de
surface, pas de durée de séjour. Une zone `COUNTING` la référence **par son nom**.

**Le piège que le code évite.** Le produit vectoriel décrit une droite
**infinie** : sans second test d'intersection de segments, un objet passant dix
mètres au-delà de l'extrémité serait compté. L'erreur est silencieuse — les
chiffres restent plausibles, ils sont seulement faux.

**Aucune librairie.** `supervision.LineZone` coûterait ~30 Mo pour quatre
soustractions et deux multiplications.

**À vérifier à l'œil.** Aucune ligne n'est configurée par défaut. Pour éprouver :
déclarer une `CrossingLine` dans `config.CROSSING_LINES` (exemple commenté dans
le fichier), relancer, et vérifier les lignes *« Porte : person #4 — entrée »*
dans la chronologie. **Attention au sens** : pour une ligne tracée de gauche à
droite, le sens positif va vers le **bas** de l'image. Si le comptage est
inversé, échanger `start` et `end`.

## Phase 6 — Objet abandonné relationnel

**Ce qui a changé.** `TrackedObject.owner_votes` / `bind_owner()` / `owner_id`.
L'association se fait pendant `owner_binding_s` (5 s). Le déclenchement a lieu
quand `tracker.get(owner)` rend `None`.

**Le problème.** « Aucune personne dans le rayon **maintenant** » est fragile dans
les deux sens : en foule il y a toujours quelqu'un — un sac réellement abandonné
dans un hall de gare n'était jamais signalé ; dans un lieu désert, le premier
passant qui s'éloigne suffisait à déclencher.

**Le moment où l'on regarde.** Quand un sac est immobile depuis trente secondes,
la personne qui l'a posé est partie depuis longtemps. Il faut avoir regardé au
bon moment, pas au moment du déclenchement.

**Repli assumé.** Un objet apparu seul n'a pas de porteur observable ; le
voisinage instantané reste alors le seul critère. Le rapport distingue les deux
cas : *« #9 (parti) »* et *« aucun observé »*.

**À vérifier à l'œil.** Dans le rapport d'un incident « objet abandonné », le
champ **Proprietaire presume** doit dire lequel des deux cas s'applique.

## Phase 8 — Garde-fous, rétention, capture extraite

**Ce qui a changé.**
1. `config.validate()`, appelée **à l'import**.
2. `EVIDENCE.retention_days = 30`, purge au démarrage.
3. Accroche de floutage (`blur_bystanders`), **désactivée par défaut**.
4. `sentinel/evidence.py` : la capture quitte le moteur de règles.

**Le problème (1).** Une faute de frappe dans `config.py` ne produit pas une
erreur mais une **règle silencieusement inopérante**. Ces défauts ne se voient
qu'en relisant un rapport vide ou saturé — après l'analyse.

**Le problème (4).** `events.py` affirmait « le moteur ne détecte rien et ne
dessine rien » tout en ouvrant des fichiers cent vingt lignes plus bas. Les deux
responsabilités ont des raisons de changer différentes. La séparation est ce qui
a permis d'ajouter la rétention et le floutage **sans relire une ligne de règle**.

**Ce que le floutage n'est pas.** Ce n'est pas de l'anonymisation au sens
réglementaire : le flou porte sur la boîte entière et non sur les visages, et il
ne s'applique qu'aux personnes que le détecteur a vues. Une personne manquée par
le modèle n'est pas floutée. Le présenter autrement serait une promesse que le
code ne tient pas.

**À vérifier à l'œil.**
1. Introduire volontairement une faute dans `config.ZONES` (un sommet à `1.5`) :
   l'application doit **refuser de démarrer**, avec le nom de la zone en cause.
2. Les fichiers de `evidence/` datant de plus de 30 jours doivent disparaître au
   lancement. `.gitkeep` doit rester.
3. Passer `EVIDENCE.blur_bystanders = True` et relancer : sur une capture avec
   plusieurs personnes, seule celle en cause doit rester nette.

## Phase 7 — Ré-association des pistes perdues *(optionnelle)*

**Ce qui a changé.** `sentinel/reidentification.py` et `config.REID`, avec
`enabled = False`.

**Pourquoi elle est désactivée.** Son mode de panne est pire que le problème
qu'elle résout. Un chronomètre remis à zéro fait manquer un incident — faux
négatif, visible et corrigeable. Une ré-association erronée **fusionne deux
personnes en une seule piste**, et le rapport affirme qu'une personne est restée
quarante minutes là où deux se sont succédé : un document présenté comme
opposable énonce alors un fait faux. Entre manquer un incident et en fabriquer
un, un système de sécurité choisit le premier.

**Le garde demandé est en place.** Triple condition — position prédite, taille
apparente, histogramme `cv2.calcHist` — et **le doute vaut refus** : pas d'image,
région trop petite, classe différente, écart hors de [3 s, 15 s], piste sans
signature, ou **deux candidats également plausibles**. Sur les 26 tests du
module, la majorité vérifie un **refus**.

**Ce que ça débloque une fois activé.** Le rôdage survit à une occlusion de 3 à
15 secondes : `_resurrect()` restaure chronomètres de zone, votes de classe,
porteur présumé et anti-rebond. La foule dense n'est **pas** visée.

**À vérifier à l'œil.** Par défaut : rien ne change, le tracker n'invente aucune
identité. Pour éprouver : `config.REID` avec `enabled=True`, sur une vidéo où
quelqu'un passe derrière un obstacle — l'identifiant affiché sur la boîte doit
rester le même à la réapparition.

---

## Décisions prises en cours de route

Aucun point d'architecture n'a été laissé en suspens. Trois choix n'étaient pas
couverts par les arbitrages initiaux :

1. **`conftest.py` à la racine** (phase 2). Le dossier `tests/` entrait en
   collision avec un paquet `tests` installé sur l'environnement 3.14. Les
   doublures s'importent désormais sans préfixe.
2. **Réorganisation des sections de `config.py`** (phase 2). Une zone déclare les
   types d'incidents qu'elle accepte : le vocabulaire doit exister avant elle.
   C'est aussi l'ordre de lecture naturel — nommer les incidents possibles, puis
   dire où chacun s'applique.
3. **`REID.enabled = False`** (phase 7). Le brief autorisait à ne pas implémenter
   la phase. Elle l'est, entièrement testée, mais **inerte par défaut** : c'est
   strictement plus que le repli acceptable, sans en subir le risque.

## Ce qui reste hors périmètre

Inchangé, et documenté au § 8.3 du README : multi-caméras, ONVIF, base de
données, alertes poussées, authentification. Une source à la fois.
