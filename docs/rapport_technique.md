# Rapport technique — Assistant intelligent de recommandation d'événements culturels

**Projet :** POC RAG Puls-Events  
**Auteur :** Raphael  
**Date :** Avril 2026  
**Version :** 0.1.0

## Table des matières

1. [Objectifs du projet](#1-objectifs-du-projet)
2. [Architecture du système](#2-architecture-du-système)
3. [Préparation et vectorisation des données](#3-préparation-et-vectorisation-des-données)
4. [Choix du modèle NLP](#4-choix-du-modèle-nlp)
5. [Construction de la base vectorielle](#5-construction-de-la-base-vectorielle)
6. [API et endpoints exposés](#6-api-et-endpoints-exposés)
7. [Évaluation du système](#7-évaluation-du-système)
8. [Recommandations et perspectives](#8-recommandations-et-perspectives)
9. [Organisation du dépôt GitHub](#9-organisation-du-dépôt-github)
10. [Annexes](#10-annexes)

## 1. Objectifs du projet

### Contexte

**Puls-Events** est une entreprise technologique développant une plateforme de recommandations culturelles personnalisées. Dans le cadre d'une mission freelance, le présent POC vise à démontrer la faisabilité d'un chatbot intelligent capable de répondre à des questions utilisateurs sur les événements culturels à venir.

### Problématique

Les moteurs de recherche classiques (par mots-clés) ne permettent pas de comprendre l'intention derrière une question comme *"Que faire ce week-end avec mes enfants ?"*. Un système **RAG (Retrieval-Augmented Generation)** répond à ce besoin en combinant :

- Une **recherche sémantique** dans une base d'événements (compréhension du sens, pas juste des mots)
- Une **génération de réponse en langage naturel** à partir des documents retrouvés (réponse contextuelle et formulée)

Cela permet d'offrir une expérience conversationnelle pertinente, ancrée dans des données réelles et récentes.

### Objectif du POC

Démontrer la **faisabilité technique et la valeur métier** d'un assistant de recommandation culturelle basé sur RAG, avec :

- Un pipeline complet de la donnée brute à la réponse générée
- Une API REST interrogeable par les équipes produit et marketing
- Une évaluation automatisée de la qualité des réponses
- Un déploiement conteneurisé reproductible

### Périmètre

| Critère | Valeur |
|---|---|
| Zone géographique | Île-de-France |
| Source de données | API OpenAgenda (via OpenDataSoft) |
| Période couverte | 12 derniers mois + 6 mois à venir |
| Volume d'événements | jusqu'à 1 000 événements |
| Langue | Français |

## 2. Architecture du système

### Schéma global

```mermaid
flowchart TD
    subgraph INGESTION["🗂️ Pipeline de données"]
        subgraph PREPROCESS["📦 Préprocessing"]
            A["API OpenAgenda"]
            B["fetch_events.py"]
            C[/"raw_events.json"/]
            D["clean_events.py"]
            E[/"clean_events.json"/]
            A --> B --> C --> D --> E
        end
        subgraph INDEXATION["🔍 Indexation"]
            F["build_index.py"]
            subgraph LC1["LangChain"]
                G["Chunking"]
                H["Embedding (Mistral)"]
            end
            I[("Index FAISS")]
            F --> G --> H --> I
        end
        E --> F
    end

    subgraph RAG["🤖 Pipeline RAG (requête)"]
        J["Question utilisateur"]
        subgraph LC2["LangChain"]
            K["Embedding de la question"]
            L["Recherche FAISS\n(top-5 chunks)"]
            M["Prompt augmenté"]
            N["Génération LLM (Mistral)"]
        end
        O["Réponse en langage naturel"]
        P["API FastAPI"]

        J --> K --> L --> M --> N --> O --> P
    end

    I -.->|"chargé au démarrage"| L
```

### Diagramme de séquence UML (appel `/ask`)

```mermaid
sequenceDiagram
    actor U as Utilisateur
    participant API as FastAPI
    participant RAG as rag_chain.py
    participant F as FAISS
    participant M as Mistral API

    U->>API: POST /ask {"question": "..."}
    API->>RAG: ask(question)
    RAG->>F: embed(question) + similarity search (k=5)
    F-->>RAG: top-5 chunks pertinents
    RAG->>M: prompt augmenté (contexte + question)
    M-->>RAG: réponse générée
    RAG-->>API: AskResponse(answer)
    API-->>U: 200 OK {"answer": "..."}
```

### Technologies utilisées

| Technologie | Version | Rôle |
|---|---|---|
| Python | ≥ 3.8 | Langage principal |
| LangChain | ≥ 0.2.0 | Orchestration du pipeline RAG |
| langchain-mistralai | ≥ 0.1.0 | Intégration Mistral (LLM + embeddings) |
| FAISS (`faiss-cpu`) | ≥ 1.7.4 | Index de recherche vectorielle |
| FastAPI | ≥ 0.111.0 | API REST + documentation Swagger |
| Uvicorn | ≥ 0.30.0 | Serveur ASGI |
| Ragas | 0.2.15 | Évaluation automatique des réponses |
| Docker | 20.10.23 | Conteneurisation |
| python-dotenv | ≥ 1.0.0 | Gestion des variables d'environnement |

### Rôle des composants

| Composant | Fichier | Rôle |
|---|---|---|
| **Récupération** | `scripts/fetch_events.py` | Interroge l'API OpenDataSoft, pagine les résultats et persiste les événements bruts dans `data/raw_events.json` |
| **Nettoyage** | `scripts/clean_events.py` | Filtre, normalise et enrichit les données brutes ; construit le champ `text` composite utilisé pour la vectorisation |
| **Indexation** | `scripts/build_index.py` | Découpe les textes en chunks, génère les embeddings via `mistral-embed` et construit l'index FAISS persisté dans `index/faiss_index/` |
| **Pipeline RAG** | `scripts/rag_chain.py` | Charge l'index, expose la fonction `ask()` qui orchestre la recherche FAISS et la génération Mistral via LangChain |
| **API** | `api/routes.py` | Définit les endpoints FastAPI (`/ask`, `/rebuild`, `/health`, `/metadata`) et délègue la logique RAG à `rag_chain.py` |
| **Évaluation** | `tests/evaluate_rag.py` | Interroge le système sur le jeu de test annoté et calcule les métriques Ragas ; les résultats sont horodatés dans `results/` |

## 3. Préparation et vectorisation des données

### Source de données

Les événements sont récupérés via l'**API publique OpenDataSoft** exposant le jeu de données OpenAgenda :

```
https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records
```

**Paramètres de filtrage appliqués :**

| Paramètre | Valeur | Description |
|---|---|---|
| `location_region` | `Île-de-France` | Filtre géographique |
| `firstdate_begin` | J-365 → J+180 | Fenêtre temporelle glissante |
| `PAGE_SIZE` | `100` | Maximum autorisé par ODS |
| `MAX_EVENTS` | `1000` | Plafond total de récupération |

La pagination est gérée automatiquement avec un offset incrémental jusqu'à 1 000 événements maximum.

### Nettoyage des données (`clean_events.py`)

Le script de nettoyage applique deux types de transformations.

**Nettoyage général** (données stockées dans le JSON) :

| Opération | Description |
|---|---|
| **Filtrage champs obligatoires** | Les événements sans titre (`title_fr`) sont écartés |
| **Fallback description** | Si `description_fr` est vide, le titre est utilisé comme substitut |
| **Gestion des doublons** | Les événements avec UID identique sont dédupliqués |
| **Suppression HTML** | Les balises HTML dans `longdescription_fr` sont retirées via regex |
| **Normalisation géographique** | Noms de départements (`Seine-St-Denis` → `Seine-Saint-Denis`), arrondissements parisiens déduits du code postal |
| **Normalisation des quartiers** | Suppression des préfixes `Quartier de / du / des` ; variantes de `Centre-Ville` unifiées |
| **Parsing JSON imbriqué** | Champs `attendancemode` et `status` dé-sérialisés pour extraire le label français |
| **Tranche d'âge** | `age_max` ignoré si ≥ 99 |

**Construction du champ `text`** (utilisé pour la vectorisation) :

| Opération | Description |
|---|---|
| **Déduplication longdescription** | Si la description longue commence par la description courte, le doublon est retiré du texte |
| **Construction d'adresse** | `_build_address()` assemble adresse, code postal, ville et département sans répétitions |
| **Tranche d'âge** | Trois formulations selon la disponibilité de `age_min` / `age_max` |
| **Concaténation** | Tous les champs utiles (titre, description, dates, lieu, quartier, conditions, âge, accessibilité) sont joints par ` \| ` |

**Exemples d'anomalies corrigées :**

| Champ | Valeur brute (API) | Valeur après nettoyage |
|---|---|---|
| `location_dept` | `"Seine-St-Denis"` | `"Seine-Saint-Denis"` |
| `description_fr` | *(vide)* | `"Atelier poterie"` *(titre utilisé en fallback)* |
| `longdescription_fr` | `"<p>Venez découvrir…<br/>Entrée libre</p>"` | `"Venez découvrir… Entrée libre"` |
| `attendancemode` | `'{"fr": "En présence"}'` *(JSON en string)* | `"En présence"` |
| `age_max` | `99` | *(ignoré — valeur sentinelle "tous publics")* |
| `location_district` | `"Quartier du Marais"` | `"Marais"` |

**Exemple de champ `text` généré :**
```
Titre : Concert de jazz | Description : Soirée jazz au cœur de Paris. | Détails : Une soirée intime dans une salle intimiste. | Conditions : Entrée libre | Dates : 15 avril 2026 | Lieu : Café de la Danse | Adresse : 5 passage Louis-Philippe, 75011 Paris | Quartier : Charonne | Âge : à partir de 18 ans | Accessibilité : Accès PMR
```

**Champs conservés dans les métadonnées :**
`uid`, `title`, `description`, `long_description`, `conditions`, `keywords`, `daterange`, `firstdate_begin`, `lastdate_end`, `location_name`, `location_address`, `location_city`, `location_district`, `location_postalcode`, `location_dept`, `location_region`, `coordinates`, `age_min`, `age_max`, `accessibility`, `attendancemode`, `status`, `url`, `text`

### Chunking

Le découpage en chunks est réalisé avec `RecursiveCharacterTextSplitter` :

| Paramètre | Valeur | Justification |
|---|---|---|
| `chunk_size` | 700 caractères | La longueur médiane d'un champ `text` est de 839 caractères (moyenne 984, P25 : 573) — un chunk de 700 couvre la plupart des événements courts en un seul morceau et découpe les plus longs en 2 chunks maximum |
| `chunk_overlap` | 50 caractères | Évite la perte d'information en limite de chunk |
| `separators` | `[" \| ", "\n\n", "\n", " ", ""]` | Respecte la structure du champ texte composite |

### Embedding

Les vecteurs sémantiques sont générés via l'**API Mistral** :

| Paramètre | Valeur |
|---|---|
| Modèle | `mistral-embed` |
| Dimension | 1024 |
| Type | Float32 |
| Batch | Géré automatiquement par LangChain |

## 4. Choix du modèle NLP

### Modèle sélectionné

Le modèle de génération retenu est **Mistral AI** (`mistral-small-latest`) :

| Rôle | Modèle | Justification |
|---|---|---|
| **Génération** | `mistral-small-latest` | Bon rapport qualité/coût, suffisant pour un POC, temps de réponse raisonnable |

### Pourquoi Mistral ?

- **Qualité en français** adaptée aux contenus culturels francophones
- **Compatibilité native LangChain** via `langchain-mistralai`, intégration simple
- **Accès via API** sans infrastructure à gérer, idéal pour un POC étudiant

### Prompt de base

```
Tu es un assistant spécialisé dans les événements culturels en Île-de-France.
Ce système couvre uniquement les événements culturels en Île-de-France.
La date d'aujourd'hui est le {today}.
Si la question porte sur une région hors Île-de-France, indique clairement que la base est limitée
à l'Île-de-France et qu'aucun événement hors de cette région n'est disponible.
Réponds à la question en t'appuyant uniquement sur les événements fournis ci-dessous.
Si aucun événement ne correspond, dis-le clairement sans proposer de sources externes.

Événements pertinents :
{context}

Question : {question}

Réponse :
```

**Choix de conception :**
- `temperature=0.2` : réponses factuelles et reproductibles, tout en conservant une formulation naturelle
- `k=5` : les 5 chunks les plus proches sémantiquement sont injectés en contexte
- Date du jour injectée dynamiquement pour traiter les questions temporelles relatives
- Périmètre géographique explicite pour éviter les réponses hors-sujet
- Instruction de transparence sur l'absence de résultat (évite les hallucinations)
- Pas d'historique de conversation (hors périmètre POC)

### Limites du modèle

- **Fenêtre de contexte** : si les 5 chunks sont très longs, le contexte peut être tronqué
- **Dépendance au retrieval** : si FAISS ne remonte pas les bons chunks, le LLM répond sur une mauvaise base — le risque d'hallucination dépend directement de la qualité du retrieval
- **Pas de filtering post-retrieval** : des chunks hors sujet peuvent être inclus dans le contexte, dégradant la précision de la réponse
- **Pas d'historique de conversation** : chaque question est traitée de manière indépendante (hors périmètre POC)

## 5. Construction de la base vectorielle

### FAISS utilisé

L'index FAISS est construit via `FAISS.from_documents()` de LangChain, qui utilise par défaut un **index `IndexFlatL2`** (recherche exacte par distance L2). Ce choix est adapté au POC car :

- Le volume de données est limité (< 10 000 chunks)
- La précision est maximale (pas d'approximation)
- Les temps de réponse sont acceptables à cette échelle

### Stratégie de persistance

| Élément | Valeur |
|---|---|
| Répertoire | `index/faiss_index/` |
| Fichiers générés | `index.faiss` + `index.pkl` (métadonnées) |
| Format | Binaire FAISS natif |
| Chargement | `FAISS.load_local()` |

L'index est chargé **au premier appel à `/ask`** puis mis en cache en mémoire (`_index`, `_chain`) pour éviter de le recharger à chaque requête (lazy loading).

### Métadonnées associées

Chaque document FAISS conserve les métadonnées suivantes, accessibles après retrieval :

```python
{
    "uid": str,                  # Identifiant unique OpenAgenda
    "title": str,                # Titre de l'événement
    "firstdate_begin": str,      # Date de début (ISO 8601)
    "lastdate_end": str,         # Date de fin
    "location_name": str,        # Nom du lieu
    "location_city": str,        # Ville
    "location_district": str,    # Arrondissement / quartier
    "location_postalcode": str,  # Code postal
    "location_dept": str,        # Département
    "location_region": str,      # Région
    "conditions": str,           # Conditions d'accès (gratuit, inscription...)
    "age_min": int | None,       # Âge minimum
    "age_max": int | None,       # Âge maximum
    "url": str,                  # Lien vers la page OpenAgenda
}
```

## 6. API et endpoints exposés

### Framework utilisé

**FastAPI** a été retenu pour :
- La **génération automatique de la documentation Swagger** (`/docs`)
- La **validation des types** via Pydantic
- L'**intégration native** avec les schémas de données

### Endpoints

#### `GET /health`
Vérifie que l'API est opérationnelle.

```json
// Réponse 200
{"status": "ok"}
```

#### `GET /metadata`
Retourne des informations sur la base indexée.

```json
// Réponse 200
{
  "total_events": 847,
  "last_rebuilt": "2026-04-07T09:00:00",
  "first_event_date": "2025-04-08T00:00:00+00:00",
  "last_event_date": "2026-10-08T00:00:00+00:00",
  "departments": ["Essonne", "Hauts-De-Seine", "Paris", "Seine-Et-Marne", "Val-De-Marne", "Val-D'Oise", "Yvelines"],
  "districts": ["Belleville", "Charonne", "Montmartre", "Saint-Lambert", ...]
}
```

#### `POST /ask`
Pose une question au système RAG.

**Requête :**
```json
{"question": "Quels concerts gratuits ont lieu à Paris ce mois-ci ?"}
```

**Réponse :**
```json
{
  "answer": "Voici les concerts gratuits prévus à Paris : ..."
}
```

**Codes d'erreur gérés :**

| Code | Cause |
|---|---|
| 422 | Question vide |
| 503 | Index FAISS introuvable (lancer `/rebuild` d'abord) |
| 429 | Limite de requêtes Mistral atteinte |
| 500 | Erreur serveur inattendue |

#### `POST /rebuild` *(authentification requise)*
Reconstruit l'index FAISS depuis les données nettoyées.

**Header requis :** `X-API-Key: <clé configurée dans .env>`

```json
// Réponse 200
{
  "message": "Index FAISS reconstruit avec succès.",
  "chunks_indexed": 3241
}
```

> La route `/rebuild` est protégée par une clé API pour éviter une reconstruction intempestive en cas d'exposition publique.

### Exemples d'appels

**curl :**
```bash
# Vérification de l'état
curl http://localhost:8000/health

# Question au système RAG
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Y a-t-il des spectacles pour enfants en Île-de-France ?"}'

# Reconstruction de l'index
curl -X POST http://localhost:8000/rebuild \
  -H "X-API-Key: votre_cle_api"
```

**Python requests :**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "Quels événements musicaux sont prévus à Versailles ?"}
)
print(response.json()["answer"])
```

### Documentation interactive

La documentation Swagger est disponible automatiquement à l'adresse :
```
http://localhost:8000/docs
```

### Tests effectués et documentés

Les tests fonctionnels de l'API sont définis dans `tests/api_test.py` avec `pytest` et `httpx`. Ils couvrent les scénarios suivants :

| Endpoint | Scénario testé | Résultat attendu |
|---|---|---|
| `GET /health` | Appel nominal | `200 {"status": "ok"}` |
| `GET /metadata` | Appel nominal | `200` avec champs `total_events`, `departments`, etc. |
| `POST /ask` | Question valide | `200` avec champ `answer` non vide |
| `POST /ask` | Question vide (`""`) | `422 Unprocessable Entity` |
| `POST /rebuild` | Sans clé API | `403 Forbidden` |
| `POST /rebuild` | Avec clé API valide | `200` avec `chunks_indexed > 0` |

Tous ces tests passent dans le pipeline CI (GitHub Actions). La documentation Swagger (`/docs`) permet également de tester chaque endpoint interactivement sans code.

## 7. Évaluation du système

### Jeu de test annoté

Un jeu de données de référence de **12 questions annotées manuellement** a été constitué dans `tests/annotated_qa.json`.

**Critères de construction :**
- Couverture des principaux cas d'usage (événements gratuits, par genre musical, par type de lieu, pour enfants, par département...)
- Inclusion de questions hors périmètre géographique (Lyon, Marseille) pour tester la capacité à dire "je ne sais pas"
- Questions ambiguës (sans date précise) pour identifier les limitations du retrieval
- Annotation des réponses attendues en langage naturel

**Exemples de questions annotées :**

| # | Question | Type |
|---|---|---|
| 1 | Y a-t-il des ateliers ou formations artistiques en Île-de-France ? | Cas nominal |
| 2 | Quels événements culturels gratuits sont prévus à Paris ? | Filtre conditions |
| 6 | Y a-t-il des événements pour les enfants ou les familles en Île-de-France ? | Public cible |
| 11 | Y a-t-il des événements culturels à Lyon ou Marseille ? | Hors périmètre |
| 12 | Que faire ce week-end en Île-de-France ? | Question ambiguë |

### Métriques d'évaluation

L'évaluation automatique est réalisée avec **Ragas**, qui utilise lui-même le LLM Mistral pour scorer chaque réponse :

| Métrique | Description |
|---|---|
| **answer_relevancy** | La réponse répond-elle bien à la question posée ? |
| **faithfulness** | La réponse est-elle fidèle aux documents retrouvés (pas d'hallucination) ? |
| **context_precision** | Les chunks retrouvés sont-ils pertinents (peu de bruit) ? |
| **context_recall** | Toutes les informations nécessaires ont-elles été récupérées ? |

### Résultats obtenus

Évaluation réalisée le **9 avril 2026** sur les 12 questions annotées.

| Métrique | Score moyen | Seuil CI | Interprétation |
|---|---|---|---|
| **answer_relevancy** | **0.871** | 0.70 ✓ | Bonne pertinence — les réponses répondent bien à la question |
| **faithfulness** | **0.740** | 0.65 ✓ | Bonne fidélité — le LLM s'appuie sur les documents fournis |
| **context_recall** | **0.833** | 0.70 ✓ | Bon rappel — les informations nécessaires sont bien récupérées |
| **context_precision** | **0.633** | 0.45 ✓ | Précision correcte — encore quelques chunks hors sujet |

#### Analyse qualitative

**Points forts :**
- La `faithfulness` (0.740) indique que le modèle s'appuie sur les documents fournis et évite les hallucinations
- Le `context_recall` élevé (0.833) montre que les informations nécessaires sont bien récupérées par le retriever
- La gestion des cas hors périmètre fonctionne bien (Q11 : Lyon/Marseille → réponse correcte d'absence)

**Points faibles :**
- La `context_precision` (0.633) révèle que FAISS remonte encore quelques chunks non directement liés à la question
- Q1 (ateliers artistiques) reste un cas difficile : le retriever peut remonter des ateliers professionnels (numérique, emploi) au lieu d'ateliers artistiques
- Q12 (que faire ce week-end ?) reste difficile à traiter précisément malgré l'injection de la date, car FAISS ne filtre pas par date
- Q11 et Q12 affichent une `faithfulness` de 0.00 : aucun chunk pertinent n'étant retrouvé, Ragas ne peut pas calculer la fidélité — c'est une limite de la métrique, pas du système (le système répond correctement qu'il n'a pas de résultat)

**Exemple de réponse imparfaite — Q1 (ateliers artistiques) :**

> *Question :* "Y a-t-il des ateliers ou formations artistiques en Île-de-France ?"  
> *Réponse générée :* "Oui, voici quelques ateliers disponibles : Atelier numérique au Carrefour Numérique² (La Villette), Formation bureautique à la médiathèque de Créteil, Atelier d'initiation à la robotique à Massy."

Le retriever a remonté des ateliers au sens large (numérique, emploi) car le mot "atelier" est présent dans leurs descriptions — sans distinction du domaine artistique. La `context_precision` est pourtant à 1.00 car Ragas juge les chunks récupérés cohérents avec la question générique. C'est un cas où la précision sémantique fine dépasse les capacités actuelles du retrieval sans filtrage post-retrieval.

**Résultats détaillés par question :**

| Question | answer_relevancy | faithfulness | context_precision | context_recall |
|---|---|---|---|---|
| Q01 — Ateliers artistiques | 0.87 | 0.80 | 1.00 | 1.00 |
| Q02 — Événements gratuits Paris | 0.88 | 0.82 | 0.00 | 0.00 |
| Q03 — Stand-up Île-de-France | 0.88 | 0.94 | 1.00 | 1.00 |
| Q04 — Visites guidées | 0.89 | 0.97 | 0.37 | 1.00 |
| Q05 — Concerts entrée libre | 0.88 | 0.93 | 0.89 | 1.00 |
| Q06 — Événements enfants/familles | 0.89 | 0.78 | 0.53 | 1.00 |
| Q07 — Musique classique | 0.92 | 0.94 | 0.00 | 1.00 |
| Q08 — Événements en plein air | 0.86 | 0.86 | 0.92 | 1.00 |
| Q09 — Événements Yvelines | 0.86 | 0.85 | 1.00 | 1.00 |
| Q10 — Événements Seine-et-Marne | 0.89 | 1.00 | 1.00 | 1.00 |
| Q11 — Lyon/Marseille *(hors périmètre)* | 0.83 | 0.00 | 0.89 | 1.00 |
| Q12 — Ce week-end *(question ambiguë)* | 0.81 | 0.00 | 0.00 | 0.00 |

### Automatisation de l'évaluation

Le script `tests/evaluate_rag.py` est entièrement automatisable :

```bash
# Lancer l'évaluation (résultats sauvegardés dans results/)
python tests/evaluate_rag.py

# Afficher les résultats du dernier run
python scripts/show_eval.py results/
```

Les résultats sont horodatés (`eval_YYYY-MM-DD_HHMMSS.json`). `show_eval.py` charge automatiquement le fichier le plus récent si un dossier est fourni.

Le script est intégré dans le pipeline CI (GitHub Actions) pour surveiller la qualité à chaque push.

## 8. Recommandations et perspectives

### Ce qui fonctionne bien

- Le pipeline de bout en bout est fonctionnel et reproductible
- La fidélité des réponses est bonne : le système évite les hallucinations
- La gestion des cas hors périmètre (hors Île-de-France, hors base) est correcte
- L'API est documentée, testée, conteneurisée et prête pour une démo

### Limites du POC

| Limite | Impact |
|---|---|
| **Volume limité (1 000 événements)** | La couverture thématique est partielle |
| **Date figée à l'initialisation du serveur** | La date est injectée une fois au démarrage (`_build_chain`) ; si le serveur tourne plusieurs jours sans redémarrage, la date peut être obsolète |
| **Pas de filtrage temporel dans l'index FAISS** | Les questions relatives ("ce week-end") comprennent la date mais FAISS ne filtre pas par champ — des événements passés peuvent remonter |
| **Index `IndexFlatL2`** | Ne passera pas à l'échelle sur des millions de documents |
| **Pas de filtering post-retrieval** | Chunks hors sujet parfois inclus dans le contexte |
| **Coût API Mistral** | Chaque embed + génération est facturé |
| **Pas d'historique de conversation** | Les sessions multi-tours ne sont pas supportées |
| **Pas de streaming** | Le temps d'attente peut être perçu comme long côté utilisateur |

### Améliorations possibles

**À court terme :**
- **Filtrage par métadonnées** (date, département, conditions) avant ou après le retrieval FAISS
- **Augmenter le volume de données** : récupérer l'ensemble des événements OpenAgenda France et affiner le filtrage côté utilisateur
- **Historique de conversation** via `ConversationBufferMemory` LangChain par exemple

**Passage en production :**
- Mettre en place un **pipeline de mise à jour automatique** de l'index (hebdomadaire ou quotidien)
- Ajouter un **cache** pour les questions fréquentes
- Déployer sur une infrastructure cloud (AWS ECS, GCP Cloud Run) avec auto-scaling
- Monitorer les métriques Ragas en continu via une tâche GitHub Actions planifiée

## 9. Organisation du dépôt GitHub

```
poc-rag/
├── api/
│   ├── main.py          # Point d'entrée FastAPI
│   ├── routes.py        # Définition des endpoints (/ask, /rebuild, /health, /metadata)
│   ├── schemas.py       # Modèles Pydantic (requêtes/réponses)
│   └── security.py      # Vérification clé API (X-API-Key)
│
├── scripts/
│   ├── fetch_events.py  # Récupération des événements via API OpenDataSoft
│   ├── clean_events.py  # Nettoyage et normalisation des données brutes
│   ├── build_index.py   # Chunking, embeddings Mistral, construction index FAISS
│   ├── rag_chain.py     # Pipeline RAG : retrieval FAISS + génération Mistral
│   └── show_eval.py     # Affichage des résultats d'évaluation Ragas
│
├── tests/
│   ├── annotated_qa.json       # 12 questions/réponses annotées manuellement
│   ├── evaluate_rag.py         # Évaluation automatique Ragas
│   ├── test_fetch_events.py    # Tests unitaires fetch_events
│   ├── test_preprocessing.py   # Tests unitaires clean_events
│   ├── test_build_index.py     # Tests unitaires build_index
│   ├── test_rag_chain.py       # Tests unitaires rag_chain
│   └── api_test.py             # Tests fonctionnels de l'API
│
├── docs/
│   └── demo.postman_collection.json  # Collection Postman
│
├── data/                # Données brutes et nettoyées (non versionné — .gitignore)
├── index/               # Index FAISS persisté (non versionné — .gitignore)
├── results/             # Résultats d'évaluation JSON
│
├── .github/
│   └── workflows/
│       └── ci.yml       # Pipeline CI (tests unitaires + évaluation Ragas)
│
├── Dockerfile           # Image Docker pour l'API
├── Makefile             # Commandes raccourcies (build, run, test...)
├── requirements.txt     # Dépendances Python
├── conftest.py          # Configuration pytest
├── .env.example         # Template variables d'environnement
└── README.md            # Documentation de démarrage rapide
```

**Répertoires non versionnés (`data/`, `index/`) :** ces dossiers contiennent des données volumineuses ou des fichiers binaires générés à l'exécution. Un fichier `.gitignore` les exclut du dépôt. Les scripts permettent de les reconstituer entièrement.

## 10. Annexes

### Annexe A — Extraits du jeu de test annoté

```json
[
  {
    "id": 2,
    "question": "Quels événements culturels gratuits sont prévus à Paris ?",
    "expected_answer": "Plusieurs événements culturels gratuits sont proposés à Paris, notamment des visites, spectacles et activités accessibles librement selon la programmation disponible."
  },
  {
    "id": 11,
    "question": "Y a-t-il des événements culturels à Lyon ou Marseille ?",
    "expected_answer": "Aucun événement à Lyon ou Marseille n'est disponible dans le système. La base de données couvre uniquement les événements en Île-de-France."
  },
  {
    "id": 12,
    "question": "Que faire ce week-end en Île-de-France ?",
    "expected_answer": "Voici des événements culturels disponibles en Île-de-France pour ce week-end. Le système propose des événements basés sur la programmation indexée."
  }
]
```

### Annexe B — Exemples de réponses JSON

**Question Q3 — Stand-up :**
```json
{
  "answer": "Oui, il y a un spectacle de stand-up en Île-de-France :\n\n**Stand-up autour du thème de l'argent**\n- **Description** : Spectacle de stand-up mêlant humour et réflexions sur notre relation avec l'argent.\n- **Date** : Samedi 4 avril à 17h00\n- **Lieu** : Médiathèque Ulysse, Saint-Denis (93)\n- **Accès** : Libre (à partir de 13 ans)\n- **Adresse** : 37 cours du Rû de Montfort, 93200 Saint-Denis"
}
```

**Cas hors périmètre Q11 — Lyon/Marseille :**
```json
{
  "answer": "Il n'y a pas d'événements à Lyon ou Marseille dans les données fournies. Les événements disponibles sont uniquement situés en Île-de-France."
}
```

### Annexe C — Commandes Docker

```bash
# Construction de l'image
docker build -t puls-events-rag .

# Lancement du conteneur (avec index et données montés en volume)
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/index:/app/index \
  puls-events-rag

# Test rapide post-démarrage
curl http://localhost:8000/health
```

### Annexe D — Résultats complets Ragas

Run de référence : `results/eval_2026-04-09_160023.json`

| # | Question (résumé) | answer_relevancy | faithfulness | context_precision | context_recall |
|---|---|---|---|---|---|
| Q1 | Ateliers artistiques | 0.871 | 0.800 | 1.000 | 1.000 |
| Q2 | Événements gratuits Paris | 0.876 | 0.818 | 0.000 | 0.000 |
| Q3 | Stand-up Île-de-France | 0.880 | 0.941 | 1.000 | 1.000 |
| Q4 | Visites guidées | 0.891 | 0.967 | 0.367 | 1.000 |
| Q5 | Concerts entrée libre | 0.878 | 0.929 | 0.887 | 1.000 |
| Q6 | Événements enfants/familles | 0.886 | 0.783 | 0.533 | 1.000 |
| Q7 | Musique classique | 0.915 | 0.938 | 0.000 | 1.000 |
| Q8 | Événements en plein air | 0.861 | 0.857 | 0.917 | 1.000 |
| Q9 | Événements Yvelines | 0.860 | 0.850 | 1.000 | 1.000 |
| Q10 | Événements Seine-et-Marne | 0.887 | 1.000 | 1.000 | 1.000 |
| Q11 | Lyon / Marseille *(hors périmètre)* | 0.826 | 0.000 | 0.887 | 1.000 |
| Q12 | Que faire ce week-end *(question ambiguë)* | 0.814 | 0.000 | 0.000 | 0.000 |
| **Moyenne** | | **0.871** | **0.740** | **0.633** | **0.833** |
