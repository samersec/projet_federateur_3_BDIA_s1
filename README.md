# TuniCase AI
### Advanced RAG for Tunisian Commercial Case Law

<p align="center">
  **Intelligent Legal Research · Hybrid Retrieval · Explainable AI**
</p>

---

## 1. Présentation du projet

**TuniCase AI** est une plateforme intelligente de recherche et d'analyse de jurisprudence commerciale tunisienne, basée sur une architecture **Advanced Retrieval-Augmented Generation (RAG)**.

Le projet vise à faciliter l'accès aux décisions judiciaires tunisiennes en permettant aux professionnels du droit de rechercher des décisions pertinentes à partir de questions formulées en langage naturel, en arabe ou en français.

Contrairement à une recherche classique par mots-clés, TuniCase AI combine recherche sémantique et recherche lexicale afin de retrouver des décisions juridiquement pertinentes, même lorsque les formulations utilisées dans les documents diffèrent de celles de la question.

Chaque réponse générée doit être accompagnée de références vérifiables vers les décisions et les passages sources.

## 2. Problématique

La recherche de jurisprudence commerciale représente un défi pour les professionnels du droit :

- Les décisions judiciaires sont souvent disponibles sous forme de documents PDF.
- Les recherches classiques par mots-clés ne permettent pas toujours de retrouver des décisions exprimant une même idée juridique avec des formulations différentes.
- Les documents peuvent contenir du texte arabe, français ou des pages numérisées nécessitant un traitement OCR.
- L'identification des passages pertinents et la comparaison de plusieurs décisions demandent du temps.
- Les réponses produites par une IA doivent rester vérifiables et fondées sur des sources juridiques fiables.

**Notre objectif :** développer un système de recherche juridique intelligent capable de retrouver, classer et synthétiser les décisions pertinentes tout en conservant la traçabilité des sources.

## 3. Objectifs du projet

### Objectif général

Concevoir et implémenter une architecture Advanced RAG spécialisée dans la jurisprudence commerciale tunisienne.

### Objectifs spécifiques

1. Collecter et structurer un corpus de décisions judiciaires.
2. Extraire le contenu des documents PDF en arabe et en français.
3. Construire des embeddings multilingues adaptés au domaine juridique.
4. Mettre en place une recherche hybride combinant Dense Retrieval et BM25.
5. Améliorer la pertinence des résultats avec un mécanisme de reranking.
6. Générer des réponses contextualisées à partir des décisions retrouvées.
7. Fournir des citations précises permettant de vérifier chaque réponse.
8. Évaluer quantitativement et qualitativement les performances du système.

## 4. Périmètre fonctionnel

Le périmètre initial se concentre sur la jurisprudence commerciale tunisienne, notamment :

| Domaine | Exemples de sujets |
|---|---|
| Droit des sociétés | Responsabilité des dirigeants, associés, capital social |
| Contrats commerciaux | Inexécution, rupture contractuelle, responsabilité |
| Fonds de commerce | Cession, location-gérance, oppositions |
| Effets de commerce | Chèques, lettres de change, recouvrement |
| Procédure commerciale | Pourvoi en cassation, moyens de cassation, vices de procédure |

Le corpus initial privilégie les décisions de la Cour de cassation tunisienne.

**Source institutionnelle principale :** [Cour de cassation tunisienne](http://www.cassation.tn/)

Le périmètre pourra être étendu à d'autres sources juridiques après validation de leur disponibilité, de leur qualité et de leurs conditions d'utilisation.

## 5. Architecture technique

### Architecture générale

```text
                  DATA SOURCES
                       |
                       v
              DATA INGESTION
          Python / BeautifulSoup
          PyMuPDF / OCR / Docling
                       |
                       v
             DOCUMENT PROCESSING
          Extraction / Cleaning
          Metadata / Semantic Chunking
                       |
                       v
                 EMBEDDINGS
                    BGE-M3
                       |
                       v
                VECTOR STORAGE
                    Qdrant
                       |
                       v
               HYBRID RETRIEVAL
              Dense + Sparse Search
                       |
                       v
                 RRF FUSION
                       |
                       v
                CROSS-ENCODER
                   Reranking
                       |
                       v
                 LLM GENERATION
              Contextualized Answer
                       |
                       v
                 FASTAPI BACKEND
                       |
                       v
                 NEXT.JS FRONTEND
                       |
                       v
              ANSWER + CITATIONS
```

### Stack technologique

| Couche | Technologie | Responsabilité |
|---|---|---|
| Backend | FastAPI | Exposition des API et orchestration |
| Validation | Pydantic v2 | Validation et sérialisation des données |
| Vector Database | Qdrant | Stockage et recherche vectorielle |
| Embeddings | BGE-M3 | Représentation multilingue des passages |
| RAG Framework | LlamaIndex | Gestion des documents et pipelines RAG |
| Retrieval | BM25 + Dense Search + RRF | Recherche hybride |
| Reranking | Cross-Encoder | Reclassement des passages pertinents |
| LLM | Modèle multilingue à évaluer | Génération des réponses |
| Frontend | Next.js | Interface utilisateur |
| Streaming | Server-Sent Events (SSE) | Transmission progressive des réponses |
| Évaluation | Ragas | Évaluation des performances RAG |
| Observabilité | Langfuse | Traçabilité et monitoring |
| Infrastructure | Docker | Environnement reproductible |
| CI/CD | GitHub Actions | Tests et intégration continue |

### Justification des choix

- **FastAPI :** adapté aux API asynchrones et aux applications intégrant des modèles IA.
- **Qdrant :** recherche vectorielle HNSW, filtrage par métadonnées et prise en charge des recherches denses et sparse.
- **BGE-M3 :** modèle multilingue permettant de traiter des contenus dans plusieurs langues, notamment l'arabe et le français. Sa pertinence juridique devra être évaluée sur notre corpus.
- **BM25 :** permet de retrouver des correspondances lexicales précises, notamment les termes juridiques et références d'articles.
- **RRF :** fusionne les résultats issus de plusieurs mécanismes de recherche.
- **Cross-Encoder :** améliore le classement des passages avant leur transmission au LLM.
- **Ragas :** permet d'évaluer la fidélité, la pertinence et la qualité du contexte récupéré.
- **Langfuse :** facilite le suivi des requêtes, des prompts, des latences et des erreurs.

## 6. Stratégie d'ingestion des données

Le pipeline d'ingestion suit les étapes suivantes :

1. **Collecte :** récupération des décisions depuis les sources juridiques retenues.
2. **Extraction :** lecture des fichiers PDF avec PyMuPDF et recours à l'OCR lorsque nécessaire.
3. **Nettoyage :** suppression des éléments parasites tout en préservant le contenu juridique.
4. **Structuration :** extraction des métadonnées des décisions.
5. **Chunking :** segmentation du contenu en passages sémantiquement cohérents.
6. **Embedding :** transformation des passages en vecteurs avec BGE-M3.
7. **Indexation :** stockage des vecteurs et métadonnées dans Qdrant.

### Métadonnées envisagées

Chaque décision pourra contenir :

```json
{
  "decision_id": "DEC-001",
  "decision_number": null,
  "decision_date": null,
  "court": "Cour de cassation",
  "domain": "droit_des_societes",
  "language": "ar",
  "source_url": null,
  "document_path": "data/raw/decisions/decision_001.pdf",
  "page_number": 1,
  "chunk_id": "DEC-001-CHUNK-001"
}
```

Les champs dont les valeurs ne sont pas encore extraites restent à renseigner pendant le traitement des documents.

### Organisation des données

```text
data/
├── raw/
│   ├── decisions/
│   ├── markdown/
│   ├── json/
│   └── sql/
│
├── processed/
│   ├── extracted/
│   ├── cleaned/
│   └── chunks/
│
└── README.md
```

- `raw/` : documents originaux, conservés sans modification.
- `processed/` : documents extraits, nettoyés et structurés.
- `chunks/` : passages préparés pour l'indexation vectorielle.

## 7. Stratégie de recherche Advanced RAG

Le système utilise une architecture Hybrid Retrieval :

**Dense Retrieval :** recherche des passages selon leur similarité sémantique.

**Sparse Retrieval (BM25) :** recherche des correspondances lexicales exactes.

**Reciprocal Rank Fusion (RRF) :** fusion des classements issus des deux mécanismes.

**Cross-Encoder Reranking :** réévaluation de la pertinence des passages sélectionnés.

**LLM Generation :** génération d'une réponse en utilisant uniquement le contexte juridique récupéré, avec références aux sources.

En l'absence de sources suffisamment pertinentes, le système doit signaler qu'il ne dispose pas d'éléments suffisants pour répondre.

## 8. Évaluation et indicateurs de performance

La qualité du système sera évaluée à travers un benchmark constitué de questions juridiques et de décisions pertinentes identifiées et validées par l'équipe.

| Indicateur | Objectif initial |
|---|---:|
| Recall@5 | ≥ 80 % |
| MRR@10 | ≥ 0.75 |
| Faithfulness | ≥ 0.90 |
| Answer Relevance | ≥ 0.85 |
| Citation Accuracy | ≥ 95 % |

Ces valeurs sont des **objectifs expérimentaux**, et non des performances déjà obtenues.

### Évaluation

- Ragas pour les métriques automatisées.
- Validation humaine pour vérifier la pertinence juridique.
- Tests de récupération sur un jeu de questions de référence.
- Vérification de la correspondance entre les citations et les pages originales.
- Mesure des latences de recherche et de génération.

## 9. Organisation du projet

### Structure du repository

```text
TuniCase-AI/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── ingestion/
│   │   ├── retrieval/
│   │   ├── generation/
│   │   └── evaluation/
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── docs/
│   ├── architecture.md
│   ├── technical_choices.md
│   └── backlog.md
│
├── infrastructure/
│   └── docker-compose.yml
│
├── .env.example
├── .gitignore
└── README.md
```

### Matrice des responsabilités

| Membre | Responsabilité principale | Contributions |
|---|---|---|
| Mohamed Amine Ouelhazi | Architecture IA & RAG | Ingestion, embeddings, retrieval et Qdrant |
| Samer Soltani | Backend & Infrastructure | FastAPI, API, Docker et CI/CD |
| Sirine Rahal | Data & Évaluation / Frontend | Préparation du corpus, benchmark, évaluation et interface |

Cette répartition constitue une organisation initiale et pourra évoluer selon les besoins du projet.

## 10. Installation et démarrage

### Prérequis

- Git
- Python 3.11+
- Docker Desktop
- Docker Compose
- Node.js (pour le frontend)

### Cloner le repository

```bash
git clone <GITHUB_REPOSITORY_URL>
cd TuniCase-AI
```

### Configurer les variables d'environnement

```bash
cp .env.example .env
```

Renseigner les variables nécessaires, notamment les paramètres du modèle LLM si un service externe est utilisé.

### Démarrer Qdrant avec Docker

Depuis le dossier `infrastructure/` :

```bash
docker compose up -d
```

Vérifier le conteneur :

```bash
docker compose ps
```

Consulter les logs :

```bash
docker compose logs -f qdrant
```

### Accès à Qdrant

- API : http://localhost:6333
- Dashboard : http://localhost:6333/dashboard

### Arrêter l'environnement

```bash
docker compose down
```

Les données restent conservées dans le volume Docker `qdrant_storage`.

## 11. Roadmap

| Phase | Objectif | Statut |
|---|---|---|
| Phase 1 | Cadrage et définition de l'architecture | En cours |
| Phase 2 | Constitution du corpus initial | À réaliser |
| Phase 3 | Pipeline d'extraction et de traitement documentaire | À réaliser |
| Phase 4 | Embeddings et indexation Qdrant | À réaliser |
| Phase 5 | Recherche hybride et reranking | À réaliser |
| Phase 6 | API FastAPI et génération RAG | À réaliser |
| Phase 7 | Interface Next.js et streaming | À réaliser |
| Phase 8 | Évaluation, optimisation et déploiement | À réaliser |

## 12. Limites et considérations éthiques

TuniCase AI est un outil d'aide à la recherche juridique et non un système de décision judiciaire.

- Les réponses doivent être accompagnées de références vérifiables.
- Le système ne doit pas inventer des décisions, des articles ou des citations.
- Les arguments des parties doivent être distingués du raisonnement propre de la juridiction.
- Les documents originaux doivent être conservés pour permettre la vérification.
- Les limites du corpus doivent être clairement signalées.
- Toute interprétation juridique doit rester soumise à la validation d'un professionnel du droit.

## 13. Équipe du projet

**Projet :** TuniCase AI – Advanced RAG for Tunisian Commercial Case Law

**Équipe :**
- Mohamed Amine Ouelhazi
- Samer Soltani
- Sirine Rahal

**Domaine :** Artificial Intelligence & Big Data

**Type :** Projet académique d'ingénierie – Advanced RAG

---

*Building a more accessible, traceable and intelligent approach to Tunisian commercial case law.*