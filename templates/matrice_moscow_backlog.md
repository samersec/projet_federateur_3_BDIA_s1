# Backlog Technique MoSCoW — Projet Fédérateur AI-BD_BI
## TuniCase AI — Advanced RAG for Tunisian Commercial Case Law
### Découpage Agile des User Stories & Critères d'Acceptation

**Membres de l'équipe :**
1. Mohamed Amine Ouelhazi
2. Samer Soltani
3. Sirine Rahal

---

# 1. MUST HAVE (Indispensables pour la Livraison Finale S7)

Ces fonctionnalités constituent le socle technique indispensable pour livrer un moteur Advanced RAG opérationnel, capable d'exploiter la jurisprudence commerciale tunisienne.

| ID | User Story Technique | Critères d'Acceptation (DoD) | Séance Cible |
|---|---|---|---|
| US-01 | **Collecte et ingestion de la jurisprudence** — En tant que Data Engineer, je veux collecter automatiquement les décisions commerciales depuis le site officiel de la Cour de cassation tunisienne afin de constituer un corpus juridique exploitable. | - Script Python de collecte avec gestion des erreurs et des doublons.<br>- Téléchargement des PDF et conservation des URL sources.<br>- Extraction des métadonnées disponibles : numéro, date, source et catégorie.<br>- Journalisation des documents téléchargés et des erreurs. | S1 |
| US-02 | **Extraction documentaire et Chunking sémantique** — En tant que système RAG, je veux extraire et segmenter les décisions judiciaires en passages juridiquement cohérents afin de préserver leur contexte. | - Extraction du texte des PDF natifs et OCR des documents numérisés.<br>- Conservation de la langue originale arabe ou française.<br>- Découpage selon les sections juridiques lorsque leur structure est identifiable.<br>- Métadonnées obligatoires : doc_id, decision_number, decision_date, page_number, chunk_id, source et legal_domain.<br>- Conservation du PDF original et de la correspondance entre chunks et pages.<br>- Vérification de la qualité de l'extraction. | S2 |
| US-03 | **Indexation vectorielle et recherche hybride** — En tant que moteur de recherche juridique, je veux indexer les décisions dans Qdrant afin de retrouver les passages pertinents par recherche sémantique et lexicale. | - Qdrant opérationnel via Docker Compose.<br>- Embeddings multilingues BGE-M3 (1024 dimensions).<br>- Index vectoriel HNSW avec distance Cosine.<br>- Recherche dense et sparse BM25.<br>- Filtrage par date, numéro de décision et domaine juridique.<br>- Temps d'indexation mesuré sur un batch de 64 chunks. | S2 / S3 |
| US-04 | **Fusion des résultats et Reranking** — En tant que moteur RAG, je veux fusionner les résultats des recherches dense et sparse puis les reranker afin de maximiser la pertinence juridique des passages récupérés. | - Fusion des résultats avec Reciprocal Rank Fusion (RRF).<br>- Génération d'un Top-20 initial.<br>- Classement final avec un Cross-Encoder multilingue.<br>- Retour des 5 passages les plus pertinents.<br>- Mesure du Recall@5 avec un benchmark juridique annoté.<br>- Objectif initial : Recall@5 ≥ 80 %. | S3 |
| US-05 | **API REST FastAPI et validation Pydantic v2** — En tant qu'application cliente, je veux interroger le moteur juridique à travers une API typée et validée. | - Endpoints REST pour la recherche juridique et la génération de réponses.<br>- Schémas QueryRequest et QueryResponse avec Pydantic v2.<br>- Validation des questions vides et des limites de longueur.<br>- Retour HTTP 422 en cas de données invalides.<br>- Documentation interactive OpenAPI / Swagger.<br>- Gestion des erreurs et des délais d'attente. | S4 |
| US-06 | **Génération de réponses juridiques avec Streaming SSE** — En tant qu'avocat, je veux recevoir progressivement une réponse à ma question juridique avec les décisions et les passages sources afin de réduire le temps de recherche. | - Endpoint `/api/v1/chat/stream` utilisant Server-Sent Events.<br>- Génération progressive de la réponse.<br>- Transmission des références des décisions et des pages sources.<br>- Affichage des sources utilisées et des extraits justificatifs.<br>- Gestion des réponses sans résultats pertinents.<br>- Mesure du Time-to-First-Token (TTFT). | S4 / S5 |
| US-07 | **Interface Web de recherche juridique** — En tant qu'utilisateur, je veux poser une question en langage naturel et consulter les réponses ainsi que les décisions judiciaires citées. | - Interface Web avec Next.js / React.<br>- Zone de saisie pour les questions en arabe et en français.<br>- Affichage de la réponse en streaming.<br>- Présentation des décisions citées : numéro, date et domaine juridique.<br>- Consultation des passages originaux et des pages correspondantes.<br>- Gestion des états : idle, loading, error et success.<br>- Navigation sans rechargement de page. | S5 |
| US-08 | **Assurance qualité et CI/CD automatisée** — En tant que Lead Developer, je veux automatiser les tests et les contrôles de qualité pour garantir la stabilité du moteur RAG. | - Tests unitaires et d'intégration avec Pytest.<br>- Mocks pour le LLM, Qdrant et les services externes.<br>- Tests du scraping, du chunking, du retrieval et de l'API.<br>- Vérification de la qualité du code avec Ruff.<br>- Workflow GitHub Actions déclenché à chaque Pull Request.<br>- Blocage du merge en cas d'échec des tests. | S6 |
| US-09 | **Conteneurisation et déploiement** — En tant qu'ingénieur DevOps, je veux déployer les composants du système à travers Docker Compose afin de faciliter l'installation et la reproductibilité. | - Dockerfile multi-stage pour le backend.<br>- Services conteneurisés : FastAPI, Qdrant et PostgreSQL.<br>- Configuration centralisée via variables d'environnement.<br>- Utilisation d'un utilisateur non-root lorsque possible.<br>- Persistance des données avec Docker Volumes.<br>- Secrets exclus du dépôt GitHub. | S7 |

---

# 2. SHOULD HAVE (Forte Valeur Ajoutée Industrielle)

Ces fonctionnalités renforcent la fiabilité, l'observabilité et la qualité des réponses du moteur juridique.

| ID | User Story Technique | Critères d'Acceptation | Séance Cible |
|---|---|---|---|
| US-10 | **Tracing MLOps distribué (Langfuse / OpenTelemetry)** — En tant que responsable IA, je veux tracer chaque étape du pipeline RAG afin d'identifier les problèmes de performance. | - Traçage des étapes : ingestion, embedding, retrieval, reranking et génération.<br>- Mesure de la latence par étape.<br>- Enregistrement des tokens consommés et du coût estimé lorsque disponible.<br>- Identification des requêtes ayant généré des réponses insuffisamment sourcées. | S3 / S4 |
| US-11 | **Cache sémantique Redis** — En tant qu'architecte système, je veux réutiliser les réponses de questions juridiquement similaires afin de réduire les appels au LLM et améliorer les performances. | - Stockage des questions et réponses précédentes.<br>- Comparaison sémantique avec un seuil configurable.<br>- Réutilisation uniquement si les sources et le contexte restent valides.<br>- Invalidation configurable après mise à jour du corpus.<br>- Mesure du taux de cache hit et de la latence. | S4 |
| US-12 | **Évaluation continue du moteur RAG avec Ragas** — En tant que Data Scientist, je veux mesurer automatiquement la qualité des réponses juridiques afin de détecter les hallucinations et les erreurs de retrieval. | - Création d'un benchmark de questions juridiques commerciales.<br>- Calcul des métriques Faithfulness et Answer Relevance.<br>- Évaluation du contexte récupéré.<br>- Comparaison des résultats entre différentes configurations de retrieval.<br>- Enregistrement des résultats dans un rapport d'évaluation.<br>- Vérification humaine de la fidélité des références juridiques. | S3 / S6 |
| US-13 | **Recherche de décisions similaires** — En tant qu'avocat, je veux retrouver des décisions présentant des faits ou des problématiques juridiques similaires à mon affaire afin d'enrichir mon analyse. | - Recherche sémantique sur les décisions indexées.<br>- Filtrage par domaine juridique et période.<br>- Présentation des décisions similaires avec leurs références.<br>- Justification des ressemblances par des passages sources.<br>- Évaluation sur un ensemble de cas annotés. | S5 / S6 |

---

# 3. COULD HAVE (Différenciateurs M2 Optionnels)

Ces fonctionnalités constituent des extensions avancées qui pourront être développées après la validation du moteur RAG principal.

| ID | User Story Technique | Critères d'Acceptation |
|---|---|---|
| US-14 | **GraphRAG juridique avec Neo4j** — En tant que chercheur juridique, je veux explorer les relations entre décisions, articles de loi et notions juridiques afin d'effectuer des recherches multi-relations. | - Création d'un graphe reliant les décisions, les articles de loi et les concepts juridiques.<br>- Extraction des entités et relations.<br>- Stockage dans Neo4j.<br>- Recherche de relations entre plusieurs décisions et références légales. |
| US-15 | **Routage agentique multi-outils (Query Routing)** — En tant qu'utilisateur, je veux que le système sélectionne automatiquement la stratégie adaptée à ma question juridique. | - Classification des questions.<br>- Routage vers Hybrid RAG, GraphRAG ou recherche documentaire.<br>- Sélection dynamique des outils.<br>- Traçage du chemin de traitement choisi. |
| US-16 | **Support multimodal (PDF, images et tableaux)** — En tant que Data Engineer, je veux traiter les documents numérisés et les éléments visuels des décisions afin de réduire les pertes d'information. | - OCR des PDF numérisés.<br>- Conservation des références de pages.<br>- Extraction des tableaux lorsqu'ils existent.<br>- Vérification de la qualité du texte extrait. |
| US-17 | **Analyse comparative de jurisprudence** — En tant qu'avocat, je veux comparer plusieurs décisions sur une même question juridique afin d'identifier les similitudes, les différences et les évolutions du raisonnement de la Cour. | - Sélection de plusieurs décisions.<br>- Extraction des questions juridiques et des solutions.<br>- Comparaison des raisonnements et des références légales.<br>- Citations précises pour chaque élément comparé. |

---

# 4. WON'T HAVE (Exclus du Périmètre Projet)

Les fonctionnalités suivantes sont explicitement exclues afin de préserver la faisabilité du projet fédérateur S7.

- Entraînement complet d'un LLM from scratch ou pré-entraînement de modèles de grande taille.
- Développement d'un système de prédiction des décisions judiciaires ou de l'issue d'un procès.
- Génération automatique de consultations juridiques définitives sans validation humaine.
- Gestion complète d'un cabinet d'avocats (facturation, dossiers clients, comptabilité).
- Marketplace juridique et mise en relation entre avocats et clients.
- Système complet de facturation SaaS et de paiement en ligne.
- Application mobile native iOS / Android.
- Déploiement Kubernetes complet, sauf si le temps et les ressources le permettent.

---

# 5. Priorisation et stratégie de livraison

**Priorité de développement :**

1. Collecter et constituer un corpus fiable de jurisprudence commerciale.
2. Extraire, nettoyer et structurer les décisions.
3. Développer l'indexation et le Hybrid Retrieval.
4. Ajouter le Reranking.
5. Exposer le moteur via FastAPI.
6. Développer la génération avec citations et streaming SSE.
7. Construire l'interface Web.
8. Tester, évaluer et conteneuriser l'application.

**Principe de validation :**

Chaque fonctionnalité MUST HAVE devra être testée et validée avant de passer à la fonctionnalité suivante.

Le projet privilégiera la précision du retrieval, la traçabilité des sources et la fiabilité juridique des réponses plutôt que l'accumulation de fonctionnalités avancées.
