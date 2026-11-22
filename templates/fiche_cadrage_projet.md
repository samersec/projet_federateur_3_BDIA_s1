# Fiche de Cadrage d'Ingénierie — Projet Fédérateur AI-BD_BI
## Spécification du Moteur RAG / Data Intelligent

---

# 1. Fiche d'Identité du Projet

- **Nom du projet :** TuniCase AI – Advanced RAG for Tunisian Commercial Case Law
- **Domaine :** Intelligence artificielle juridique (LegalTech), jurisprudence commerciale tunisienne.
- **Membres de l'équipe :**
  1. Mohamed Amine Ouelhazi — Développement IA / Advanced RAG
  2. Samer Soltani — Développement Data Engineering / Backend
  3. Sirine Rahal — Analyse juridique / Data Processing / Évaluation

  *Répartition provisoire des responsabilités, à valider collectivement.*
- **Lien du dépôt GitHub :** À définir.
- **Branche de travail principale :** `main` (protégée).

---

# 2. Cadrage du Problème Métier (Problem-First)

## 2.1 Persona Cible & Contexte Opérationnel

- **Utilisateur visé :**
  - Avocats spécialisés en droit commercial.
  - Juristes d'entreprise.
  - Magistrats et chercheurs en droit.
  - Étudiants et enseignants en droit.

- **Point de douleur majeur (Pain Point) :**

  En Tunisie, la recherche et l'exploitation de la jurisprudence commerciale nécessitent souvent de consulter manuellement un grand nombre de décisions judiciaires, notamment celles rendues par la Cour de cassation (محكمة التعقيب).

  Les professionnels du droit rencontrent plusieurs difficultés :
  - Retrouver rapidement une décision à partir d'une question juridique.
  - Identifier des décisions similaires portant sur une même problématique.
  - Comprendre le raisonnement juridique adopté par la Cour.
  - Identifier les articles de loi et les principes juridiques cités.
  - Comparer plusieurs décisions traitant d'un même sujet.
  - Exploiter des documents PDF contenant du texte arabe, parfois numérisés ou mal structurés.

- **Limites des approches classiques :**

  Une recherche classique par mots-clés ou Ctrl+F ne suffit pas toujours pour retrouver les décisions pertinentes.

  Par exemple, un avocat peut poser la question suivante :

  « Dans quelles conditions la Cour de cassation tunisienne considère-t-elle qu'une rupture contractuelle constitue une faute engageant la responsabilité commerciale ? »

  Une recherche textuelle classique risque de manquer des décisions qui utilisent des formulations juridiques différentes pour exprimer le même principe.

  De plus, les moteurs de recherche traditionnels ne permettent pas nécessairement de croiser les faits, les moyens du pourvoi, le raisonnement de la Cour et les textes juridiques cités.

## 2.2 Proposition de Valeur

Développer une plateforme intelligente spécialisée dans la jurisprudence commerciale tunisienne, basée sur une architecture Advanced RAG.

La solution permettra aux utilisateurs de :

- Effectuer des recherches en langage naturel en arabe ou en français.
- Retrouver les décisions de la Cour de cassation pertinentes.
- Identifier les décisions similaires sur une problématique juridique.
- Extraire les faits, les questions juridiques, les arguments et les solutions retenues.
- Identifier les articles de loi mentionnés dans les décisions.
- Obtenir des synthèses juridiques avec des références vérifiables.
- Consulter les passages originaux et les pages des décisions utilisées pour construire chaque réponse.

**Objectif principal :** fournir des réponses juridiques documentées et traçables, sans remplacer l'analyse ni le jugement du professionnel du droit.

---

# 3. Données & Volumétrie

## 3.1 Sources de données

**Source principale :**
- Cour de cassation tunisienne — site officiel : http://www.cassation.tn/
- Rubrique : فقه القضاء.
- Décisions judiciaires relatives aux litiges commerciaux.

**Périmètre juridique initial :**

1. Droit des sociétés :
   - Conflits entre associés.
   - Responsabilité des dirigeants.
   - Augmentation et réduction du capital.
   - Assemblées générales.

2. Contrats commerciaux :
   - Exécution et inexécution contractuelle.
   - Rupture des contrats.
   - Responsabilité contractuelle.
   - Preuve des contrats.

3. Fonds de commerce :
   - Cession du fonds de commerce.
   - Propriété commerciale.
   - Location-gérance.
   - Oppositions et contestations.

4. Effets de commerce et obligations :
   - Chèques.
   - Lettres de change.
   - Recouvrement des créances.
   - Preuve des obligations commerciales.

5. Procédure commerciale :
   - Compétence des chambres commerciales.
   - Conditions du pourvoi en cassation.
   - Vices de procédure.
   - Motifs de cassation et de rejet.

**Sources complémentaires potentielles :**
- Code de commerce tunisien.
- Code des sociétés commerciales.
- Textes législatifs et réglementaires tunisiens.
- Publications juridiques officielles accessibles légalement.

## 3.2 Format des données sources

- [x] PDF natifs.
- [x] PDF numérisés / Scans (selon disponibilité).
- [x] Texte brut extrait des décisions.
- [x] JSON pour les métadonnées et les fiches juridiques.
- [x] Markdown pour la documentation et les résultats structurés.
- [ ] CSV / Excel pour les jeux de données d'évaluation.
- [ ] Tables relationnelles SQL pour les métadonnées structurées.

## 3.3 Volumétrie

- **PoC initial (S1–S3) :** 30 à 50 décisions commerciales, à titre d'objectif initial.
- **Corpus cible du projet :** 200 à 500 décisions commerciales, selon la disponibilité et la qualité des documents.
- **Objectif à long terme :** extension progressive à l'ensemble des décisions commerciales accessibles et exploitables.

La volumétrie réelle sera déterminée après l'exploration du site et l'évaluation de la qualité des documents.

## 3.4 Fréquence de mise à jour

- [x] Collecte Batch.
- [ ] Streaming / Temps réel.
- [x] Mise à jour périodique du corpus selon la publication de nouvelles décisions.

---

# 4. Architecture Technique & Composants RAG

## 4.1 Pipeline d'ingestion documentaire

1. Collecte automatisée des décisions depuis le site officiel de la Cour de cassation.
2. Téléchargement et archivage des fichiers PDF.
3. Extraction du texte avec conservation des références de pages.
4. OCR pour les documents numérisés lorsque nécessaire.
5. Nettoyage et normalisation du texte arabe et français.
6. Extraction des métadonnées juridiques.
7. Structuration des décisions en fiches juridiques JSON.
8. Indexation dans les systèmes de stockage.

**Outils envisagés :** Python, BeautifulSoup, Requests, PyMuPDF, Docling et Tesseract OCR selon les formats rencontrés.

## 4.2 Stratégie de Chunking

- [ ] Recursive Character Chunking.
- [x] Semantic Chunking.
- [x] Parent-Document / Hierarchical Chunking.

**Stratégie retenue :**

Utiliser un découpage sémantique adapté à la structure des décisions judiciaires.

Les segments devront préserver les parties juridiquement importantes :
- Faits et procédure.
- Moyens de cassation.
- Question juridique.
- Raisonnement de la Cour.
- Solution de la Cour.
- Articles de loi et références.

**Paramètres initiaux à expérimenter :**
- Taille d'un chunk : 400 à 700 tokens.
- Overlap : 50 à 100 tokens.
- Conservation du numéro de décision, de la date, de la page et de la section juridique dans les métadonnées.

Les paramètres seront ajustés à partir des tests de retrieval.

## 4.3 Modèle d'Embeddings

- **Modèle envisagé :** BAAI/bge-m3.
- **Dimension :** 1024.
- **Justification :**

  BGE-M3 prend en charge les représentations multilingues, ce qui est important pour un corpus tunisien contenant des décisions en arabe et potentiellement en français.

  Il permet également d'explorer des stratégies de retrieval dense et sparse.

## 4.4 Base Vectorielle

- [x] Qdrant.
- [ ] Milvus.
- [ ] pgvector.

**Justification du choix :**

Qdrant permet :
- Le stockage des embeddings.
- La recherche vectorielle.
- Le filtrage par métadonnées juridiques.
- La recherche par numéro, date, domaine et type de décision.
- L'intégration avec les pipelines Python et FastAPI.

**Collections envisagées :**
- `commercial_decisions`
- `legal_texts`
- `legal_articles`

## 4.5 Retrieval & Reranking

**Stratégie : Hybrid RAG**

1. Dense Retrieval : recherche sémantique avec BGE-M3 et Qdrant.
2. Sparse Retrieval : recherche lexicale BM25 pour retrouver les termes juridiques précis.
3. Fusion des résultats : Reciprocal Rank Fusion (RRF).
4. Reranking : classement des passages avec un Cross-Encoder multilingue.

**Reranker envisagé :** BAAI/bge-reranker-v2-m3.

## 4.6 Extensions Advanced RAG

- **GraphRAG :** utilisation éventuelle de Neo4j pour représenter les relations entre décisions, articles de loi, notions juridiques et parties anonymisées.
- **Multimodal RAG :** traitement des PDF numérisés et des éléments visuels des documents lorsque nécessaire.
- **Agentic Routing :** routage des questions selon leur nature :
  - Recherche d'une décision.
  - Recherche d'un article juridique.
  - Recherche de décisions similaires.
  - Analyse comparative de plusieurs décisions.

Ces extensions seront développées progressivement après validation du Hybrid RAG.

## 4.7 LLM de Génération

- **Modèle envisagé :** modèle multilingue compatible avec le traitement de l'arabe et du français.
- **Options à évaluer :** Qwen, Mistral ou un modèle accessible via API.
- **Critères de sélection :**
  - Compréhension du vocabulaire juridique arabe.
  - Capacité à analyser des textes longs.
  - Respect des instructions.
  - Qualité des réponses en arabe et en français.
  - Coût et latence.

Le modèle définitif sera choisi après des tests comparatifs.

## 4.8 Backend & Frontend

- **Backend :** FastAPI asynchrone.
- **Validation des données :** Pydantic v2.
- **Frontend :** Next.js.
- **Streaming :** Server-Sent Events (SSE).
- **Base relationnelle :** PostgreSQL pour les métadonnées et les évaluations.
- **Déploiement :** Docker et Docker Compose.

---

# 5. Métriques Quantitatives & SLAs

Les objectifs suivants sont des seuils initiaux de validation du prototype, à ajuster selon les résultats expérimentaux.

| Métrique | Définition / Outil | Objectif cible |
|---|---|---:|
| Recall@10 | Présence d'une décision pertinente dans les 10 premiers résultats | ≥ 85 % |
| Precision@5 | Proportion de résultats pertinents dans le Top 5 | ≥ 80 % |
| MRR@10 | Position du premier résultat pertinent | ≥ 0,75 |
| Faithfulness | Fidélité de la réponse aux passages récupérés (Ragas + validation humaine) | ≥ 0,90 |
| Answer Relevance | Pertinence de la réponse par rapport à la question | ≥ 0,85 |
| Citation Accuracy | Exactitude des références de décisions et de pages | ≥ 95 % |
| Latence First-Token (TTFT) | Temps avant le premier token affiché | À mesurer |
| Latence E2E P95 | Temps total de réponse au 95e percentile | À mesurer |

**Évaluation :**

Un jeu de questions juridiques de référence sera construit à partir des décisions du corpus.

Chaque question sera associée à :
- Une ou plusieurs décisions pertinentes.
- Les passages justificatifs.
- Les pages sources.
- Une réponse de référence validée.

L'évaluation combinera des métriques automatiques et une vérification humaine de la qualité juridique.

---

# 6. Gestion des Risques & Garde-fous

## 6.1 Prévention des hallucinations

Instruction intégrée au prompt système :

« Répondez uniquement à partir des documents et passages juridiques fournis. Chaque affirmation juridique doit être accompagnée d'une référence vérifiable. Si l'information n'est pas présente dans les extraits récupérés, indiquez explicitement que le corpus disponible ne permet pas de conclure. Ne créez jamais de numéro de décision, de date, d'article de loi ou de citation juridique. »

## 6.2 Fiabilité des sources

- Conserver les PDF originaux.
- Conserver les URL officielles.
- Associer chaque chunk à son document et à sa page.
- Distinguer les arguments des parties du raisonnement de la Cour.
- Ne pas présenter une synthèse générée comme une citation textuelle.
- Signaler les documents incomplets ou dont l'OCR est incertain.

## 6.3 Protection des données

- Utilisation de variables d'environnement via `.env`.
- Aucun token ou secret dans le dépôt GitHub.
- Protection des documents et des données sensibles éventuellement présents dans les décisions.
- Anonymisation lorsque nécessaire.
- Respect des conditions d'accès, d'utilisation et de réutilisation des documents.

## 6.4 Isolation et sécurité

- Conteneurisation des services avec Docker.
- Isolation réseau de Qdrant et PostgreSQL.
- Authentification des endpoints API.
- Gestion des erreurs et journalisation des requêtes.
- Contrôle des accès aux documents.

## 6.5 Limites juridiques

La plateforme est un outil d'aide à la recherche et à l'analyse documentaire. Elle ne remplace pas l'interprétation du droit par un professionnel qualifié et ne doit pas présenter les réponses générées comme des consultations juridiques définitives.