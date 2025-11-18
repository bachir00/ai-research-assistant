# 📚 AI Research Assistant - Documentation Complète

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-enabled-orange.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vectorstore-purple.svg)

## 🎯 Vue d'ensemble

**AI Research Assistant** est un système intelligent de recherche et d'analyse documentaire utilisant LangGraph, plusieurs agents IA spécialisés, et un système de mémoire vectorielle avancé. Le système automatise l'ensemble du processus de recherche : de la collecte d'informations sur le web jusqu'à la génération de rapports de synthèse structurés.

### ✨ Fonctionnalités principales

- 🔍 **Recherche web automatisée** avec extraction de mots-clés intelligente
- 📄 **Extraction de contenu** depuis des pages web avec parsing avancé
- 📝 **Génération de résumés** détaillés et structurés
- 🎯 **Synthèse globale** avec analyse thématique transversale
- 💾 **Système de mémoire** vectorielle et conversationnelle
- 🤖 **Orchestration par LLM** via LangGraph
- 🚫 **Déduplication automatique** des documents
- ⚡ **Cache intelligent** avec TTL configurable

---

## 🏗️ Architecture du Projet

### Structure des dossiers

```
langGraphe-ai-research-assistant-main/
│
├── config/                      # Configuration globale
│   ├── settings.py             # Paramètres de l'application
│   └── prompts.py              # Templates de prompts
│
├── src/                         # Code source principal
│   ├── agents/                 # Agents spécialisés
│   │   ├── base_agent.py       # Agent de base
│   │   ├── researcher_agent.py # Recherche web
│   │   ├── content_extractor_agent.py # Extraction
│   │   ├── summarizer_agent.py # Résumés
│   │   └── global_synthesizer_agent.py # Synthèse
│   │
│   ├── services/               # Services partagés
│   │   ├── search_api.py      # APIs de recherche (Tavily, Serper)
│   │   ├── content_extraction.py # Extraction de contenu web
│   │   ├── llm_service.py     # Service LLM (Groq)
│   │   └── text_chunking.py   # Découpage de texte
│   │
│   ├── models/                 # Modèles de données
│   │   ├── research_models.py  # Modèles de recherche
│   │   ├── document_models.py  # Modèles de documents
│   │   ├── synthesis_models.py # Modèles de synthèse
│   │   ├── report_models.py    # Modèles de rapports
│   │   └── state_models.py     # États du graphe
│   │
│   ├── graph/                  # LangGraph
│   │   └── nodes.py           # Nœuds du graphe
│   │
│   ├── core/                   # Fonctionnalités de base
│   │   └── logging.py         # Configuration des logs
│   │
│   ├── memory_system.py        # Système de mémoire vectorielle
│   ├── memory_integration.py   # Intégration de la mémoire
│   ├── enhanced_system_prompt.py # Prompts avancés
│   └── graph.py                # Graphe LangGraph principal
│
├── tests/                       # Tests unitaires et d'intégration
│   ├── test_researcher.py
│   ├── test_content_extractor_agent.py
│   ├── test_summarizer_agent.py
│   └── api_tests.py
│
├── logs/                        # Fichiers de logs
├── .env                         # Variables d'environnement
├── requirements.txt             # Dépendances Python
└── README.md                    # Documentation principale
```

---

## 🔧 Architecture Technique

### Diagramme du Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         UTILISATEUR                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM ORCHESTRATEUR                            │
│              (ChatGroq avec LangGraph)                          │
│  • Analyse la requête utilisateur                               │
│  • Décide des outils à utiliser                                 │
│  • Gère le flow de conversation                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
┌─────────────────────┐       ┌─────────────────────┐
│   RECHERCHE CACHE   │       │  NOUVELLE RECHERCHE │
│                     │       │                     │
│ • Vérif. cache 24h  │       │ • Pipeline complet  │
│ • Recherche mémoire │       │ • 4 agents séquence │
│ • Topics similaires │       │ • Stockage mémoire  │
└──────────┬──────────┘       └──────────┬──────────┘
           │                             │
           │            ┌────────────────┘
           │            │
           ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTÈME DE MÉMOIRE                           │
│                                                                 │
│  ┌─────────────────────┐         ┌──────────────────────┐       │
│  │  MÉMOIRE VECTORIELLE│         │   MÉMOIRE AGENT      │       │
│  │  (ChromaDB)         │         │ (Cache + Historique) │       │
│  │                     │         │                      │       │
│  │ • Embeddings        │         │ • Conversations      │       │
│  │ • Recherche top-k   │         │ • Cache recherches   │       │
│  │ • Déduplication     │◄────────┤ • Topics + keywords  │       │
│  │ • Persistance       │         │ • Compression auto   │       │
│  └─────────────────────┘         └──────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   PIPELINE DE RECHERCHE                          │
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐            │
│  │   AGENT 1   │   │   AGENT 2   │   │   AGENT 3    │            │
│  │ RESEARCHER  │──►│  EXTRACTOR  │──►│ SUMMARIZER   │            │
│  │             │   │             │   │              │            │
│  │ • Tavily    │   │ • Parsing   │   │ • LLM        │            │
│  │ • Serper    │   │ • Nettoyage │   │ • Chunking   │            │
│  │ • Keywords  │   │ • Validation│   │ • Points-clés│            │
│  └─────────────┘   └─────────────┘   └──────┬───────┘            │
│                                             │                    │
│                                             ▼                    │
│                                    ┌──────────────────┐          │
│                                    │     AGENT 4      │          │
│                                    │GLOBAL SYNTHESIZER│          │
│                                    │                  │          │
│                                    │ • Thèmes         │          │
│                                    │ • Consensus      │          │
│                                    │ • Rapport final  │          │
│                                    └──────────────────┘          │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                   RAPPORT STRUCTURÉ
              (Markdown, HTML, Text, JSON)
```

---

## 🤖 Description des Agents

### 1. 🔍 ResearcherAgent
**Rôle** : Recherche web et extraction de mots-clés

**Responsabilités** :
- Extraction automatique de mots-clés via LLM
- Recherche multi-API (Tavily, Serper)
- Filtrage et validation des résultats
- Gestion du rate limiting

**Inputs** :
```python
ResearchQuery(
    topic: str,
    keywords: List[str],
    max_results: int = 10,
    search_depth: str = "basic"
)
```

**Outputs** :
```python
ResearchOutput(
    results: List[SearchResult],
    total_found: int,
    search_engine: str,
    search_time: float
)
```

### 2. 📄 ContentExtractorAgent
**Rôle** : Extraction et nettoyage du contenu web

**Responsabilités** :
- Parsing HTML avec BeautifulSoup
- Nettoyage et normalisation du texte
- Détection du type de document
- Validation de la qualité

**Inputs** :
```python
ResearchOutput  # Provient du ResearcherAgent
```

**Outputs** :
```python
ExtractionResult(
    documents: List[Document],
    successful_extractions: int,
    failed_urls: List[str],
    extraction_stats: Dict
)
```

### 3. 📝 SummarizerAgent
**Rôle** : Génération de résumés détaillés

**Responsabilités** :
- Découpage intelligent du texte (chunking)
- Résumés exécutifs et détaillés
- Extraction de points-clés et arguments
- Analyse de sentiment et crédibilité

**Inputs** :
```python
ExtractionResult  # Provient du ContentExtractorAgent
```

**Outputs** :
```python
SummarizationOutput(
    summaries: List[DocumentSummary],
    total_documents: int,
    average_credibility: float,
    common_themes: List[str]
)
```

### 4. 🎯 GlobalSynthesizerAgent
**Rôle** : Synthèse globale et génération de rapport

**Responsabilités** :
- Analyse thématique transversale
- Identification de consensus et conflits
- Génération de rapport structuré
- Export multi-format (Markdown, HTML, Text)

**Inputs** :
```python
SummarizationOutput  # Provient du SummarizerAgent
```

**Outputs** :
```python
GlobalSynthesisOutput(
    final_report: FinalReport,
    synthesis_metadata: Dict,
    processing_stats: Dict,
    formatted_outputs: Dict[str, str]
)
```

---

## 💾 Système de Mémoire

### Architecture de la Mémoire

Le système utilise **deux types de mémoire complémentaires** :

#### 1. 🗄️ Mémoire Vectorielle (ChromaDB)

```python
VectorMemoryManager(
    persist_directory="./chroma_db",
    collection_name="research_documents",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Fonctionnalités** :
- **Embeddings** : Modèles HuggingFace pour représentation vectorielle
- **Recherche sémantique** : Top-K avec scores de similarité cosinus
- **Déduplication** : Hash MD5 pour éviter les doublons
- **Persistance** : Stockage permanent sur disque
- **Nettoyage auto** : Suppression des documents > 30 jours

**Méthodes principales** :
```python
# Ajout de documents
stats = vector_memory.add_documents(
    documents=[{
        'content': "...",
        'title': "...",
        'url': "..."
    }],
    source='research',
    check_duplicates=True
)

# Recherche sémantique
results = vector_memory.semantic_search(
    query="intelligence artificielle",
    k=5,
    filter_dict={'source': 'research'}
)

# Nettoyage
deleted = vector_memory.clear_old_documents(days=30)
```

#### 2. 🧠 Mémoire d'Agent (Cache + Historique)

```python
AgentMemoryManager(
    memory_file="./agent_memory.pkl",
    max_history=100,
    compression_threshold=50
)
```

**Fonctionnalités** :
- **Historique conversationnel** : Deque avec limite (100 entrées)
- **Cache des recherches** : TTL 24h par défaut
- **Keywords tracking** : Association topic → keywords
- **Compression auto** : Après 50 entrées
- **Persistance pickle** : Sauvegarde sur disque

**Méthodes principales** :
```python
# Ajouter une conversation
agent_memory.add_conversation(
    user_message="Résume l'IA",
    assistant_response="...",
    metadata={'sources': 5}
)

# Récupérer du cache
result = agent_memory.get_research_result(
    topic="intelligence artificielle",
    max_age_hours=24
)

# Topics similaires
related = agent_memory.get_related_topics(
    topic="IA dans la santé",
    threshold=0.5
)
```

### 🔗 Système Intégré

```python
IntegratedMemorySystem()
```

Combine les deux mémoires pour :
- Stockage automatique de tous les résultats de recherche
- Récupération intelligente du contexte
- Vérification du cache avant nouvelle recherche
- Enrichissement des réponses avec contexte historique

---

## 🛠️ Installation

### Prérequis

- **Python** : 3.12+
- **Pip** : version récente
- **Git** : pour cloner le projet

### Étapes d'installation

```bash
# 1. Cloner le projet
git clone https://github.com/votre-repo/ai-research-assistant.git
cd ai-research-assistant

# 2. Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API
```

### Configuration `.env`

```env
# LLM
GROQ_API_KEY=your_groq_api_key_here

# Search APIs
TAVILY_API_KEY=your_tavily_api_key_here
SERPER_API_KEY=your_serper_api_key_here

# Optional
LOG_LEVEL=INFO
MAX_RETRIES=3
TIMEOUT=30
```

---
############################################################################
## 🚀 Utilisation

### Mode CLI Direct

```bash
# Recherche simple
python src/graph.py "impact de l'IA sur l'emploi"

# Mode test
python src/graph.py test

# Statistiques mémoire
python src/graph.py stats
```

### Mode Interactif

```python
from src.graph import app_with_memory, run_test

# Lancer une recherche
run_test("Résume les énergies renouvelables", "Ma recherche")

# Ou utiliser directement le graphe
inputs = {"messages": [("user", "Résume l'IA dans la santé")]}
for state in app_with_memory.stream(inputs, stream_mode="values"):
    print(state["messages"][-1])
```

### Mode Menu Interactif

```bash
python tests/test_memory_system.py
```

Menu disponible :
```
1. Poser une question / Lancer une recherche
2. Rechercher dans la mémoire
3. Voir l'historique
4. Statistiques de la mémoire
5. Lancer la suite de tests
6. Réinitialiser la mémoire
0. Quitter
```

### Intégration dans votre code

```python

from src.agents.researcher_agent import ResearcherAgent
from src.agents.content_extractor_agent import ContentExtractorAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.global_synthesizer_agent import GlobalSynthesizerAgent
from src.models.research_models import ResearchQuery

# Initialiser les agents
researcher = ResearcherAgent()
extractor = ContentExtractorAgent()
summarizer = SummarizerAgent()
synthesizer = GlobalSynthesizerAgent()

# Pipeline complet
async def recherche_complete(topic: str):
    # 1. Recherche
    query = ResearchQuery(
        topic=topic,
        keywords=await researcher.extract_keywords_with_llm(topic),
        max_results=5
    )
    research_data = await researcher.process(query)
    
    # 2. Extraction
    extraction_data = await extractor.process_from_research_output(
        research_output=research_data
    )
    
    # 3. Résumés
    summarization_data = await summarizer.process_from_extraction_result(
        extraction_result=extraction_data
    )
    
    # 4. Synthèse
    synthesis = await synthesizer.process_from_summarization_output(
        summarization_output=summarization_data
    )
    
    return synthesis.final_report.formatted_outputs['markdown']
```

---

## 📊 Exemples d'Utilisation

### Exemple 1 : Recherche Simple avec Cache

```python
# Première recherche (pipeline complet)
inputs = {
    "messages": [
        ("user", "Résume l'impact de l'IA sur le marché du travail")
    ]
}

for state in app_with_memory.stream(inputs):
    print(state["messages"][-1].content)

# Résultat : Pipeline complet exécuté, résultats mis en cache

# Même recherche 10 minutes après (utilise le cache)
inputs = {
    "messages": [
        ("user", "Rappelle-moi ce que tu as trouvé sur l'IA et l'emploi")
    ]
}

for state in app_with_memory.stream(inputs):
    print(state["messages"][-1].content)

# Résultat : Réponse instantanée depuis le cache
```

### Exemple 2 : Recherche dans la Mémoire

```python
# Après plusieurs recherches sur l'IA
inputs = {
    "messages": [
        ("user", "Qu'as-tu trouvé sur l'intelligence artificielle ?")
    ]
}

# Le LLM utilise automatiquement search_in_memory
# au lieu de lancer une nouvelle recherche web
```

### Exemple 3 : Historique et Statistiques

```python
from src.memory_system import memory_system

# Voir l'historique
history = list(memory_system.agent_memory.conversation_history)
for conv in history[-5:]:
    print(f"{conv['timestamp']}: {conv['user']}")

# Statistiques
print(f"Documents en mémoire: {memory_system.vector_memory.collection.count()}")
print(f"Recherches en cache: {len(memory_system.agent_memory.research_cache)}")
```

### Exemple 4 : Recherche Approfondie

```python
from src.memory_integration import research_complete_pipeline_with_memory

# Recherche avec plus de sources
result = research_complete_pipeline_with_memory(
    topic="énergies renouvelables et transition écologique",
    max_results=10,  # Plus de sources
    use_cache=False  # Forcer une nouvelle recherche
)

print(result)  # Rapport Markdown complet
```

---


## 📝 Logs et Monitoring

### Structure des logs

```
logs/
├── agent_researcher.log          # Recherche web
├── agent_content_extractor.log   # Extraction
├── agent_summarizer.log          # Résumés
├── agent_global_synthesizer.log  # Synthèse
├── search_manager.log            # APIs de recherche
├── llm_service.log               # Appels LLM
└── complete_pipeline.log         # Pipeline complet
```

### Niveaux de log

```python
# Dans config/settings.py
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Exemple de logs

```
2025-11-18 10:30:15 | INFO | agent_researcher | Recherche pour: "IA emploi"
2025-11-18 10:30:18 | INFO | agent_researcher | Trouvé 5 sources
2025-11-18 10:30:20 | INFO | agent_content_extractor | Extraction: 5/5 succès
2025-11-18 10:30:45 | INFO | agent_summarizer | 5 résumés générés
2025-11-18 10:31:10 | INFO | agent_global_synthesizer | Rapport: 1250 mots
2025-11-18 10:31:12 | INFO | memory_system | Stockage en mémoire réussi
```

---

## ⚙️ Configuration Avancée

### Personnaliser les prompts

```python
# config/prompts.py

CUSTOM_RESEARCH_PROMPT = """
Analyse approfondie sur {topic}.
Focus sur les aspects suivants :
- Impact économique
- Implications sociales
- Perspectives futures
"""

# Utilisation
from config.prompts import CUSTOM_RESEARCH_PROMPT

prompt = CUSTOM_RESEARCH_PROMPT.format(topic="IA générative")
```

### Ajuster les paramètres LLM

```python
# src/services/llm_service.py

class LLMService:
    def __init__(self):
        self.model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,      # Créativité (0-1)
            max_tokens=2048,      # Longueur max
            top_p=0.9,           # Nucleus sampling
            frequency_penalty=0.5 # Pénalité répétition
        )
```

### Configurer la mémoire vectorielle

```python
# src/memory_system.py

vector_memory = VectorMemoryManager(
    persist_directory="./custom_chroma_db",
    collection_name="my_research_docs",
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingue
)
```

---

## 🔌 Intégration API(Futur)

### REST API (FastAPI)

```python
# api/main.py
from fastapi import FastAPI
from src.graph import app_with_memory

app = FastAPI()

@app.post("/research")
async def research_endpoint(topic: str, max_results: int = 3):
    inputs = {"messages": [("user", f"Résume: {topic}")]}
    result = []
    
    for state in app_with_memory.stream(inputs):
        result.append(state["messages"][-1].content)
    
    return {"result": result[-1]}
```

### WebSocket (temps réel)

```python
from fastapi import WebSocket

@app.websocket("/ws/research")
async def websocket_research(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        data = await websocket.receive_text()
        inputs = {"messages": [("user", data)]}
        
        for state in app_with_memory.stream(inputs):
            await websocket.send_text(
                state["messages"][-1].content
            )
```

---

## 🐛 Dépannage

### Problèmes courants

#### 1. Erreur de clé API manquante

```
ValueError: GROQ_API_KEY non définie
```

**Solution** : Vérifier le fichier `.env` et s'assurer que les clés sont présentes.


#### 3. Rate limit atteint

```
WARNING | llm_service | Rate limit atteint, attente 12s
```

**Solution** : C'est normal, le système attend automatiquement. Pour éviter :
- Réduire `max_results`

#### 4. Mémoire saturée

```
MemoryError: Cannot allocate memory
```

**Solution** : Nettoyer la mémoire :
```
memory_system.vector_memory.clear_old_documents(days=7)
```

---

```bash
# Build
docker build -t ai-research-assistant .

# Run
docker run -e GROQ_API_KEY=xxx -e TAVILY_API_KEY=yyy ai-research-assistant
```

### Production (Gunicorn)

```bash
gunicorn api.main:app --workers 4 --bind 0.0.0.0:8000
```

---

## 📈 Roadmap

### Version 1.1 (En cours)
- [ ] Interface web avec Streamlit
- [ ] Support multilingue complet
- [ ] Export PDF des rapports
- [ ] Notifications par email

### Version 2.0 (Futur)
- [ ] Agents spécialisés par domaine (santé, finance, tech)
- [ ] Intégration avec bases de données externes
- [ ] Système de fact-checking automatique
- [ ] API GraphQL

---

## 🤝 Contribution

Les contributions sont les bienvenues !

---

## 👥 Auteurs

- **Bachir** - *Développeur Principal* - [GitHub](https://github.com/bachir00)

---

## 🙏 Remerciements

- LangChain & LangGraph pour le framework
- Groq pour l'accès aux LLMs
- ChromaDB pour le stockage vectoriel
- Tavily & Serper pour les APIs de recherche
- La communauté open-source

---

## 📞 Support

- 📧 Email : bassiroukane@esp.sn