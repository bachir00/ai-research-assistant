"""
Système de Mémoire et Stockage Vectoriel pour l'Assistant de Recherche
Gère : embeddings, recherche sémantique, historique et déduplication
"""

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
import json
import pickle
from pathlib import Path
from collections import deque

# ============================================================================
# GESTIONNAIRE DE MÉMOIRE VECTORIELLE
# ============================================================================

class VectorMemoryManager:
    """Gère le stockage vectoriel des documents et résumés"""
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "research_documents",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialise le gestionnaire de mémoire vectorielle
        
        Args:
            persist_directory: Dossier de persistance de ChromaDB
            collection_name: Nom de la collection ChromaDB
            embedding_model: Modèle d'embeddings HuggingFace
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Initialisation du système de mémoire vectorielle...")
        
        # Configuration des embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Configuration ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Créer ou récupérer la collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Collection '{collection_name}' récupérée ({self.collection.count()} documents)")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ Nouvelle collection '{collection_name}' créée")
        
        # Initialiser le vectorstore LangChain
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Cache pour déduplication rapide
        self.content_hashes = set()
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Charge les hashes des documents existants pour déduplication"""
        try:
            results = self.collection.get(include=['metadatas'])
            for metadata in results['metadatas']:
                if 'content_hash' in metadata:
                    self.content_hashes.add(metadata['content_hash'])
            print(f"📋 {len(self.content_hashes)} hashes chargés pour déduplication")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des hashes: {e}")
    
    def _compute_hash(self, content: str) -> str:
        """Calcule le hash MD5 d'un contenu"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, content: str) -> bool:
        """Vérifie si un document est un doublon"""
        content_hash = self._compute_hash(content)
        return content_hash in self.content_hashes
    
    def add_documents(self, 
                     documents: List[Dict[str, any]], 
                     source: str = "research",
                     check_duplicates: bool = True) -> Dict[str, int]:
        """
        Ajoute des documents au vectorstore
        
        Args:
            documents: Liste de dicts avec 'content', 'title', 'url', etc.
            source: Source des documents (research, summary, synthesis)
            check_duplicates: Vérifier les doublons avant ajout
        
        Returns:
            Dict avec statistiques d'ajout
        """
        print(f"\n📥 Ajout de {len(documents)} documents (source: {source})...")
        
        added = 0
        skipped = 0
        
        docs_to_add = []
        metadatas_to_add = []
        ids_to_add = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            # Vérification des doublons
            if check_duplicates and self.is_duplicate(content):
                skipped += 1
                continue
            
            # Création du document LangChain
            content_hash = self._compute_hash(content)
            doc_id = f"{source}_{content_hash[:8]}_{datetime.now().timestamp()}"
            
            metadata = {
                'title': doc.get('title', 'Sans titre'),
                'url': doc.get('url', ''),
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'content_hash': content_hash,
                'word_count': len(content.split())
            }
            
            docs_to_add.append(content)
            metadatas_to_add.append(metadata)
            ids_to_add.append(doc_id)
            self.content_hashes.add(content_hash)
            added += 1
        
        # Ajout batch à ChromaDB
        if docs_to_add:
            self.collection.add(
                documents=docs_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
        
        stats = {
            'added': added,
            'skipped': skipped,
            'total_in_db': self.collection.count()
        }
        
        print(f"✅ Ajoutés: {added} | Doublons ignorés: {skipped} | Total DB: {stats['total_in_db']}")
        return stats
    
    def semantic_search(self, 
                       query: str, 
                       k: int = 5,
                       filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Recherche sémantique dans le vectorstore
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats à retourner
            filter_dict: Filtres sur les métadonnées (ex: {'source': 'research'})
        
        Returns:
            Liste de tuples (Document, score)
        """
        print(f"\n🔍 Recherche sémantique: '{query}' (top-{k})")
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        print(f"✅ {len(results)} résultats trouvés")
        return results
    
    def get_relevant_context(self, 
                            query: str, 
                            k: int = 3,
                            source_filter: Optional[str] = None) -> str:
        """
        Récupère le contexte pertinent pour une requête
        
        Args:
            query: Requête
            k: Nombre de documents à récupérer
            source_filter: Filtrer par source (research, summary, etc.)
        
        Returns:
            Contexte formaté en string
        """
        filter_dict = {"source": source_filter} if source_filter else None
        results = self.semantic_search(query, k=k, filter_dict=filter_dict)
        
        if not results:
            return ""
        
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(
                f"[Source {i} - Pertinence: {score:.2f}]\n"
                f"Titre: {doc.metadata.get('title', 'N/A')}\n"
                f"{doc.page_content[:500]}...\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def clear_old_documents(self, days: int = 30) -> int:
        """
        Supprime les documents plus anciens que X jours
        
        Args:
            days: Nombre de jours de rétention
        
        Returns:
            Nombre de documents supprimés
        """
        print(f"\n🧹 Nettoyage des documents > {days} jours...")
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        results = self.collection.get(include=['metadatas'])
        ids_to_delete = []
        
        for doc_id, metadata in zip(results['ids'], results['metadatas']):
            timestamp_str = metadata.get('timestamp', '')
            try:
                doc_date = datetime.fromisoformat(timestamp_str)
                if doc_date < cutoff_date:
                    ids_to_delete.append(doc_id)
                    hash_to_remove = metadata.get('content_hash')
                    if hash_to_remove:
                        self.content_hashes.discard(hash_to_remove)
            except:
                continue
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        
        print(f"✅ {len(ids_to_delete)} documents supprimés")
        return len(ids_to_delete)


# ============================================================================
# GESTIONNAIRE DE MÉMOIRE D'AGENT
# ============================================================================

class AgentMemoryManager:
    """Gère l'historique des conversations et résumés"""
    
    def __init__(self, 
                 memory_file: str = "./agent_memory.pkl",
                 max_history: int = 100,
                 compression_threshold: int = 50):
        """
        Initialise le gestionnaire de mémoire d'agent
        
        Args:
            memory_file: Fichier de sauvegarde de la mémoire
            max_history: Nombre maximum d'entrées dans l'historique
            compression_threshold: Seuil pour compression de mémoire
        """
        self.memory_file = Path(memory_file)
        self.max_history = max_history
        self.compression_threshold = compression_threshold
        
        # Structures de données
        self.conversation_history = deque(maxlen=max_history)
        self.research_cache = {}  # topic -> result
        self.summary_cache = {}    # topic -> summary
        self.topic_keywords = {}   # topic -> keywords
        
        print(f"🧠 Initialisation du gestionnaire de mémoire d'agent...")
        self._load_memory()
    
    def _load_memory(self):
        """Charge la mémoire depuis le fichier"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.conversation_history = data.get('conversation_history', deque(maxlen=self.max_history))
                    self.research_cache = data.get('research_cache', {})
                    self.summary_cache = data.get('summary_cache', {})
                    self.topic_keywords = data.get('topic_keywords', {})
                print(f"✅ Mémoire chargée: {len(self.conversation_history)} conversations, "
                      f"{len(self.research_cache)} recherches en cache")
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de la mémoire: {e}")
        else:
            print("ℹ️ Nouvelle mémoire initialisée")
    
    def _save_memory(self):
        """Sauvegarde la mémoire dans le fichier"""
        try:
            data = {
                'conversation_history': self.conversation_history,
                'research_cache': self.research_cache,
                'summary_cache': self.summary_cache,
                'topic_keywords': self.topic_keywords
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde de la mémoire: {e}")
    
    def add_conversation(self, user_message: str, assistant_response: str, metadata: Optional[Dict] = None):
        """Ajoute une conversation à l'historique"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'assistant': assistant_response,
            'metadata': metadata or {}
        }
        self.conversation_history.append(entry)
        
        # Compression si nécessaire
        if len(self.conversation_history) >= self.compression_threshold:
            self._compress_memory()
        
        self._save_memory()
    
    def add_research_result(self, topic: str, result: any, keywords: List[str]):
        """Cache un résultat de recherche"""
        self.research_cache[topic] = {
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        self.topic_keywords[topic] = keywords
        self._save_memory()
    
    def get_research_result(self, topic: str, max_age_hours: int = 24) -> Optional[any]:
        """Récupère un résultat de recherche en cache"""
        if topic not in self.research_cache:
            return None
        
        cached = self.research_cache[topic]
        cached_time = datetime.fromisoformat(cached['timestamp'])
        
        from datetime import timedelta
        if datetime.now() - cached_time > timedelta(hours=max_age_hours):
            print(f"ℹ️ Cache expiré pour '{topic}'")
            return None
        
        print(f"✅ Résultat récupéré du cache pour '{topic}'")
        return cached['result']
    
    def add_summary(self, topic: str, summary: str):
        """Ajoute un résumé au cache"""
        self.summary_cache[topic] = {
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        self._save_memory()
    
    def get_conversation_context(self, n_last: int = 5) -> str:
        """Récupère le contexte des N dernières conversations"""
        recent = list(self.conversation_history)[-n_last:]
        
        if not recent:
            return ""
        
        context = "Contexte des conversations récentes:\n"
        for i, conv in enumerate(recent, 1):
            context += f"\n[Conversation {i}]\n"
            context += f"User: {conv['user'][:100]}...\n"
            context += f"Assistant: {conv['assistant'][:100]}...\n"
        
        return context
    
    def _compress_memory(self):
        """Compresse la mémoire en gardant seulement les éléments importants"""
        print("🗜️ Compression de la mémoire...")
        
        # Supprimer les anciennes recherches en cache (> 7 jours)
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=7)
        
        topics_to_remove = []
        for topic, data in self.research_cache.items():
            if datetime.fromisoformat(data['timestamp']) < cutoff:
                topics_to_remove.append(topic)
        
        for topic in topics_to_remove:
            del self.research_cache[topic]
            if topic in self.topic_keywords:
                del self.topic_keywords[topic]
        
        print(f"✅ {len(topics_to_remove)} anciennes recherches supprimées")
        self._save_memory()
    
    def get_related_topics(self, topic: str, threshold: float = 0.5) -> List[str]:
        """Trouve les topics similaires dans l'historique"""
        from difflib import SequenceMatcher
        
        related = []
        for cached_topic in self.research_cache.keys():
            similarity = SequenceMatcher(None, topic.lower(), cached_topic.lower()).ratio()
            if similarity > threshold:
                related.append((cached_topic, similarity))
        
        return [t for t, _ in sorted(related, key=lambda x: x[1], reverse=True)]
    
    def clear_all(self):
        """Réinitialise complètement la mémoire"""
        print("🗑️ Réinitialisation complète de la mémoire...")
        self.conversation_history.clear()
        self.research_cache.clear()
        self.summary_cache.clear()
        self.topic_keywords.clear()
        self._save_memory()
        print("✅ Mémoire réinitialisée")


# ============================================================================
# GESTIONNAIRE INTÉGRÉ
# ============================================================================

class IntegratedMemorySystem:
    """Système de mémoire intégré combinant vectoriel et agent"""
    
    def __init__(self):
        self.vector_memory = VectorMemoryManager()
        self.agent_memory = AgentMemoryManager()
        print("✨ Système de mémoire intégré initialisé\n")
    
    def process_research_result(self, 
                               topic: str, 
                               extraction_result: any,
                               summarization_result: any,
                               global_synthesis: any):
        """
        Traite et stocke tous les résultats d'une recherche
        
        Args:
            topic: Sujet de la recherche
            extraction_result: Résultat de l'extraction
            summarization_result: Résultat des résumés
            global_synthesis: Synthèse globale
        """
        print(f"\n💾 Stockage des résultats pour '{topic}'...")
        
        # 1. Stocker les documents extraits dans le vectorstore
        if extraction_result and hasattr(extraction_result, 'documents'):
            docs_to_store = []
            for doc in extraction_result.documents:
                docs_to_store.append({
                    'content': doc.content,
                    'title': doc.title,
                    'url': str(doc.url)
                })
            self.vector_memory.add_documents(docs_to_store, source='research')
        
        # 2. Stocker les résumés
        if summarization_result and hasattr(summarization_result, 'summaries'):
            summaries_to_store = []
            for summary in summarization_result.summaries:
                summaries_to_store.append({
                    'content': summary.detailed_summary,
                    'title': summary.title,
                    'url': str(summary.url)
                })
            self.vector_memory.add_documents(summaries_to_store, source='summary')
        
        # 3. Stocker la synthèse globale
        if global_synthesis and hasattr(global_synthesis, 'final_report'):
            synthesis_text = global_synthesis.final_report.formatted_outputs.get('text', '')
            self.vector_memory.add_documents([{
                'content': synthesis_text,
                'title': f"Synthèse: {topic}",
                'url': ''
            }], source='synthesis')
        
        # 4. Mettre en cache dans la mémoire agent
        keywords = []
        if hasattr(extraction_result, 'documents'):
            # Extraire quelques mots-clés simples
            all_text = ' '.join([doc.content[:100] for doc in extraction_result.documents[:3]])
            keywords = list(set(all_text.split()[:10]))
        
        self.agent_memory.add_research_result(topic, global_synthesis, keywords)
        
        print("✅ Tous les résultats stockés avec succès")
    
    def retrieve_context_for_query(self, query: str, use_cache: bool = True) -> Dict:
        """
        Récupère le contexte pertinent pour une requête
        
        Args:
            query: Requête de l'utilisateur
            use_cache: Utiliser le cache si disponible
        
        Returns:
            Dict avec le contexte vectoriel et conversationnel
        """
        context = {
            'semantic_context': '',
            'conversation_context': '',
            'cached_result': None,
            'related_topics': []
        }
        
        # 1. Vérifier le cache
        if use_cache:
            context['cached_result'] = self.agent_memory.get_research_result(query)
        
        # 2. Recherche sémantique
        context['semantic_context'] = self.vector_memory.get_relevant_context(query, k=3)
        
        # 3. Contexte conversationnel
        context['conversation_context'] = self.agent_memory.get_conversation_context(n_last=3)
        
        # 4. Topics similaires
        context['related_topics'] = self.agent_memory.get_related_topics(query)
        
        return context


# ============================================================================
# INITIALISATION GLOBALE
# ============================================================================

# Instance globale du système de mémoire
memory_system = IntegratedMemorySystem()

print("="*60)
print("✅ SYSTÈME DE MÉMOIRE PRÊT")
print("="*60)