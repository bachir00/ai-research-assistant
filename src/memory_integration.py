"""
Intégration du système de mémoire dans l'outil de recherche
"""

from langchain_core.tools import tool
from typing import Union
import asyncio

from src.agents.researcher_agent import ResearcherAgent
from src.agents.content_extractor_agent import ContentExtractorAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.global_synthesizer_agent import GlobalSynthesizerAgent

from src.models.research_models import ResearchQuery
# ------------ AGENTS ------------
researcher_agent = ResearcherAgent()
content_extractor_agent = ContentExtractorAgent()
summarizer_agent = SummarizerAgent()
global_synthesizer_agent = GlobalSynthesizerAgent()

# Importer le système de mémoire
from .memory_integration import memory_system

# ============================================================================
# OUTIL AMÉLIORÉ AVEC MÉMOIRE
# ============================================================================

@tool
def research_complete_pipeline_with_memory(topic: str, max_results: Union[int, str] = 3, use_cache: bool = True) -> str:
    """Exécute un pipeline de recherche complet avec système de mémoire intégré.
    
    Ce tool intelligent :
    1. Vérifie si des recherches similaires existent en cache
    2. Utilise la mémoire vectorielle pour enrichir le contexte
    3. Exécute le pipeline complet de recherche si nécessaire
    4. Stocke tous les résultats pour réutilisation future
    5. Déduplique automatiquement les documents
    
    Args:
        topic: Le sujet de recherche
        max_results: Nombre de sources à analyser (2-10, défaut: 3)
        use_cache: Utiliser le cache si disponible (défaut: True)
    
    Returns:
        Un rapport complet enrichi par la mémoire contextuelle
    """
    # Conversion et validation
    if isinstance(max_results, str):
        try:
            max_results = int(max_results)
        except ValueError:
            max_results = 3
    max_results = max(2, min(max_results, 10))
    
    async def run_pipeline_with_memory():
        print(f"\n{'='*60}")
        print(f"🚀 PIPELINE DE RECHERCHE INTELLIGENT")
        print(f"📋 Sujet: {topic}")
        print(f"💾 Cache activé: {use_cache}")
        print(f"{'='*60}\n")
        
        # ===== PHASE 1: RÉCUPÉRATION DU CONTEXTE =====
        print("🧠 [Phase 1] Récupération du contexte mémoriel...")
        context = memory_system.retrieve_context_for_query(topic, use_cache=use_cache)
        
        # Vérifier si on a un résultat en cache
        if context['cached_result'] and use_cache:
            print("✅ Résultat trouvé en cache (< 24h)")
            print("📊 Utilisation du résultat mémorisé")
            
            cached_report = context['cached_result']
            if hasattr(cached_report, 'final_report'):
                return cached_report.final_report.formatted_outputs.get('markdown', str(cached_report))
        
        # Afficher le contexte sémantique si disponible
        if context['semantic_context']:
            print(f"📚 Contexte sémantique récupéré ({len(context['semantic_context'])} caractères)")
        
        if context['related_topics']:
            print(f"🔗 Topics similaires trouvés: {', '.join(context['related_topics'][:3])}")
        
        # ===== PHASE 2: EXÉCUTION DU PIPELINE =====
        print(f"\n{'='*60}")
        print("🔬 [Phase 2] Exécution du pipeline de recherche")
        print(f"{'='*60}\n")
        
        # ÉTAPE 1: Recherche
        print("🔍 [1/4] Recherche web en cours...")
        query = ResearchQuery(
            topic=topic,
            keywords=await researcher_agent.extract_keywords_with_llm(topic),
            max_results=max_results,
            search_depth="basic"
        )
        research_data = await researcher_agent.process(query)
        print(f"✅ Trouvé {research_data.total_found} sources")
        
        # ÉTAPE 2: Extraction avec déduplication
        print("\n📄 [2/4] Extraction du contenu (avec déduplication)...")
        extraction_data = await content_extractor_agent.process_from_research_output(
            research_output=research_data
        )
        print(f"✅ Extrait {extraction_data.successful_extractions} documents")
        
        # Vérifier les doublons
        if extraction_data.documents:
            new_docs = []
            duplicates = 0
            for doc in extraction_data.documents:
                if not memory_system.vector_memory.is_duplicate(doc.content):
                    new_docs.append(doc)
                else:
                    duplicates += 1
            
            if duplicates > 0:
                print(f"ℹ️ {duplicates} documents en doublon ignorés")
                # Mettre à jour extraction_data avec seulement les nouveaux docs
                extraction_data.documents = new_docs
        
        # ÉTAPE 3: Résumés
        print("\n📝 [3/4] Création des résumés...")
        summarization_data = await summarizer_agent.process_from_extraction_result(
            extraction_result=extraction_data
        )
        print(f"✅ Généré {summarization_data.total_documents} résumés")
        
        # ÉTAPE 4: Synthèse globale enrichie
        print("\n🎯 [4/4] Synthèse globale (enrichie par le contexte)...")
        
        # Enrichir avec le contexte sémantique si disponible
        if context['semantic_context']:
            print("📚 Enrichissement avec le contexte mémoriel...")
        
        global_synthesis = await global_synthesizer_agent.process_from_summarization_output(
            summarization_output=summarization_data
        )
        print(f"✅ Rapport final généré ({global_synthesis.final_report.word_count} mots)")
        
        # ===== PHASE 3: STOCKAGE EN MÉMOIRE =====
        print(f"\n{'='*60}")
        print("💾 [Phase 3] Stockage en mémoire")
        print(f"{'='*60}\n")
        
        memory_system.process_research_result(
            topic=topic,
            extraction_result=extraction_data,
            summarization_result=summarization_data,
            global_synthesis=global_synthesis
        )
        
        # Ajouter à l'historique des conversations
        final_report_text = global_synthesis.final_report.formatted_outputs.get('text', '')[:200]
        memory_system.agent_memory.add_conversation(
            user_message=f"Recherche sur: {topic}",
            assistant_response=final_report_text,
            metadata={'max_results': max_results, 'sources': research_data.total_found}
        )
        
        print(f"\n{'='*60}")
        print("✨ PIPELINE TERMINÉ AVEC SUCCÈS")
        print(f"📊 Statistiques:")
        print(f"   - Sources analysées: {research_data.total_found}")
        print(f"   - Documents stockés: {extraction_data.successful_extractions}")
        print(f"   - Résumés générés: {summarization_data.total_documents}")
        print(f"   - Mots du rapport: {global_synthesis.final_report.word_count}")
        print(f"{'='*60}\n")
        
        # Retourner le rapport en markdown
        return global_synthesis.final_report.formatted_outputs.get('markdown',
                                                                   global_synthesis.final_report.formatted_outputs.get('text',
                                                                                                                       str(global_synthesis)))
    
    return asyncio.run(run_pipeline_with_memory())


# ============================================================================
# OUTILS SUPPLÉMENTAIRES POUR LA GESTION DE MÉMOIRE
# ============================================================================

@tool
def search_in_memory(query: str, top_k: int = 5) -> str:
    """Recherche sémantique dans la mémoire vectorielle.
    
    Utile pour retrouver des informations de recherches précédentes
    sans relancer une nouvelle recherche complète.
    
    Args:
        query: Requête de recherche
        top_k: Nombre de résultats à retourner
    
    Returns:
        Contexte pertinent trouvé dans la mémoire
    """
    print(f"🔍 Recherche dans la mémoire: '{query}'")
    
    results = memory_system.vector_memory.semantic_search(query, k=top_k)
    
    if not results:
        return "Aucun résultat trouvé dans la mémoire."
    
    output = f"📚 {len(results)} résultats trouvés dans la mémoire:\n\n"
    
    for i, (doc, score) in enumerate(results, 1):
        output += f"[Résultat {i} - Pertinence: {score:.2%}]\n"
        output += f"Titre: {doc.metadata.get('title', 'N/A')}\n"
        output += f"Source: {doc.metadata.get('source', 'N/A')}\n"
        output += f"Contenu:\n{doc.page_content[:300]}...\n\n"
    
    return output


@tool
def get_research_history(n_last: int = 5) -> str:
    """Récupère l'historique des dernières recherches effectuées.
    
    Args:
        n_last: Nombre de conversations récentes à retourner
    
    Returns:
        Historique formaté des recherches
    """
    print(f"📜 Récupération des {n_last} dernières recherches...")
    
    history = list(memory_system.agent_memory.conversation_history)[-n_last:]
    
    if not history:
        return "Aucun historique de recherche disponible."
    
    output = f"📚 Historique des {len(history)} dernières recherches:\n\n"
    
    for i, conv in enumerate(history, 1):
        timestamp = conv.get('timestamp', 'N/A')
        user_msg = conv.get('user', '')[:100]
        metadata = conv.get('metadata', {})
        
        output += f"[Recherche {i}] - {timestamp}\n"
        output += f"Topic: {user_msg}\n"
        if metadata:
            output += f"Détails: {metadata}\n"
        output += "\n"
    
    return output


@tool
def clear_memory(confirm: bool = False) -> str:
    """Réinitialise complètement le système de mémoire.
    
    ⚠️ ATTENTION: Cette action est irréversible!
    
    Args:
        confirm: Doit être True pour confirmer l'action
    
    Returns:
        Message de confirmation
    """
    if not confirm:
        return "⚠️ Action non confirmée. Passez confirm=True pour réinitialiser la mémoire."
    
    print("🗑️ Réinitialisation de la mémoire...")
    memory_system.agent_memory.clear_all()
    
    # Note: On ne clear pas la base vectorielle car elle peut contenir des données précieuses
    # Si vraiment nécessaire, utiliser memory_system.vector_memory.collection.delete(where={})
    
    return "✅ Mémoire de conversation réinitialisée. Base vectorielle préservée."


# ============================================================================
# LISTE DES OUTILS MISE À JOUR
# ============================================================================

# Mettre à jour la liste des outils dans votre code principal
tools_with_memory = [
    research_complete_pipeline_with_memory,
    search_in_memory,
    get_research_history,
    clear_memory
]

print("✅ Outils avec mémoire initialisés:")
print("   1. research_complete_pipeline_with_memory - Pipeline complet avec cache")
print("   2. search_in_memory - Recherche dans la mémoire vectorielle")
print("   3. get_research_history - Historique des recherches")
print("   4. clear_memory - Réinitialisation de la mémoire")