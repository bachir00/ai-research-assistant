# """
# Script de test complet pour le système de mémoire
# """

# from langchain_groq import ChatGroq
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode
# from typing import TypedDict, Sequence, Annotated
# from langchain_core.messages import BaseMessage
# from langgraph.graph.message import add_messages
# from dotenv import load_dotenv
# import os

# # Importer les composants
# # from memory_integration import tools_with_memory
# # from enhanced_system_prompt import create_enhanced_model_call, ENHANCED_SYSTEM_PROMPT

# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

# load_dotenv()
# api_key = os.getenv("GROQ_API_KEY")
# if not api_key:
#     raise ValueError("GROQ_API_KEY non définie")

# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0.3,
#     max_tokens=2048,
#     api_key=api_key
# ).bind_tools(tools_with_memory)

# # ============================================================================
# # CONSTRUCTION DU GRAPHE AMÉLIORÉ
# # ============================================================================

# def should_continue(state: AgentState) -> str:
#     messages = state["messages"]
#     last_message = messages[-1]
    
#     if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#         return "continue"
#     else:
#         return "end"

# # Créer le graphe
# graph = StateGraph(AgentState)

# # Ajouter les nœuds
# model_call_enhanced = create_enhanced_model_call()
# graph.add_node("llm", model_call_enhanced)

# tool_node = ToolNode(tools=tools_with_memory)
# graph.add_node("tools", tool_node)

# # Définir les connexions
# graph.set_entry_point("llm")
# graph.add_conditional_edges(
#     "llm",
#     should_continue,
#     {
#         "continue": "tools",
#         "end": END,
#     },
# )
# graph.add_edge("tools", "llm")

# # Compiler
# app_with_memory = graph.compile()

# # ============================================================================
# # FONCTIONS UTILITAIRES
# # ============================================================================

# def print_stream(stream, show_tool_calls=True):
#     """Affiche le flux de messages de manière élégante"""
#     print("\n" + "="*70)
    
#     for i, s in enumerate(stream):
#         message = s["messages"][-1]
        
#         if hasattr(message, 'content') and message.content:
#             print(f"\n{'─'*70}")
#             if hasattr(message, 'type'):
#                 if message.type == 'human':
#                     print("👤 UTILISATEUR:")
#                 elif message.type == 'ai':
#                     print("🤖 ASSISTANT:")
#                 elif message.type == 'tool':
#                     if show_tool_calls:
#                         print("🔧 RÉSULTAT OUTIL:")
            
#             content = message.content
#             if isinstance(content, str):
#                 # Limiter l'affichage si trop long
#                 if len(content) > 1000:
#                     print(content[:1000] + "\n... (contenu tronqué)")
#                 else:
#                     print(content)
#             else:
#                 print(content)
        
#         if hasattr(message, 'tool_calls') and message.tool_calls and show_tool_calls:
#             print("\n🔧 APPELS D'OUTILS:")
#             for tool_call in message.tool_calls:
#                 print(f"  → {tool_call.get('name', 'unknown')}({tool_call.get('args', {})})")
    
#     print("\n" + "="*70)

# def run_test(user_query: str, test_name: str = ""):
#     """Exécute un test avec affichage formaté"""
#     if test_name:
#         print(f"\n\n{'#'*70}")
#         print(f"# TEST: {test_name}")
#         print(f"{'#'*70}")
    
#     inputs = {"messages": [("user", user_query)]}
#     print_stream(app_with_memory.stream(inputs, stream_mode="values"))

# # ============================================================================
# # SUITE DE TESTS
# # ============================================================================

# def run_all_tests():
#     """Exécute tous les tests du système"""
    
#     print("\n" + "="*70)
#     print(" 🧪 SUITE DE TESTS - SYSTÈME DE MÉMOIRE INTELLIGENT")
#     print("="*70)
    
#     # Test 1: Première recherche (création du cache)
#     run_test(
#         "Fais-moi un résumé complet sur l'impact de l'intelligence artificielle sur le marché du travail",
#         "Test 1 - Première recherche (cache vide)"
#     )
    
#     # Test 2: Même sujet (utilisation du cache)
#     run_test(
#         "Peux-tu me redonner les infos sur l'IA et l'emploi ?",
#         "Test 2 - Recherche dans le cache"
#     )
    
#     # Test 3: Recherche dans la mémoire
#     run_test(
#         "Qu'est-ce que tu as trouvé sur l'intelligence artificielle ?",
#         "Test 3 - Recherche sémantique dans la mémoire"
#     )
    
#     # Test 4: Historique
#     run_test(
#         "Montre-moi l'historique de mes recherches",
#         "Test 4 - Consultation de l'historique"
#     )
    
#     # Test 5: Nouvelle recherche différente
#     run_test(
#         "Fais une analyse sur les énergies renouvelables",
#         "Test 5 - Nouvelle recherche (sujet différent)"
#     )
    
#     # Test 6: Question simple (pas de recherche)
#     run_test(
#         "Bonjour, comment ça va ?",
#         "Test 6 - Conversation simple (sans recherche)"
#     )
    
#     # Test 7: Recherche croisée
#     run_test(
#         "Compare ce que tu as trouvé sur l'IA et les énergies renouvelables",
#         "Test 7 - Recherche croisée dans la mémoire"
#     )
    
#     print("\n\n" + "="*70)
#     print(" ✅ TOUS LES TESTS TERMINÉS")
#     print("="*70)


# def demo_memory_stats():
#     """Affiche les statistiques de la mémoire"""
#     from memory_system import memory_system
    
#     print("\n" + "="*70)
#     print(" 📊 STATISTIQUES DU SYSTÈME DE MÉMOIRE")
#     print("="*70)
    
#     # Stats vectorielles
#     vector_count = memory_system.vector_memory.collection.count()
#     print(f"\n🗄️ Base Vectorielle:")
#     print(f"   Documents stockés: {vector_count}")
#     print(f"   Hashes en cache: {len(memory_system.vector_memory.content_hashes)}")
    
#     # Stats agent
#     conv_count = len(memory_system.agent_memory.conversation_history)
#     research_count = len(memory_system.agent_memory.research_cache)
    
#     print(f"\n🧠 Mémoire Agent:")
#     print(f"   Conversations: {conv_count}")
#     print(f"   Recherches en cache: {research_count}")
#     print(f"   Topics mémorisés: {len(memory_system.agent_memory.topic_keywords)}")
    
#     if research_count > 0:
#         print(f"\n📚 Topics en cache:")
#         for topic in list(memory_system.agent_memory.research_cache.keys())[:5]:
#             print(f"   • {topic}")
    
#     print("\n" + "="*70)


# # ============================================================================
# # MENU INTERACTIF
# # ============================================================================

# def interactive_menu():
#     """Menu interactif pour tester le système"""
    
#     while True:
#         print("\n" + "="*70)
#         print(" 🎯 ASSISTANT DE RECHERCHE INTELLIGENT")
#         print("="*70)
#         print("\n Options:")
#         print("  1. Poser une question / Lancer une recherche")
#         print("  2. Rechercher dans la mémoire")
#         print("  3. Voir l'historique")
#         print("  4. Statistiques de la mémoire")
#         print("  5. Lancer la suite de tests")
#         print("  6. Réinitialiser la mémoire")
#         print("  0. Quitter")
        
#         choice = input("\n👉 Votre choix: ").strip()
        
#         if choice == "1":
#             query = input("\n💬 Votre question: ")
#             run_test(query, "Recherche utilisateur")
        
#         elif choice == "2":
#             query = input("\n🔍 Recherche dans la mémoire: ")
#             run_test(f"Cherche dans ta mémoire: {query}", "Recherche mémoire")
        
#         elif choice == "3":
#             run_test("Montre-moi mon historique de recherches", "Historique")
        
#         elif choice == "4":
#             demo_memory_stats()
        
#         elif choice == "5":
#             run_all_tests()
        
#         elif choice == "6":
#             confirm = input("\n⚠️ Êtes-vous sûr de vouloir réinitialiser ? (oui/non): ")
#             if confirm.lower() == "oui":
#                 from memory_system import memory_system
#                 memory_system.agent_memory.clear_all()
#                 print("✅ Mémoire réinitialisée")
#             else:
#                 print("❌ Annulé")
        
#         elif choice == "0":
#             print("\n👋 Au revoir!")
#             break
        
#         else:
#             print("\n❌ Choix invalide")


# # ============================================================================
# # POINT D'ENTRÉE
# # ============================================================================

# if __name__ == "__main__":
#     import sys
    
#     print("\n" + "🚀"*35)
#     print(" SYSTÈME DE RECHERCHE INTELLIGENT AVEC MÉMOIRE")
#     print("🚀"*35 + "\n")
    
#     if len(sys.argv) > 1:
#         if sys.argv[1] == "test":
#             # Mode test automatique
#             run_all_tests()
#             demo_memory_stats()
#         elif sys.argv[1] == "stats":
#             # Afficher uniquement les stats
#             demo_memory_stats()
#         else:
#             # Exécuter une requête directe
#             query = " ".join(sys.argv[1:])
#             run_test(query, "Requête CLI")
#     else:
#         # Mode interactif
#         interactive_menu()