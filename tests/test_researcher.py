"""
Tests pour l'agent Researcher.
"""

import sys
import asyncio
from pathlib import Path

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agents.researcher_agent import ResearcherAgent
from src.models.research_models import ResearchQuery
from src.core.logging import setup_logger

# Configuration du logger de test
logger = setup_logger("test_researcher")


async def test_researcher_basic():
    """Test basique de l'agent Researcher."""
    logger.info("=== Test basique de l'agent Researcher avec API ===")
    
    # Création de l'agent
    try:
        agent = ResearcherAgent()
        logger.info(f"Agent créé: {agent}")
        
        # Vérification des APIs disponibles
        stats = agent.get_search_stats()
        logger.info(f"APIs disponibles: {stats['available_apis']}")
        
        if not stats['available_apis']:
            logger.error("Aucune API de recherche disponible - vérifiez vos clés API")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'agent: {e}")
        return False


async def test_researcher_search():
    """Test de recherche avec l'agent."""
    logger.info("=== Test de recherche ===")
    
    try:
        agent = ResearcherAgent()
        
        # Création d'une requête de test
        query = ResearchQuery(
            topic="intelligence artificielle et emploi",
            keywords=["IA", "automatisation", "travail"],
            max_results=3,
            search_depth="basic"
        )
        
        logger.info(f"Requête de test: {query.topic}")
        
        # Exécution de la recherche
        result = await agent.execute(query)
        
        logger.info(f"Recherche terminée:")
        logger.info(f"- Nombre de résultats: {len(result.results)}")
        logger.info(f"- Temps de recherche: {result.search_time:.2f}s")
        logger.info(f"- Moteur utilisé: {result.search_engine}")
        logger.info(f"- Total trouvé: {result.total_found}")
        
        # Affichage des premiers résultats
        for i, search_result in enumerate(result.results[:2], 1):
            logger.info(f"\nRésultat {i}:")
            logger.info(f"  Titre: {search_result.title}")
            logger.info(f"  URL: {search_result.url}")
            logger.info(f"  Score: {search_result.score:.3f}")
            logger.info(f"  Extrait: {search_result.snippet[:100]}...")
        
        return len(result.results) > 0
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        return False


async def test_researcher_validation():
    """Test de validation des inputs."""
    logger.info("=== Test de validation ===")
    
    try:
        agent = ResearcherAgent()
        
        # Test avec requête invalide (sujet trop court)
        invalid_query = ResearchQuery(
            topic="IA",  # Trop court
            max_results=3
        )
        
        try:
            await agent.execute(invalid_query)
            logger.error("La validation aurait dû échouer")
            return False
        except ValueError:
            logger.info("✓ Validation correctement rejetée pour sujet trop court")
        
        # Test avec nombre de résultats invalide
        invalid_query2 = ResearchQuery(
            topic="intelligence artificielle",
            max_results=25  # Trop élevé
        )
        
        try:
            await agent.execute(invalid_query2)
            logger.error("La validation aurait dû échouer")
            return False
        except ValueError:
            logger.info("✓ Validation correctement rejetée pour max_results trop élevé")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors des tests de validation: {e}")
        return False


async def test_researcher_simple_search():
    """Test avec la méthode de recherche simple."""
    logger.info("=== Test de recherche simple ===")
    
    try:
        agent = ResearcherAgent()
        
        # Recherche simple
        results = await agent.search_with_fallback(
            "intelligence artificielle impact emploi",
            max_results=2
        )
        
        logger.info(f"Recherche simple: {len(results)} résultats")
        
        if results:
            logger.info(f"Premier résultat: {results[0].title}")
        
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche simple: {e}")
        return False


async def run_all_tests():
    """Exécute tous les tests de l'agent Researcher."""
    logger.info("🧪 Début des tests de l'agent Researcher")
    
    tests = [
        ("Création de l'agent", test_researcher_basic),
        ("Recherche complète", test_researcher_search),
        ("Validation des inputs", test_researcher_validation),
        ("Recherche simple", test_researcher_simple_search)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"Erreur inattendue dans {test_name}: {e}")
            results[test_name] = False
    
    # Résumé des résultats
    logger.info("\n=== RÉSUMÉ DES TESTS ===")
    passed = 0
    for test_name, success in results.items():
        status = "✅ PASSÉ" if success else "❌ ÉCHOUÉ"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nTests réussis: {passed}/{len(tests)}")
    
    if passed == len(tests):
        logger.info("🎉 Tous les tests sont passés!")
    else:
        logger.warning("⚠️ Certains tests ont échoué")
    
    return passed == len(tests)


if __name__ == "__main__":
    # Exécution des tests
    success = asyncio.run(run_all_tests())