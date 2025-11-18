"""
Tests unitaires pour valider les modèles Pydantic.
"""

import sys
from pathlib import Path

from src.core.logging import setup_logger
logger = setup_logger("test_models_logger")

# Ajouter le répertoire racine au path
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

try:
    import pytest
except ImportError:
    pytest = None

from datetime import datetime
from pydantic import ValidationError

from src.models import (
    ResearchQuery,
    SearchResult, 
    ResearchOutput,
    Document,
    DocumentSummary,
    Report,
    GraphState,
    AgentType,
    ProcessingStep
)


def test_research_query():
    """Test du modèle ResearchQuery."""
    query = ResearchQuery(
        topic="Intelligence artificielle et emploi",
        keywords=["IA", "automatisation"],
        max_results=10
    )
    
    assert query.topic == "Intelligence artificielle et emploi"
    assert "IA" in query.keywords
    assert query.max_results == 10
    assert query.search_depth == "basic"  # valeur par défaut


def test_search_result():
    """Test du modèle SearchResult."""
    result = SearchResult(
        title="L'IA transforme le travail",
        url="https://example.com/article",
        snippet="L'intelligence artificielle modifie..."
    )
    
    assert result.title == "L'IA transforme le travail"
    assert str(result.url) == "https://example.com/article"
    assert result.snippet.startswith("L'intelligence")


def test_document():
    """Test du modèle Document."""
    doc = Document(
        title="Article sur l'IA",
        url="https://example.com/doc",
        content="Contenu de l'article sur l'IA...",
        word_count=150
    )
    
    assert doc.title == "Article sur l'IA"
    assert doc.word_count == 150
    assert doc.doc_type == "article"  # valeur par défaut


def test_graph_state():
    """Test du modèle GraphState."""
    state = GraphState(
        session_id="test_session_123",
        current_step=ProcessingStep.INIT
    )
    
    assert state.session_id == "test_session_123"
    assert state.current_step == ProcessingStep.INIT
    assert len(state.agents) == 3  # 3 agents par défaut
    assert AgentType.RESEARCHER in state.agents
    assert not state.all_agents_completed()


def test_graph_state_methods():
    """Test des méthodes du GraphState."""
    state = GraphState(session_id="test")
    
    # Test de l'agent en cours
    assert state.get_current_agent() is None
    
    # Marquer un agent comme en cours
    state.agents[AgentType.RESEARCHER].start_execution()
    assert state.get_current_agent() == AgentType.RESEARCHER
    
    # Terminer l'agent
    state.agents[AgentType.RESEARCHER].complete_execution()
    assert state.is_agent_completed(AgentType.RESEARCHER)
    
    # Test des erreurs
    assert not state.has_errors()
    state.add_global_error("Erreur de test")
    assert state.has_errors()


def test_validation_errors():
    """Test des erreurs de validation."""
    
    # URL invalide pour SearchResult
    with pytest.raises(ValidationError):
        SearchResult(
            title="Test",
            url="invalid_url",  # URL invalide
            snippet="Test snippet"
        )
    
    # max_results négatif pour ResearchQuery
    with pytest.raises(ValidationError):
        ResearchQuery(
            topic="Test",
            max_results=-1  # Valeur invalide
        )


if __name__ == "__main__":
    # Exécution rapide des tests
    test_research_query()
    test_search_result() 
    test_document()
    test_graph_state()
    test_graph_state_methods()
    
    logger.info("✅  Tous les tests des modèles sont passés avec succès!")
    
    # Test des erreurs de validation
    try:
        test_validation_errors()
        logger.warning("⚠️   Les tests de validation d'erreurs nécessitent pytest")
    except ImportError:
        print("ℹ️ pytest non installé - tests de validation d'erreurs ignorés")