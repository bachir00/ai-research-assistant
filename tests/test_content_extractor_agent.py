"""
Tests pour l'agent Content Extractor.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.agents.content_extractor_agent import ContentExtractorAgent, extract_content_from_urls
from src.models.document_models import Document, DocumentType, ExtractionInput, ExtractionResult
from src.services.content_extraction import ContentExtractionError


class TestContentExtractorAgent:
    """Tests pour la classe ContentExtractorAgent."""
    
    @pytest.fixture
    def agent(self):
        """Fixture pour l'agent."""
        return ContentExtractorAgent(max_concurrent_extractions=2, max_retries=1)
    
    @pytest.fixture
    def sample_urls(self):
        """URLs d'exemple pour les tests."""
        return [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/pdf-doc.pdf"
        ]
    
    @pytest.fixture
    def sample_documents(self):
        """Documents d'exemple."""
        return [
            Document(
                title="Article 1",
                url="https://example.com/article1",
                content="Ceci est le contenu de l'article 1. " * 20,
                doc_type=DocumentType.ARTICLE,
                word_count=100,
                language="fr"
            ),
            Document(
                title="Article 2",
                url="https://example.com/article2",
                content="Ceci est le contenu de l'article 2. " * 15,
                doc_type=DocumentType.ARTICLE,
                word_count=75,
                language="fr"
            )
        ]
    
    @pytest.fixture
    def extraction_input(self, sample_urls):
        """Input d'extraction d'exemple."""
        return ExtractionInput(
            urls=sample_urls,
            content_filters={
                'min_content_length': 50,
                'max_content_length': 10000,
                'language': 'fr'
            }
        )
    
    def test_agent_initialization(self):
        """Test de l'initialisation de l'agent."""
        agent = ContentExtractorAgent()
        
        assert agent.name == "content_extractor"
        assert agent.extraction_manager is not None
        assert agent.logger is not None
    
    def test_filter_valid_urls(self, agent):
        """Test du filtrage des URLs valides."""
        urls = [
            "https://example.com/valid",
            "http://test.org/also-valid",
            "invalid-url",
            "",
            None,
            "ftp://not-supported.com",
            "https://",
            "https://valid-domain.co.uk/path?param=value"
        ]
        
        valid_urls = agent._filter_valid_urls(urls)
        
        expected_valid = [
            "https://example.com/valid",
            "http://test.org/also-valid",
            "https://valid-domain.co.uk/path?param=value"
        ]
        
        assert len(valid_urls) == len(expected_valid)
        for url in expected_valid:
            assert url in valid_urls
    
    def test_clean_content(self, agent):
        """Test du nettoyage de contenu."""
        dirty_content = """
        
        Voici un texte    avec des espaces multiples.
        
        
        
        Et des sauts de ligne excessifs.
        	Et des tabulations.
        
        """
        
        cleaned = agent._clean_content(dirty_content)
        
        assert "  " not in cleaned  # Pas d'espaces doubles
        assert "\n\n\n" not in cleaned  # Pas de triple saut de ligne
        assert cleaned.startswith("Voici")  # Pas d'espaces en début
        assert cleaned.endswith("tabulations.")  # Pas d'espaces en fin
    
    def test_calculate_content_quality(self, agent, sample_documents):
        """Test du calcul de qualité de contenu."""
        doc = sample_documents[0]
        doc.author = "Test Author"
        doc.published_date = datetime.now()
        
        quality = agent._calculate_content_quality(doc)
        
        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Devrait avoir une bonne qualité
    
    def test_calculate_stats(self, agent, sample_documents):
        """Test du calcul des statistiques."""
        stats = agent._calculate_stats(sample_documents)
        
        assert 'total_words' in stats
        assert 'average_words_per_doc' in stats
        assert 'doc_types' in stats
        assert 'languages' in stats
        
        assert stats['total_words'] == 175  # 100 + 75
        assert stats['average_words_per_doc'] == 87  # 175 // 2
        assert stats['doc_types']['article'] == 2
        assert stats['languages']['fr'] == 2
    
    def test_calculate_stats_empty(self, agent):
        """Test du calcul des statistiques avec liste vide."""
        stats = agent._calculate_stats([])
        
        assert stats['total_words'] == 0
        assert stats['average_words_per_doc'] == 0
        assert stats['doc_types'] == {}
        assert stats['languages'] == {}
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent, extraction_input, sample_documents):
        """Test d'exécution réussie."""
        # Mock du gestionnaire d'extraction
        with patch.object(agent.extraction_manager, 'extract_multiple', 
                         new_callable=AsyncMock, return_value=sample_documents):
            
            result = await agent.execute(extraction_input)
            
            assert isinstance(result, ExtractionResult)
            assert result.successful_extractions == 2
            assert result.failed_extractions == 1  # 3 URLs - 2 succès
            assert len(result.documents) == 2
            assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_execute_no_urls(self, agent):
        """Test d'exécution sans URLs."""
        input_data = ExtractionInput(urls=[])
        
        with pytest.raises(ValueError, match="Aucune URL fournie"):
            await agent.execute(input_data)
    
    @pytest.mark.asyncio
    async def test_execute_too_many_urls(self, agent):
        """Test d'exécution avec trop d'URLs."""
        urls = [f"https://example.com/page{i}" for i in range(51)]
        input_data = ExtractionInput(urls=urls)
        
        with pytest.raises(ValueError, match="Trop d'URLs"):
            await agent.execute(input_data)
    
    @pytest.mark.asyncio
    async def test_execute_no_valid_urls(self, agent):
        """Test d'exécution sans URLs valides."""
        input_data = ExtractionInput(urls=["invalid", "also-invalid"])
        
        with pytest.raises(ValueError, match="Aucune URL valide trouvée"):
            await agent.execute(input_data)
    
    @pytest.mark.asyncio
    async def test_execute_extraction_error(self, agent, extraction_input):
        """Test d'exécution avec erreur d'extraction."""
        # Mock qui lève une exception
        with patch.object(agent.extraction_manager, 'extract_multiple', 
                         new_callable=AsyncMock, side_effect=Exception("Erreur test")):
            
            with pytest.raises(ContentExtractionError):
                await agent.execute(extraction_input)
    
    def test_apply_content_filters(self, agent, sample_documents):
        """Test d'application des filtres de contenu."""
        filters = {
            'min_content_length': 600,  # Plus que le document 2 (75 mots = ~375 caractères)
            'language': 'fr'
        }
        
        filtered = agent._apply_content_filters(sample_documents, filters)
        
        # Seul le premier document devrait passer (100 mots = ~500+ caractères)
        assert len(filtered) == 1
        assert filtered[0].title == "Article 1"
    
    def test_apply_content_filters_keywords(self, agent, sample_documents):
        """Test des filtres avec mots-clés requis."""
        # Modifier le contenu pour les tests
        sample_documents[0].content = "Intelligence artificielle et machine learning"
        sample_documents[1].content = "Recette de cuisine traditionnelle"
        
        filters = {
            'required_keywords': ['intelligence', 'machine']
        }
        
        filtered = agent._apply_content_filters(sample_documents, filters)
        
        assert len(filtered) == 1
        assert filtered[0].title == "Article 1"
    
    def test_is_valid_document(self, agent, extraction_input):
        """Test de validation de document."""
        # Document valide
        valid_doc = Document(
            title="Test",
            url="https://example.com",
            content="Contenu suffisamment long pour être valide " * 10,
            doc_type=DocumentType.ARTICLE,
            word_count=50
        )
        
        assert agent._is_valid_document(valid_doc, extraction_input) is True
        
        # Document invalide (trop court)
        invalid_doc = Document(
            title="Test",
            url="https://example.com",
            content="Trop court",
            doc_type=DocumentType.ARTICLE,
            word_count=2
        )
        
        assert agent._is_valid_document(invalid_doc, extraction_input) is False


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires."""
    
    @pytest.mark.asyncio
    async def test_extract_content_from_urls(self):
        """Test de la fonction utilitaire d'extraction."""
        urls = ["https://example.com/test"]
        
        # Mock de l'agent
        with patch('src.agents.content_extractor_agent.ContentExtractorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_result = ExtractionResult(
                documents=[],
                total_urls=1,
                successful_extractions=0,
                failed_extractions=1,
                execution_time=1.0,
                extraction_stats={}
            )
            mock_agent.execute = AsyncMock(return_value=mock_result)
            
            documents = await extract_content_from_urls(urls)
            
            assert documents == []
            mock_agent.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_from_search_results(self):
        """Test d'extraction depuis des résultats de recherche."""
        from src.agents.content_extractor_agent import extract_from_search_results
        
        search_results = [
            {'url': 'https://example.com/1', 'title': 'Test 1'},
            {'url': 'https://example.com/2', 'title': 'Test 2'}
        ]
        
        with patch('src.agents.content_extractor_agent.extract_content_from_urls') as mock_extract:
            mock_extract.return_value = []
            
            documents = await extract_from_search_results(search_results)
            
            mock_extract.assert_called_once_with(['https://example.com/1', 'https://example.com/2'])
            assert documents == []
    
    @pytest.mark.asyncio
    async def test_extract_from_search_results_with_objects(self):
        """Test d'extraction avec des objets ayant un attribut url."""
        from src.agents.content_extractor_agent import extract_from_search_results
        
        # Mock d'objets avec attribut url
        class SearchResult:
            def __init__(self, url):
                self.url = url
        
        search_results = [
            SearchResult('https://example.com/1'),
            SearchResult('https://example.com/2')
        ]
        
        with patch('src.agents.content_extractor_agent.extract_content_from_urls') as mock_extract:
            mock_extract.return_value = []
            
            documents = await extract_from_search_results(search_results)
            
            mock_extract.assert_called_once_with(['https://example.com/1', 'https://example.com/2'])


if __name__ == "__main__":
    # Tests basiques synchrones
    pytest.main([__file__, "-v"])