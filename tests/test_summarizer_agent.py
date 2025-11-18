"""
Tests pour l'agent Summarizer.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.agents.summarizer_agent import SummarizerAgent, SummarizationInput
from src.models.document_models import Document, DocumentType, DocumentSummary, KeyPoint
from src.services.text_chunking import TextChunk


class TestSummarizerAgent:
    """Tests pour la classe SummarizerAgent."""
    
    @pytest.fixture
    def agent(self):
        """Fixture pour l'agent."""
        return SummarizerAgent(max_retries=1, timeout=300.0)
    
    @pytest.fixture
    def sample_documents(self):
        """Documents d'exemple pour les tests."""
        return [
            Document(
                title="L'impact de l'IA sur l'emploi",
                url="https://example.com/ia-emploi",
                content="L'intelligence artificielle transforme le marché de l'emploi. " * 50,
                doc_type=DocumentType.ARTICLE,
                word_count=250,
                language="fr"
            ),
            Document(
                title="Automatisation et productivité",
                url="https://example.com/automatisation",
                content="L'automatisation améliore la productivité des entreprises. " * 30,
                doc_type=DocumentType.ARTICLE,
                word_count=150,
                language="fr"
            )
        ]
    
    @pytest.fixture
    def summarization_input(self, sample_documents):
        """Input de summarization d'exemple."""
        return SummarizationInput(
            documents=sample_documents,
            summary_options={
                'include_sentiment': True,
                'include_citations': True,
                'max_key_points': 3,
                'detailed_analysis': True
            }
        )
    
    def test_agent_initialization(self):
        """Test de l'initialisation de l'agent."""
        agent = SummarizerAgent()
        
        assert agent.name == "summarizer"
        assert agent.llm_manager is not None
        assert agent.chunking_manager is not None
        assert agent.logger is not None
    
    def test_validate_input_success(self, agent, summarization_input):
        """Test de validation d'input valide."""
        assert agent.validate_input(summarization_input) is True
    
    def test_validate_input_no_documents(self, agent):
        """Test de validation sans documents."""
        input_data = SummarizationInput(documents=[])
        assert agent.validate_input(input_data) is False
    
    def test_validate_input_too_many_documents(self, agent, sample_documents):
        """Test de validation avec trop de documents."""
        # Créer une liste de 25 documents
        many_docs = sample_documents * 13  # 2 * 13 = 26 documents
        input_data = SummarizationInput(documents=many_docs)
        assert agent.validate_input(input_data) is False
    
    def test_validate_input_no_content(self, agent):
        """Test de validation avec documents sans contenu."""
        empty_docs = [
            Document(
                title="Document vide",
                url="https://example.com/empty",
                content="",
                doc_type=DocumentType.ARTICLE,
                word_count=0,
                language="fr"
            )
        ]
        input_data = SummarizationInput(documents=empty_docs)
        assert agent.validate_input(input_data) is False
    
    def test_generate_document_id(self, agent, sample_documents):
        """Test de génération d'ID de document."""
        doc = sample_documents[0]
        doc_id = agent._generate_document_id(doc)
        
        assert doc_id.startswith("doc_")
        assert len(doc_id) == 12  # "doc_" + 8 caractères de hash
        
        # Le même document doit produire le même ID
        doc_id2 = agent._generate_document_id(doc)
        assert doc_id == doc_id2
    
    def test_parse_detailed_analysis(self, agent):
        """Test du parsing d'analyse détaillée."""
        content = """
Voici l'analyse du document:

Points clés:
- L'IA transforme 60% des emplois
- Formation nécessaire pour s'adapter
- Nouveaux métiers émergent

Autres informations importantes...
"""
        
        result = agent._parse_detailed_analysis(content)
        
        assert 'summary' in result
        assert 'key_points' in result
        assert len(result['key_points']) == 3
        
        # Vérifier les points clés
        points = result['key_points']
        assert any("60% des emplois" in point.content for point in points)
        assert any("Formation" in point.content for point in points)
    
    def test_parse_sentiment_analysis(self, agent):
        """Test du parsing d'analyse de sentiment."""
        content_positif = "Le sentiment général est positif. Crédibilité: 0.8"
        result = agent._parse_sentiment_analysis(content_positif)
        
        assert result['sentiment'] == 'positif'
        assert result['credibility_score'] == 0.8
        
        content_negatif = "Ton très négatif dans l'article. Score: 60%"
        result = agent._parse_sentiment_analysis(content_negatif)
        
        assert result['sentiment'] == 'négatif'
        assert result['credibility_score'] == 0.6
    
    def test_create_error_summary(self, agent, sample_documents):
        """Test de création de résumé d'erreur."""
        doc = sample_documents[0]
        error_msg = "Erreur de test"
        
        summary = agent._create_error_summary(doc, error_msg)
        
        assert isinstance(summary, DocumentSummary)
        assert summary.title == doc.title
        assert summary.url == doc.url
        assert error_msg in summary.executive_summary
        assert error_msg in summary.detailed_summary
        assert summary.credibility_score is None
    
    def test_calculate_average_credibility(self, agent):
        """Test du calcul de crédibilité moyenne."""
        summaries = [
            DocumentSummary(
                document_id="doc1",
                title="Test 1",
                url="https://example.com/1",
                executive_summary="Résumé 1",
                detailed_summary="Détails 1",
                credibility_score=0.8
            ),
            DocumentSummary(
                document_id="doc2",
                title="Test 2",
                url="https://example.com/2",
                executive_summary="Résumé 2",
                detailed_summary="Détails 2",
                credibility_score=0.6
            ),
            DocumentSummary(
                document_id="doc3",
                title="Test 3",
                url="https://example.com/3",
                executive_summary="Résumé 3",
                detailed_summary="Détails 3",
                credibility_score=None  # Pas de score
            )
        ]
        
        avg = agent._calculate_average_credibility(summaries)
        assert avg == 0.7  # (0.8 + 0.6) / 2
        
        # Test avec aucun score
        summaries_no_score = [s for s in summaries if s.credibility_score is None]
        avg_none = agent._calculate_average_credibility(summaries_no_score)
        assert avg_none is None
    
    def test_parse_global_analysis(self, agent):
        """Test du parsing d'analyse globale."""
        content = """
Thèmes communs identifiés:
- Intelligence artificielle
- Transformation du travail
- Formation professionnelle

Points de consensus:
- L'IA va changer l'emploi
- La formation est cruciale

Points conflictuels:
- Débat sur le nombre d'emplois perdus
- Désaccord sur la rapidité du changement
"""
        
        result = agent._parse_global_analysis(content)
        
        assert 'common_themes' in result
        assert 'consensus_points' in result
        assert 'conflicting_views' in result
        
        assert len(result['common_themes']) == 3
        assert len(result['consensus_points']) == 2
        assert len(result['conflicting_views']) == 2
        
        assert "Intelligence artificielle" in result['common_themes']
        assert "L'IA va changer l'emploi" in result['consensus_points']
        assert "emplois perdus" in result['conflicting_views'][0]
    
    @pytest.mark.asyncio
    async def test_get_llm_response_success(self, agent):
        """Test de réponse LLM réussie."""
        with patch.object(agent.llm_manager, 'get_completion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Réponse de test"
            
            task_type, response = await agent._get_llm_response("Test prompt", "test_task")
            
            assert task_type == "test_task"
            assert response == "Réponse de test"
            mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_llm_response_error(self, agent):
        """Test de gestion d'erreur LLM."""
        with patch.object(agent.llm_manager, 'get_completion', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("Erreur LLM test")
            
            task_type, response = await agent._get_llm_response("Test prompt", "test_task")
            
            assert task_type == "test_task"
            assert "Erreur" in response
            assert "Erreur LLM test" in response
    
    @pytest.mark.asyncio
    async def test_summarize_chunks(self, agent, sample_documents):
        """Test de résumé de chunks."""
        doc = sample_documents[0]
        
        # Créer des chunks de test
        chunks = [
            TextChunk(
                content="Première partie du document...",
                start_index=0,
                end_index=100,
                chunk_id=1,
                total_chunks=2,
                word_count=20
            ),
            TextChunk(
                content="Deuxième partie du document...",
                start_index=100,
                end_index=200,
                chunk_id=2,
                total_chunks=2,
                word_count=25
            )
        ]
        
        with patch.object(agent.llm_manager, 'get_completion', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [
                "Résumé chunk 1",
                "Résumé chunk 2"
            ]
            
            summaries = await agent._summarize_chunks(chunks, doc)
            
            assert len(summaries) == 2
            assert summaries[0] == "Résumé chunk 1"
            assert summaries[1] == "Résumé chunk 2"
            assert mock_llm.call_count == 2
    
    @pytest.mark.asyncio 
    async def test_process_with_mock_llm(self, agent, summarization_input):
        """Test du processus complet avec LLM mocké."""
        with patch.object(agent.llm_manager, 'get_completion', new_callable=AsyncMock) as mock_llm:
            # Configurer les réponses mock pour différents prompts
            mock_llm.side_effect = [
                "Résumé exécutif document 1",
                "Analyse détaillée document 1\n- Point clé 1\n- Point clé 2",
                "Sentiment: positif, Crédibilité: 0.8",
                "Résumé exécutif document 2", 
                "Analyse détaillée document 2\n- Point clé 3",
                "Sentiment: neutre, Crédibilité: 0.7",
                "Analyse globale:\nThèmes communs:\n- IA\nConsensus:\n- Changement nécessaire"
            ]
            
            result = await agent.execute(summarization_input)
            
            # Vérifications
            assert len(result.summaries) == 2
            assert result.total_documents == 2
            assert result.total_processing_time >= 0
            assert result.average_credibility is not None
            
            # Vérifier les résumés
            for summary in result.summaries:
                assert isinstance(summary, DocumentSummary)
                assert summary.executive_summary
                assert summary.detailed_summary
                assert summary.processing_time > 0
                assert summary.document_id.startswith("doc_")


class TestSummarizationInput:
    """Tests pour la classe SummarizationInput."""
    
    def test_initialization_default_options(self):
        """Test d'initialisation avec options par défaut."""
        docs = [Mock()]
        input_data = SummarizationInput(docs)
        
        assert input_data.documents == docs
        assert input_data.include_sentiment is True
        assert input_data.include_citations is True
        assert input_data.max_key_points == 5
        assert input_data.detailed_analysis is True
        assert input_data.chunk_large_docs is True
        assert input_data.max_doc_size == 8000
    
    def test_initialization_custom_options(self):
        """Test d'initialisation avec options personnalisées."""
        docs = [Mock()]
        options = {
            'include_sentiment': False,
            'max_key_points': 3,
            'detailed_analysis': False,
            'max_doc_size': 5000
        }
        
        input_data = SummarizationInput(docs, options)
        
        assert input_data.include_sentiment is False
        assert input_data.max_key_points == 3
        assert input_data.detailed_analysis is False
        assert input_data.max_doc_size == 5000
        # Options non spécifiées gardent leur valeur par défaut
        assert input_data.include_citations is True


if __name__ == "__main__":
    # Tests basiques synchrones
    pytest.main([__file__, "-v"])