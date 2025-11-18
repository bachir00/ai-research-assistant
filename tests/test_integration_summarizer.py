"""
Tests d'intégration pour l'Agent Summarizer.
Ces tests vérifient le fonctionnement complet de l'agent avec des données réelles.
"""

import asyncio
import pytest
from datetime import datetime

from src.agents.summarizer_agent import SummarizerAgent, SummarizationInput
from src.models.document_models import Document


class TestSummarizerIntegration:
    """Tests d'intégration pour l'Agent Summarizer."""
    
    @pytest.fixture
    def real_documents(self):
        """Documents réels pour les tests d'intégration."""
        return [
            Document(
                title="L'Intelligence Artificielle et l'Avenir du Travail",
                url="https://example.com/ia-travail",
                content="""
                L'intelligence artificielle (IA) transforme rapidement le marché du travail mondial. 
                Cette révolution technologique présente à la fois des opportunités extraordinaires et 
                des défis considérables pour les travailleurs de tous secteurs.
                
                Les bénéfices de l'IA incluent :
                - Automatisation des tâches répétitives
                - Amélioration de la productivité
                - Création de nouveaux types d'emplois
                - Assistance dans la prise de décision
                
                Cependant, l'IA pose aussi des risques :
                - Suppression d'emplois traditionnels
                - Nécessité de reconversion professionnelle
                - Creusement des inégalités
                - Questions éthiques sur l'autonomie
                
                Les experts s'accordent sur la nécessité d'une transition progressive et d'une 
                formation continue des travailleurs. Les gouvernements et entreprises doivent 
                collaborer pour assurer une adoption responsable de l'IA.
                
                En conclusion, l'IA représente une transformation majeure qui nécessite une 
                préparation active de tous les acteurs du marché du travail.
                """,
                metadata={
                    "author": "Dr. Marie Dupont",
                    "publication_date": "2024-03-15",
                    "source": "Journal of Future Work"
                },
                timestamp=datetime.now(),
                source_type="research_paper"
            ),
            Document(
                title="Éthique et Intelligence Artificielle",
                url="https://example.com/ethique-ia",
                content="""
                L'éthique de l'intelligence artificielle est devenue un sujet central dans le 
                développement technologique moderne. Alors que l'IA devient omniprésente, nous 
                devons nous interroger sur ses implications morales et sociales.
                
                Principes éthiques fondamentaux :
                1. Transparence et explicabilité
                2. Équité et non-discrimination
                3. Respect de la vie privée
                4. Responsabilité et accountability
                5. Bienveillance et non-malfaisance
                
                Les défis éthiques majeurs incluent :
                - Biais algorithmiques dans les systèmes de décision
                - Surveillance et contrôle social
                - Manipulation des opinions publiques
                - Concentration du pouvoir technologique
                
                Les solutions proposées comprennent :
                - Régulation gouvernementale adaptée
                - Standards industriels stricts
                - Éducation du public aux enjeux IA
                - Recherche en IA responsable
                
                L'avenir de l'IA dépend de notre capacité à intégrer ces considérations éthiques 
                dès la conception des systèmes. Une approche collaborative entre technologues, 
                éthiciens et société civile est essentielle.
                """,
                metadata={
                    "author": "Prof. Jean Martin",
                    "publication_date": "2024-04-02",
                    "source": "Ethics in Technology Review"
                },
                timestamp=datetime.now(),
                source_type="academic_article"
            )
        ]
    
    @pytest.fixture
    def summarization_input(self, real_documents):
        """Configuration d'entrée pour la synthèse."""
        return SummarizationInput(
            documents=real_documents,
            summary_options={
                'detailed_analysis': True,
                'max_key_points': 8,
                'include_sentiment': True,
                'include_citations': True
            }
        )
    
    @pytest.mark.asyncio
    async def test_real_document_summarization(self, summarization_input):
        """Test de synthèse avec des documents réels."""
        agent = SummarizerAgent()
        
        # Traiter les documents
        result = await agent.execute(summarization_input)
        
        # Vérifications de base
        assert result is not None
        assert len(result.summaries) == 2
        assert result.total_documents == 2
        assert result.total_processing_time >= 0
        
        # Vérifications des résumés individuels
        for summary in result.summaries:
            assert summary.title is not None and len(summary.title) > 0
            assert summary.executive_summary is not None and len(summary.executive_summary) > 0
            assert summary.detailed_summary is not None and len(summary.detailed_summary) > 0
            assert len(summary.key_points) > 0
            assert summary.sentiment in ['positif', 'négatif', 'neutre']
            assert 0 <= summary.credibility_score <= 1
        
        # Vérifications de l'analyse globale
        assert isinstance(result.common_themes, list)
        assert isinstance(result.consensus_points, list)
        assert isinstance(result.conflicting_views, list)
        
        print(f"\n✅ Test réussi ! {result.total_documents} documents traités en {result.total_processing_time:.2f}s")
        print(f"📊 Score de crédibilité moyen: {result.average_credibility:.2f}")
        print(f"🎯 Thèmes communs identifiés: {len(result.common_themes)}")
    
    @pytest.mark.asyncio
    async def test_executive_summary_type(self, real_documents):
        """Test avec type de résumé exécutif."""
        summarization_input = SummarizationInput(
            documents=real_documents,
            summary_options={
                'detailed_analysis': False,  # Résumé exécutif plus court
                'max_key_points': 5,
                'include_sentiment': False,
                'include_citations': True
            }
        )
        
        agent = SummarizerAgent()
        result = await agent.execute(summarization_input)
        
        assert result is not None
        assert len(result.summaries) == 2
        
        # Vérifier que les résumés exécutifs sont plus courts
        for summary in result.summaries:
            assert len(summary.executive_summary) > 0
            assert len(summary.key_points) <= 5
    
    @pytest.mark.asyncio 
    async def test_error_handling_invalid_content(self):
        """Test de gestion d'erreur avec contenu invalide."""
        documents = [
            Document(
                title="Document vide",
                url="https://example.com/empty",
                content="",
                metadata={},
                timestamp=datetime.now(),
                source_type="web_page"
            )
        ]
        
        summarization_input = SummarizationInput(
            documents=documents,
            summary_options={
                'detailed_analysis': True,
                'include_sentiment': True
            }
        )
        
        agent = SummarizerAgent()
        result = await agent.execute(summarization_input)
        
        # L'agent devrait gérer gracieusement les documents vides
        assert result is not None
        assert result.total_documents == 1
        # Le résumé pourrait être un résumé d'erreur
        assert len(result.summaries) == 1
    
    @pytest.mark.asyncio
    async def test_large_document_chunking(self):
        """Test avec un document volumineux nécessitant un découpage."""
        large_content = """
        L'intelligence artificielle représente l'un des défis technologiques les plus importants de notre époque.
        """ * 100  # Répéter pour créer un contenu volumineux
        
        large_document = Document(
            title="Grande analyse de l'IA",
            url="https://example.com/large-doc",
            content=large_content,
            metadata={"length": "large"},
            timestamp=datetime.now(),
            source_type="research_paper"
        )
        
        summarization_input = SummarizationInput(
            documents=[large_document],
            summary_options={
                'detailed_analysis': True,
                'max_key_points': 10,
                'chunk_large_docs': True
            }
        )
        
        agent = SummarizerAgent()
        result = await agent.execute(summarization_input)
        
        assert result is not None
        assert len(result.summaries) == 1
        
        summary = result.summaries[0]
        assert len(summary.executive_summary) > 0
        assert len(summary.detailed_summary) > 0
        assert len(summary.key_points) > 0
        
        print(f"\n✅ Document volumineux traité avec succès ({len(large_content)} caractères)")


if __name__ == "__main__":
    # Exécution directe pour test rapide
    async def run_quick_test():
        """Test rapide pour validation."""
        test_instance = TestSummarizerIntegration()
        
        # Créer des documents de test
        documents = [
            Document(
                title="Test IA",
                url="https://test.com",
                content="L'IA est positive pour l'avenir. Elle améliore la productivité.",
                metadata={},
                timestamp=datetime.now(),
                source_type="web_page"
            )
        ]
        
        summarization_input = SummarizationInput(
            documents=documents,
            summary_options={
                'detailed_analysis': False  # Résumé exécutif plus simple
            }
        )
        
        agent = SummarizerAgent()
        result = await agent.execute(summarization_input)
        
        print(f"✅ Test rapide réussi: {len(result.summaries)} résumé(s) créé(s)")
        return result
    
    # Décommenter pour test rapide
    # asyncio.run(run_quick_test())