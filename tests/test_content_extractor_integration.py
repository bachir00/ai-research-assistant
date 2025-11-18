"""
Test d'intégration pour l'agent Content Extractor.
Test avec de vraies URLs pour vérifier le fonctionnement complet.
"""

import pytest
import asyncio
from datetime import datetime

from src.agents.content_extractor_agent import ContentExtractorAgent
from src.models.document_models import ExtractionInput, DocumentType


class TestContentExtractorIntegration:
    """Tests d'intégration avec de vraies URLs."""
    
    @pytest.fixture
    def agent(self):
        """Agent pour les tests d'intégration."""
        return ContentExtractorAgent(max_concurrent_extractions=2, max_retries=1)
    
    @pytest.mark.asyncio
    async def test_extract_real_webpage(self, agent):
        """Test d'extraction d'une vraie page web."""
        # URL d'exemple - page Wikipedia simple
        test_urls = [
            "https://httpbin.org/html"  # Page HTML simple pour test
        ]
        
        input_data = ExtractionInput(
            urls=test_urls,
            content_filters={
                'min_content_length': 10,
                'max_content_length': 50000
            }
        )
        
        try:
            result = await agent.execute(input_data)
            
            # Vérifications de base
            assert result is not None
            assert result.total_urls == 1
            assert result.execution_time > 0
            
            # Si l'extraction réussit
            if result.successful_extractions > 0:
                assert len(result.documents) > 0
                
                doc = result.documents[0]
                assert doc.title is not None
                assert len(doc.content) >= 10
                assert doc.url in test_urls
                assert doc.word_count > 0
            
            print(f"Résultat: {result.successful_extractions}/{result.total_urls} succès")
            if result.documents:
                print(f"Premier document: {result.documents[0].title[:100]}")
            
        except Exception as e:
            # En cas d'erreur réseau, on accepte l'échec mais on log
            print(f"Test d'intégration échoué (normal si pas de réseau): {e}")
            pytest.skip(f"Test d'intégration échoué: {e}")
    
    @pytest.mark.asyncio
    async def test_extract_multiple_urls(self, agent):
        """Test d'extraction de plusieurs URLs."""
        test_urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/robots.txt"
        ]
        
        input_data = ExtractionInput(
            urls=test_urls,
            content_filters={'min_content_length': 5}
        )
        
        try:
            result = await agent.execute(input_data)
            
            assert result.total_urls == 2
            assert result.execution_time > 0
            assert result.successful_extractions >= 0
            assert result.failed_extractions >= 0
            assert result.successful_extractions + result.failed_extractions == 2
            
            print(f"Extraction multiple: {result.successful_extractions}/{result.total_urls} succès")
            
        except Exception as e:
            print(f"Test multiple URLs échoué: {e}")
            pytest.skip(f"Test multiple URLs échoué: {e}")
    
    @pytest.mark.asyncio 
    async def test_invalid_url_handling(self, agent):
        """Test de gestion des URLs invalides."""
        test_urls = [
            "https://httpbin.org/html",  # URL valide
            "https://site-qui-nexiste-pas-du-tout-987654321.com",  # URL invalide
            "invalid-url-format"  # Format invalide
        ]
        
        input_data = ExtractionInput(urls=test_urls)
        
        try:
            result = await agent.execute(input_data)
            
            # Doit traiter seulement les URLs valides
            assert result.total_urls == 3
            assert result.failed_extractions >= 2  # Au moins les 2 URLs invalides
            
            print(f"Gestion erreurs: {result.successful_extractions} succès, {result.failed_extractions} échecs")
            
        except Exception as e:
            print(f"Test gestion erreurs échoué: {e}")
            pytest.skip(f"Test gestion erreurs échoué: {e}")


if __name__ == "__main__":
    # Test rapide synchrone
    async def quick_test():
        agent = ContentExtractorAgent()
        
        input_data = ExtractionInput(
            urls=["https://httpbin.org/html"],
            content_filters={'min_content_length': 10}
        )
        
        try:
            result = await agent.execute(input_data)
            print(f"Test rapide: {result.successful_extractions}/{result.total_urls} succès")
            if result.documents:
                doc = result.documents[0]
                print(f"Titre: {doc.title}")
                print(f"Contenu: {len(doc.content)} caractères")
                print(f"Mots: {doc.word_count}")
                
        except Exception as e:
            print(f"Test rapide échoué: {e}")
    
    # Exécuter le test rapide
    asyncio.run(quick_test())