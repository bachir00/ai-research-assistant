"""
Agent Content Extractor - Extraction et nettoyage de contenu web.
Extrait le contenu de pages web, PDFs et autres documents.
"""

import asyncio
from typing import List, Optional
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.models.document_models import Document, ExtractionInput, ExtractionResult
from src.models.research_models import ResearchOutput
from src.models.state_models import AgentState, AgentType
from src.services.content_extraction import ContentExtractionManager, ContentExtractionError
from src.core.logging import setup_logger


class ContentExtractorAgent(BaseAgent[ExtractionInput, ExtractionResult]):
    """
    Agent responsable de l'extraction de contenu depuis des URLs.
    
    Fonctionnalités:
    - Extraction de contenu HTML avec nettoyage intelligent
    - Support des PDFs et autres formats
    - Traitement parallèle de plusieurs URLs
    - Gestion des erreurs et retry automatique
    - Structuration et nettoyage du contenu
    """
    
    def __init__(self, max_concurrent_extractions: int = 5, max_retries: int = 2):
        super().__init__(
            agent_type=AgentType.CONTENT_EXTRACTOR,
            name="content_extractor",
            max_retries=max_retries,
            timeout=300.0  # 5 minutes
        )
        self.extraction_manager = ContentExtractionManager(
            max_concurrent=max_concurrent_extractions,
            max_retries=max_retries
        )
    
    def validate_input(self, input_data: ExtractionInput) -> bool:
        """
        Valide les données d'entrée pour l'extraction.
        
        Args:
            input_data: Input contenant les URLs à extraire
            
        Returns:
            True si les données sont valides
        """
        if not input_data.urls:
            self.logger.error("Aucune URL fournie pour l'extraction")
            return False
        
        if len(input_data.urls) > 50:  # Limite raisonnable
            self.logger.error(f"Trop d'URLs ({len(input_data.urls)}), maximum 50")
            return False
        
        # Filtrer les URLs valides
        valid_urls = self._filter_valid_urls(input_data.urls)
        if not valid_urls:
            self.logger.error("Aucune URL valide trouvée")
            return False
        
        return True
    
    async def process_from_research_output(self, research_output: ResearchOutput) -> ExtractionResult:
        """
        Traite directement un ResearchOutput pour extraire le contenu des URLs.
        
        Args:
            research_output: Résultats de recherche avec URLs à extraire
            
        Returns:
            ExtractionResult avec les documents extraits
        """
        # Extraire les URLs des résultats de recherche (conversion en string)
        urls = [str(result.url) for result in research_output.results]
        
        self.logger.info(f"Extraction de contenu depuis ResearchOutput: {len(urls)} URLs")
        self.logger.info(f"Sujet de recherche: {research_output.query.topic}")
        
        # Créer l'input d'extraction
        extraction_input = ExtractionInput(
            urls=urls,
            content_filters={
                'min_content_length': 200,  # Minimum de contenu
                'max_content_length': 50000,  # Maximum pour éviter les textes trop longs
                'required_keywords': research_output.query.keywords  # Filtrer par mots-clés de recherche
            },
            extraction_options={
                'source_query': research_output.query.topic,
                'search_keywords': research_output.query.keywords
            }
        )
        
        # Traiter avec la méthode normale
        return await self.process(extraction_input)
    
    async def process(self, input_data: ExtractionInput) -> ExtractionResult:
        """
        Exécute l'extraction de contenu pour les URLs fournies.
        
        Args:
            input_data: Input contenant les URLs à extraire et les options
            
        Returns:
            ExtractionResult avec les documents extraits
            
        Raises:
            ValueError: Si les URLs sont invalides
            ContentExtractionError: Si l'extraction échoue
        """
        start_time = datetime.now()
        self.logger.info(f"Début extraction de contenu pour {len(input_data.urls)} URLs")
        
        # Filtrer les URLs valides (validation déjà faite dans validate_input)
        valid_urls = self._filter_valid_urls(input_data.urls)
        self.logger.info(f"URLs valides à traiter: {len(valid_urls)}/{len(input_data.urls)}")
        
        try:
            # Extraction du contenu
            documents = await self._extract_all_content(valid_urls, input_data)
            
            # Post-traitement des documents
            processed_documents = self._post_process_documents(documents, input_data)
            
            # Calcul des statistiques
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Identifier les URLs qui ont échoué
            successful_urls = {str(doc.url) for doc in processed_documents}
            failed_urls = [url for url in valid_urls if url not in successful_urls]
            
            # Création du résultat
            result = ExtractionResult(
                documents=processed_documents,
                total_urls=len(input_data.urls),
                successful_extractions=len(processed_documents),
                failed_extractions=len(input_data.urls) - len(processed_documents),
                failed_urls=failed_urls,
                execution_time=execution_time,
                extraction_stats=self._calculate_stats(processed_documents)
            )
            
            self.logger.info(
                f"Extraction terminée: {result.successful_extractions}/{result.total_urls} "
                f"succès en {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction: {str(e)}")
            raise ContentExtractionError(f"Échec de l'extraction de contenu: {str(e)}")
    
    def _filter_valid_urls(self, urls: List[str]) -> List[str]:
        """Filtre et valide les URLs."""
        import re
        from urllib.parse import urlparse
        
        valid_urls = []
        url_pattern = re.compile(
            r'^https?://'  # http:// ou https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        for url in urls:
            if not url or not isinstance(url, str):
                self.logger.warning(f"URL invalide ignorée: {url}")
                continue
            
            url = url.strip()
            if not url:
                continue
            
            # Validation du format
            if not url_pattern.match(url):
                self.logger.warning(f"Format URL invalide: {url}")
                continue
            
            # Validation avec urlparse
            try:
                parsed = urlparse(url)
                if not parsed.netloc:
                    self.logger.warning(f"URL sans domaine: {url}")
                    continue
                
                valid_urls.append(url)
                
            except Exception as e:
                self.logger.warning(f"Erreur de parsing URL {url}: {e}")
                continue
        
        return valid_urls
    
    async def _extract_all_content(self, urls: List[str], input_data: ExtractionInput) -> List[Document]:
        """Extrait le contenu de toutes les URLs."""
        try:
            # Utiliser le gestionnaire d'extraction
            documents = await self.extraction_manager.extract_multiple(urls)
            
            # Appliquer les filtres si spécifiés
            if input_data.content_filters:
                documents = self._apply_content_filters(documents, input_data.content_filters)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction multiple: {str(e)}")
            raise
    
    def _apply_content_filters(self, documents: List[Document], filters: dict) -> List[Document]:
        """Applique les filtres de contenu aux documents."""
        filtered_documents = []
        
        for doc in documents:
            # Filtrer par longueur minimale
            min_length = filters.get('min_content_length', 100)
            if len(doc.content) < min_length:
                self.logger.debug(f"Document {doc.title} trop court: {len(doc.content)} caractères")
                continue
            
            # Filtrer par longueur maximale
            max_length = filters.get('max_content_length', 100000)
            if len(doc.content) > max_length:
                self.logger.debug(f"Document {doc.title} trop long, troncature")
                doc.content = doc.content[:max_length] + "... [Contenu tronqué]"
            
            # Filtrer par langue si spécifiée
            required_language = filters.get('language')
            if required_language and doc.language != required_language:
                self.logger.debug(f"Document {doc.title} ignoré: langue {doc.language}")
                continue
            
            # Filtrer par mots-clés si spécifiés
            required_keywords = filters.get('required_keywords', [])
            if required_keywords:
                content_lower = doc.content.lower()
                if not any(keyword.lower() in content_lower for keyword in required_keywords):
                    self.logger.debug(f"Document {doc.title} ignoré: mots-clés manquants")
                    continue
            
            filtered_documents.append(doc)
        
        self.logger.info(f"Filtres appliqués: {len(filtered_documents)}/{len(documents)} documents retenus")
        return filtered_documents
    
    def _post_process_documents(self, documents: List[Document], input_data: ExtractionInput) -> List[Document]:
        """Post-traitement des documents extraits."""
        processed_docs = []
        
        for doc in documents:
            # Nettoyage supplémentaire du contenu
            doc.content = self._clean_content(doc.content)
            
            # Recalcul du nombre de mots après nettoyage
            doc.word_count = len(doc.content.split())
            
            # Validation finale
            if self._is_valid_document(doc, input_data):
                processed_docs.append(doc)
            else:
                self.logger.debug(f"Document {doc.title} rejeté lors de la validation finale")
        
        return processed_docs
    
    def _clean_content(self, content: str) -> str:
        """Nettoyage avancé du contenu."""
        import re
        
        if not content:
            return ""
        
        # Supprimer les caractères de contrôle
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        # Normaliser les espaces
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Normaliser les sauts de ligne
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Supprimer les espaces en début et fin de lignes
        lines = content.split('\n')
        lines = [line.strip() for line in lines]
        content = '\n'.join(lines)
        
        # Supprimer les lignes vides multiples
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def _is_valid_document(self, doc: Document, input_data: ExtractionInput) -> bool:
        """Valide un document extrait."""
        # Vérifications de base
        if not doc.content or not doc.content.strip():
            return False
        
        if len(doc.content) < 50:  # Contenu trop court
            return False
        
        # Vérification du ratio texte/contenu (détecter les pages avec peu de contenu)
        if doc.word_count < 20:
            return False
        
        # Vérifications spécifiques aux options d'entrée
        if hasattr(input_data, 'min_quality_score'):
            quality_score = self._calculate_content_quality(doc)
            if quality_score < input_data.min_quality_score:
                return False
        
        return True
    
    def _calculate_content_quality(self, doc: Document) -> float:
        """Calcule un score de qualité pour le contenu (0-1)."""
        score = 0.0
        
        # Points pour la longueur
        if doc.word_count > 100:
            score += 0.3
        elif doc.word_count > 50:
            score += 0.1
        
        # Points pour la structure
        if doc.title and len(doc.title) > 10:
            score += 0.2
        
        if doc.author:
            score += 0.1
        
        if doc.published_date:
            score += 0.1
        
        # Points pour la richesse du contenu
        content = doc.content.lower()
        if any(marker in content for marker in ['conclusion', 'introduction', 'sommaire']):
            score += 0.2
        
        # Pénalité pour contenu répétitif
        lines = doc.content.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) > 0:
            uniqueness_ratio = len(unique_lines) / len(lines)
            if uniqueness_ratio < 0.5:
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_stats(self, documents: List[Document]) -> dict:
        """Calcule les statistiques d'extraction."""
        if not documents:
            return {
                'total_words': 0,
                'average_words_per_doc': 0,
                'doc_types': {},
                'languages': {},
                'has_authors': 0,
                'has_dates': 0
            }
        
        total_words = sum(doc.word_count for doc in documents)
        
        # Compter les types de documents
        doc_types = {}
        for doc in documents:
            doc_type = doc.doc_type.value if doc.doc_type else 'unknown'
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Compter les langues
        languages = {}
        for doc in documents:
            lang = doc.language or 'unknown'
            languages[lang] = languages.get(lang, 0) + 1
        
        # Compter les métadonnées
        has_authors = sum(1 for doc in documents if doc.author)
        has_dates = sum(1 for doc in documents if doc.published_date)
        
        return {
            'total_words': total_words,
            'average_words_per_doc': total_words // len(documents),
            'doc_types': doc_types,
            'languages': languages,
            'has_authors': has_authors,
            'has_dates': has_dates
        }


# Fonction utilitaire pour les tests
async def extract_content_from_urls(urls: List[str], **options) -> List[Document]:
    """
    Fonction utilitaire pour extraire du contenu depuis une liste d'URLs.
    
    Args:
        urls: Liste des URLs à extraire
        **options: Options d'extraction (filters, etc.)
    
    Returns:
        Liste des documents extraits
    """
    agent = ContentExtractorAgent()
    
    input_data = ExtractionInput(
        urls=urls,
        content_filters=options.get('content_filters', {}),
        extraction_options=options.get('extraction_options', {})
    )
    
    result = await agent.execute(input_data)
    return result.documents


# Fonction utilitaire pour l'intégration avec le Researcher
async def extract_from_search_results(search_results: List[dict]) -> List[Document]:
    """
    Extrait le contenu depuis des résultats de recherche.
    
    Args:
        search_results: Résultats de recherche avec URLs
        
    Returns:
        Liste des documents extraits
    """
    urls = []
    for result in search_results:
        if isinstance(result, dict) and 'url' in result:
            urls.append(result['url'])
        elif hasattr(result, 'url'):
            urls.append(result.url)
    
    if not urls:
        return []
    
    return await extract_content_from_urls(urls)


# Fonctions utilitaires pour la sauvegarde
def save_extraction_result(result: ExtractionResult, filename: str = None) -> str:
    """
    Sauvegarde un ExtractionResult dans un fichier JSON.
    
    Args:
        result: Résultat d'extraction à sauvegarder
        filename: Nom du fichier (optionnel)
        
    Returns:
        Nom du fichier sauvegardé
    """
    import json
    from datetime import datetime
    
    if not filename:
        # Générer un nom de fichier basé sur le nombre de documents et timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"extraction_result_{result.successful_extractions}docs_{timestamp}.json"
    
    try:
        # Conversion en dictionnaire avec sérialisation des dates
        result_dict = result.model_dump(mode='json')
        
        # Sauvegarde dans le fichier
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        return filename
        
    except Exception as e:
        raise Exception(f"Erreur lors de la sauvegarde: {e}")


def load_extraction_result(filename: str) -> ExtractionResult:
    """
    Charge un ExtractionResult depuis un fichier JSON.
    
    Args:
        filename: Nom du fichier à charger
        
    Returns:
        ExtractionResult chargé
    """
    import json
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruction de l'ExtractionResult
        return ExtractionResult(**data)
        
    except Exception as e:
        raise Exception(f"Erreur lors du chargement: {e}")


# Configuration du logger pour l'agent
logger = setup_logger("ContentExtractorAgent")
# Exemple d'utilisation
if __name__ == "__main__":
    import asyncio
    import json
    from src.models.research_models import ResearchOutput
    
    async def test_with_research_output():
        """Test avec un fichier ResearchOutput sauvegardé."""
        # Charger le ResearchOutput depuis le fichier JSON le plus récent
        research_file = "research_output_impact_de_lintelligence_artifi_20251116_141136.json"
        
        try:
            # Charger le ResearchOutput
            with open(research_file, 'r', encoding='utf-8') as f:
                research_data = json.load(f)
            
            research_output = ResearchOutput(**research_data)
            logger.info(f"=== CHARGEMENT DU RESEARCH OUTPUT ===")
            logger.info(f"Sujet: {research_output.query.topic}")
            logger.info(f"URLs à extraire: {len(research_output.results)}")
            
            # Créer l'agent et traiter
            agent = ContentExtractorAgent()
            
            logger.info(f"=== DÉBUT DE L'EXTRACTION DE CONTENU ===")
            extraction_result = await agent.process_from_research_output(research_output)
            
            logger.info(f"=== RÉSULTATS D'EXTRACTION ===")
            logger.info(f"URLs traitées: {extraction_result.total_urls}")
            logger.info(f"Extractions réussies: {extraction_result.successful_extractions}")
            logger.info(f"Extractions échouées: {extraction_result.failed_extractions}")
            logger.info(f"Temps d'exécution: {extraction_result.execution_time:.2f}s")
            
            # Afficher les détails des documents extraits
            for i, doc in enumerate(extraction_result.documents, 1):
                logger.info(f"\n{i}. {doc.title}")
                logger.info(f"   URL: {doc.url}")
                logger.info(f"   Mots: {doc.word_count}")
                logger.info(f"   Langue: {doc.language}")
                logger.info(f"   Type: {doc.doc_type}")
                logger.info(f"   Contenu (aperçu): {doc.content[:200]}...")
            
            # URLs qui ont échoué
            if extraction_result.failed_urls:
                logger.info(f"\n❌ URLs en échec:")
                for url in extraction_result.failed_urls:
                    logger.info(f"   • {url}")
            
            # === SAUVEGARDE DE L'EXTRACTION RESULT ===
            logger.info(f"\n=== SAUVEGARDE DE L'EXTRACTION RESULT ===")
            
            try:
                filename = save_extraction_result(extraction_result)
                logger.info(f"✅ ExtractionResult sauvegardé dans: {filename}")
                
                # Affichage du contenu sauvegardé
                logger.info("📄 Contenu sauvegardé:")
                logger.info(f"  • Documents extraits: {len(extraction_result.documents)}")
                logger.info(f"  • Temps d'extraction: {extraction_result.execution_time:.2f}s")
                logger.info(f"  • Statistiques: {extraction_result.extraction_stats}")
                
                # Test de chargement pour vérifier l'intégrité
                logger.info("=== Test de chargement ===")
                loaded_result = load_extraction_result(filename)
                logger.info(f"✅ ExtractionResult rechargé avec succès")
                logger.info(f"  • Vérification: {len(loaded_result.documents)} documents chargés")
                
                # Comparaison des données
                if loaded_result.successful_extractions == extraction_result.successful_extractions:
                    logger.info("✅ Intégrité des données vérifiée")
                else:
                    logger.error("❌ Erreur d'intégrité des données")
                
                # Affichage du format JSON pour référence
                logger.info("\n📋 EXEMPLE DE FORMAT JSON SAUVEGARDÉ:")
                logger.info("-" * 50)
                
                # Créer un exemple compact pour l'affichage
                example_result = {
                    "documents": [
                        {
                            "title": doc.title,
                            "url": str(doc.url),
                            "content": doc.content[:200] + "...",
                            "word_count": doc.word_count,
                            "language": doc.language,
                            "doc_type": doc.doc_type.value if doc.doc_type else None
                        } for doc in extraction_result.documents[:2]  # Limiter à 2 documents
                    ],
                    "total_urls": extraction_result.total_urls,
                    "successful_extractions": extraction_result.successful_extractions,
                    "failed_extractions": extraction_result.failed_extractions,
                    "failed_urls": extraction_result.failed_urls,
                    "execution_time": extraction_result.execution_time,
                    "extraction_stats": extraction_result.extraction_stats
                }
                
                print(json.dumps(example_result, indent=2, ensure_ascii=False))
                
            except Exception as save_error:
                logger.error(f"❌ Erreur lors de la sauvegarde: {save_error}")
            
        except FileNotFoundError:
            logger.error(f"❌ Fichier ResearchOutput non trouvé: {research_file}")
            logger.info("Utilisation de l'exemple avec URLs directes...")
            await test_with_direct_urls()
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement: {e}")
    
    async def test_with_direct_urls():
        """Test avec des URLs directes."""
        urls = [
            'https://www.iana.org/help/example-domains',
        ]
        
        logger.info(f"=== TEST AVEC URLS DIRECTES ===")
        documents = await extract_content_from_urls(urls)
        for doc in documents:
            logger.info(f"Title: {doc.title}, URL: {doc.url}, Word Count: {doc.word_count}, Language: {doc.language}, Content Length: {len(doc.content)}")
    
    # Choisir le test à exécuter
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        asyncio.run(test_with_direct_urls())
    else:
        asyncio.run(test_with_research_output())