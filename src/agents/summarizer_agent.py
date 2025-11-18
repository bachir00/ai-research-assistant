"""
Agent Summarizer - Analyse et résumé de documents.
Crée des résumés structurés et des analyses approfondies des documents extraits.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from src.agents.base_agent import BaseAgent
from src.models.document_models import Document, DocumentSummary, SummarizationOutput, KeyPoint, Citation
from src.models.state_models import AgentType
from src.services.llm_service import LLMManager, LLMError
from src.services.text_chunking import ChunkingManager, TextChunk
from src.core.logging import setup_logger
from config.prompts import SUMMARIZER_PROMPTS, SYSTEM_PROMPTS
import hashlib
import re


class SummarizationInput:
    """Input pour l'agent Summarizer."""
    
    def __init__(
        self,
        documents: List[Document],
        summary_options: Optional[Dict[str, Any]] = None
    ):
        self.documents = documents
        self.summary_options = summary_options or {}
        
        # Options par défaut
        self.include_sentiment = self.summary_options.get('include_sentiment', True)
        self.include_citations = self.summary_options.get('include_citations', True)
        self.max_key_points = self.summary_options.get('max_key_points', 5)
        self.detailed_analysis = self.summary_options.get('detailed_analysis', True)
        self.chunk_large_docs = self.summary_options.get('chunk_large_docs', True)
        self.max_doc_size = self.summary_options.get('max_doc_size', 8000)  # caractères


class SummarizerAgent(BaseAgent):
    """
    Agent responsable de l'analyse et du résumé de documents.
    
    Fonctionnalités:
    - Résumé exécutif et détaillé
    - Extraction de points clés et arguments
    - Analyse de sentiment et biais
    - Gestion des documents longs via chunking
    - Citations et statistiques importantes
    - Évaluation de crédibilité
    """
    
    def __init__(
        self,
        max_retries: int = 2,
        timeout: float = 600.0  # 10 minutes pour traiter plusieurs documents
    ):
        super().__init__(
            agent_type=AgentType.READER,
            name="summarizer",
            max_retries=max_retries,
            timeout=timeout
        )
        
        # Services
        self.llm_manager = LLMManager()
        self.chunking_manager = ChunkingManager()
        
        # Configuration
        self.max_concurrent_summaries = 3 # maximum de résumés parallèles
        self.chunk_overlap_threshold = 6000  # Seuil pour le chunking en caractères
    
    def validate_input(self, input_data: SummarizationInput) -> bool:
        """
        Valide les données d'entrée pour la summarization.
        
        Args:
            input_data: Input contenant les documents à résumer
            
        Returns:
            True si les données sont valides
        """
        if not input_data.documents:
            self.logger.error("Aucun document fourni pour la summarization")
            return False
        
        if len(input_data.documents) > 20:  # Limite raisonnable
            self.logger.error(f"Trop de documents ({len(input_data.documents)}), maximum 20")
            return False
        
        # Vérifier que les documents ont du contenu
        valid_docs = [doc for doc in input_data.documents if doc.content and doc.content.strip()]
        if not valid_docs:
            self.logger.error("Aucun document avec contenu valide")
            return False
        
        return True
    
    async def process(self, input_data: SummarizationInput) -> SummarizationOutput:
        """
        Traite la summarization des documents.
        
        Args:
            input_data: Input contenant les documents à résumer
            
        Returns:
            SummarizationOutput avec tous les résumés
        """
        start_time = datetime.now()
        self.logger.info(f"Début summarization de {len(input_data.documents)} documents")
        
        # Filtrer les documents valides
        valid_documents = [doc for doc in input_data.documents if doc.content and doc.content.strip()]
        self.logger.info(f"Documents valides à traiter: {len(valid_documents)}")
        
        try:
            # Traitement parallèle des résumés
            summaries = await self._summarize_all_documents(valid_documents, input_data)
            
            # Analyse globale
            global_analysis = await self._perform_global_analysis(summaries)
            
            # Calcul des métriques
            total_processing_time = (datetime.now() - start_time).total_seconds()
            average_credibility = self._calculate_average_credibility(summaries)
            
            # Création du résultat
            result = SummarizationOutput(
                summaries=summaries,
                total_documents=len(input_data.documents),
                total_processing_time=total_processing_time,
                average_credibility=average_credibility,
                common_themes=global_analysis.get('common_themes', []),
                consensus_points=global_analysis.get('consensus_points', []),
                conflicting_views=global_analysis.get('conflicting_views', [])
            )
            
            self.logger.info(
                f"Summarization terminée: {len(summaries)} résumés créés en {total_processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la summarization: {str(e)}")
            raise
    
    async def _summarize_all_documents(
        self, 
        documents: List[Document], 
        input_data: SummarizationInput
    ) -> List[DocumentSummary]:
        """Résume tous les documents en parallèle."""
        semaphore = asyncio.Semaphore(self.max_concurrent_summaries)
        
        async def summarize_single(doc: Document) -> DocumentSummary:
            async with semaphore:
                try:
                    return await self._summarize_document(doc, input_data)
                except Exception as e:
                    self.logger.error(f"Erreur résumé document {doc.title}: {e}")
                    # Créer un résumé d'erreur minimal
                    return self._create_error_summary(doc, str(e))
        
        # Lancer tous les résumés en parallèle
        tasks = [summarize_single(doc) for doc in documents]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les résultats valides
        valid_summaries = []
        for summary in summaries:
            if isinstance(summary, DocumentSummary):
                valid_summaries.append(summary)
            else:
                self.logger.error(f"Résumé invalide: {summary}")
        
        return valid_summaries
    
    async def _summarize_document(self, document: Document, input_data: SummarizationInput) -> DocumentSummary:
        """Résume un document individuel."""
        start_time = datetime.now()
        doc_id = self._generate_document_id(document)
        
        self.logger.info(f"Résumé document: {document.title} ({len(document.content)} caractères)")
        
        # Décider si chunking nécessaire
        if (input_data.chunk_large_docs and 
            len(document.content) > self.chunk_overlap_threshold):
            summary = await self._summarize_large_document(document, input_data)
        else:
            summary = await self._summarize_standard_document(document, input_data)
        
        # Finaliser le résumé
        processing_time = (datetime.now() - start_time).total_seconds()
        summary.document_id = doc_id
        summary.processing_time = processing_time
        summary.processed_at = datetime.now()
        
        return summary
    
    async def _summarize_standard_document(
        self, 
        document: Document, 
        input_data: SummarizationInput
    ) -> DocumentSummary:
        """Résume un document de taille standard."""
        
        # Préparer le contexte
        context = {
            'title': document.title,
            'author': document.author or "Non spécifié",
            'url': str(document.url),
            'content': document.content
        }
        
        # Tâches parallèles
        tasks = []
        
        # 1. Résumé exécutif
        exec_prompt = SUMMARIZER_PROMPTS['executive_summary'].format(**context)
        tasks.append(self._get_llm_response(exec_prompt, "executive_summary"))
        
        # 2. Analyse détaillée
        if input_data.detailed_analysis:
            detailed_prompt = SUMMARIZER_PROMPTS['detailed_analysis'].format(**context)
            tasks.append(self._get_llm_response(detailed_prompt, "detailed_analysis"))
        
        # 3. Analyse de sentiment (optionnelle)
        if input_data.include_sentiment:
            sentiment_prompt = SUMMARIZER_PROMPTS['sentiment_analysis'].format(**context)
            tasks.append(self._get_llm_response(sentiment_prompt, "sentiment_analysis"))
        
        # Exécuter les tâches
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Parser les résultats
        executive_summary = ""
        detailed_summary = ""
        key_points = []
        sentiment = None
        credibility_score = None
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Erreur tâche {i}: {result}")
                continue
            
            task_type, content = result
            
            if task_type == "executive_summary":
                executive_summary = content
            elif task_type == "detailed_analysis":
                # Parser l'analyse détaillée
                parsed = self._parse_detailed_analysis(content)
                detailed_summary = parsed.get('summary', content)
                key_points = parsed.get('key_points', [])
            elif task_type == "sentiment_analysis":
                # Parser l'analyse de sentiment
                parsed = self._parse_sentiment_analysis(content)
                sentiment = parsed.get('sentiment')
                credibility_score = parsed.get('credibility_score')
        
        # Créer le résumé
        summary = DocumentSummary(
            document_id="",  # Sera rempli plus tard
            title=document.title,
            url=document.url,
            executive_summary=executive_summary,
            detailed_summary=detailed_summary,
            key_points=key_points[:input_data.max_key_points],
            sentiment=sentiment,
            credibility_score=credibility_score
        )
        
        return summary
    
    async def _summarize_large_document(
        self, 
        document: Document, 
        input_data: SummarizationInput
    ) -> DocumentSummary:
        """Résume un document long via chunking."""
        self.logger.info(f"Chunking document long: {document.title}")
        
        # Découper le document
        chunks = self.chunking_manager.chunk_document(
            document.content,
            strategy="default",
            preserve_structure=True
        )
        
        self.logger.info(f"Document découpé en {len(chunks)} chunks")
        
        # Résumer chaque chunk
        chunk_summaries = await self._summarize_chunks(chunks, document)
        
        # Synthétiser les résumés partiels
        synthesis = await self._synthesize_chunk_summaries(chunk_summaries, document)
        
        return synthesis
    
    async def _summarize_chunks(self, chunks: List[TextChunk], document: Document) -> List[str]:
        """Résume chaque chunk individuellement en parallèle."""
        async def summarize_chunk(chunk: TextChunk) -> str:
            context = {
                'title': document.title,
                'chunk_index': chunk.chunk_id,
                'total_chunks': chunk.total_chunks,
                'chunk_content': chunk.content
            }
            prompt = SUMMARIZER_PROMPTS['chunked_summary'].format(**context)
            try:
                return await self.llm_manager.get_completion(
                    prompt,
                    system_prompt=SYSTEM_PROMPTS['summarizer']
                )
            except Exception as e:
                self.logger.error(f"Erreur résumé chunk {chunk.chunk_id}: {e}")
                return f"Erreur résumé chunk {chunk.chunk_id}"

        # Parallélisation sur tous les chunks
        tasks = [summarize_chunk(chunk) for chunk in chunks]
        summaries = await asyncio.gather(*tasks)
        return summaries
    
    async def _synthesize_chunk_summaries(
        self, 
        chunk_summaries: List[str], 
        document: Document
    ) -> DocumentSummary:
        """Synthétise les résumés de chunks en un résumé unifié."""
        
        # Combiner tous les résumés partiels
        combined_summaries = "\n\n".join([
            f"Partie {i+1}: {summary}" 
            for i, summary in enumerate(chunk_summaries)
        ])
        
        context = {
            'partial_summaries': combined_summaries,
            'title': document.title,
            'url': str(document.url)
        }
        
        # Synthèse finale
        synthesis_prompt = SUMMARIZER_PROMPTS['synthesis'].format(**context)
        
        try:
            synthesis_result = await self.llm_manager.get_completion(
                synthesis_prompt,
                system_prompt=SYSTEM_PROMPTS['summarizer']
            )
            
            # Parser le résultat de synthèse
            parsed = self._parse_synthesis_result(synthesis_result)
            
            summary = DocumentSummary(
                document_id="",
                title=document.title,
                url=document.url,
                executive_summary=parsed.get('executive_summary', ''),
                detailed_summary=parsed.get('detailed_summary', ''),
                key_points=parsed.get('key_points', []),
                sentiment=parsed.get('sentiment'),
                credibility_score=parsed.get('credibility_score')
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Erreur synthèse finale: {e}")
            # Fallback: créer un résumé basique
            return self._create_basic_summary_from_chunks(chunk_summaries, document)
    
    async def _get_llm_response(self, prompt: str, task_type: str) -> tuple:
        """Obtient une réponse LLM pour une tâche spécifique."""
        try:
            response = await self.llm_manager.get_completion(
                prompt,
                system_prompt=SYSTEM_PROMPTS['summarizer'],
                temperature=0.3,
                max_tokens=2000
            )
            return task_type, response
        except Exception as e:
            self.logger.error(f"Erreur LLM pour {task_type}: {e}")
            return task_type, f"Erreur: {str(e)}"
    
    def _parse_detailed_analysis(self, content: str) -> Dict[str, Any]:
        """Parse l'analyse détaillée pour extraire les composants."""
        # Implémentation simplifiée - à améliorer selon le format de réponse
        result = {'summary': content, 'key_points': []}
        
        # Chercher les points clés (format: - Point clé)
        import re
        key_point_pattern = r'^[-•]\s*(.+)$'
        lines = content.split('\n')
        
        current_key_points = []
        for line in lines:
            match = re.match(key_point_pattern, line.strip())
            if match:
                point_text = match.group(1).strip()
                if len(point_text) > 10:  # Filtrer les points trop courts
                    key_point = KeyPoint(
                        title=point_text[:50] + "..." if len(point_text) > 50 else point_text,
                        content=point_text,
                        importance=0.8,  # Score par défaut
                        category="general"
                    )
                    current_key_points.append(key_point)
        
        result['key_points'] = current_key_points
        return result
    
    def _parse_sentiment_analysis(self, content: str) -> Dict[str, Any]:
        """Parse l'analyse de sentiment."""
        result = {}
        
        # Extraction simplifiée
        content_lower = content.lower()
        
        if 'positif' in content_lower:
            result['sentiment'] = 'positif'
        elif 'négatif' in content_lower:
            result['sentiment'] = 'négatif'
        else:
            result['sentiment'] = 'neutre'
        
        # Chercher un score de crédibilité
        import re
        
        # Chercher un pattern comme "Crédibilité: 0.8" ou "0.8"
        credibility_pattern = r'crédibilité\s*:?\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*\/\s*[1510]|(\d+(?:\.\d+)?)\s*%'
        match = re.search(credibility_pattern, content_lower)
        if match:
            score = float(match.group(1) or match.group(2) or match.group(3))
            if score > 1:  # Si en pourcentage
                score = score / 100
            result['credibility_score'] = min(max(score, 0.0), 1.0)
        else:
            result['credibility_score'] = 0.5  # Valeur par défaut
        
        return result
        
        return result
    
    def _parse_synthesis_result(self, content: str) -> Dict[str, Any]:
        """Parse le résultat de synthèse."""
        # Version simplifiée - à améliorer
        return {
            'executive_summary': content[:200] + "..." if len(content) > 200 else content,
            'detailed_summary': content,
            'key_points': [],
            'sentiment': 'neutre',
            'credibility_score': 0.7
        }
    
    def _create_basic_summary_from_chunks(
        self, 
        chunk_summaries: List[str], 
        document: Document
    ) -> DocumentSummary:
        """Crée un résumé basique à partir des résumés de chunks."""
        combined = " ".join(chunk_summaries)
        
        return DocumentSummary(
            document_id="",
            title=document.title,
            url=document.url,
            executive_summary=combined[:200] + "..." if len(combined) > 200 else combined,
            detailed_summary=combined,
            key_points=[],
            sentiment="neutre",
            credibility_score=0.5
        )
    
    def _create_error_summary(self, document: Document, error: str) -> DocumentSummary:
        """Crée un résumé d'erreur minimal."""
        return DocumentSummary(
            document_id=self._generate_document_id(document),
            title=document.title,
            url=document.url,
            executive_summary=f"Erreur lors du résumé: {error}",
            detailed_summary=f"Le résumé de ce document n'a pas pu être généré: {error}",
            key_points=[],
            sentiment=None,
            credibility_score=None
        )
    
    def _generate_document_id(self, document: Document) -> str:
        """Génère un ID unique pour un document."""
        content_hash = hashlib.md5(f"{document.url}{document.title}".encode()).hexdigest()
        return f"doc_{content_hash[:8]}"
    
    async def _perform_global_analysis(self, summaries: List[DocumentSummary]) -> Dict[str, List[str]]:
        """Effectue une analyse globale de tous les résumés."""
        if len(summaries) < 2:
            return {'common_themes': [], 'consensus_points': [], 'conflicting_views': []}
        
        # Combiner tous les résumés pour l'analyse
        all_summaries = "\n\n".join([
            f"Document: {s.title}\nRésumé: {s.detailed_summary}"
            for s in summaries
        ])
        
        # Prompt d'analyse globale
        global_prompt = f"""
            Analyse les résumés de documents suivants et identifie:

            1. **Thèmes communs** : Les sujets qui reviennent dans plusieurs documents
            2. **Points de consensus** : Les idées sur lesquelles les sources s'accordent
            3. **Points conflictuels** : Les idées contradictoires entre les sources

            RÉSUMÉS:
            {all_summaries}

            Format ta réponse avec des sections claires et des listes à puces.
            """
        
        try:
            response = await self.llm_manager.get_completion(
                global_prompt,
                system_prompt="Tu es un expert en analyse comparative de documents."
            )
            
            # Parser la réponse (implémentation simplifiée)
            return self._parse_global_analysis(response)
            
        except Exception as e:
            self.logger.error(f"Erreur analyse globale: {e}")
            return {'common_themes': [], 'consensus_points': [], 'conflicting_views': []}
    
    def _parse_global_analysis(self, content: str) -> Dict[str, List[str]]:
        """Parse l'analyse globale."""
        # Implémentation simplifiée
        lines = content.split('\n')
        
        result = {
            'common_themes': [],
            'consensus_points': [],
            'conflicting_views': []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Détecter les sections
            if 'thème' in line.lower() or 'theme' in line.lower():
                current_section = 'common_themes'
            elif 'consensus' in line.lower():
                current_section = 'consensus_points'
            elif 'conflict' in line.lower() or 'contradictoire' in line.lower():
                current_section = 'conflicting_views'
            elif line.startswith('-') or line.startswith('•'):
                # Point de liste
                if current_section:
                    point = line[1:].strip()
                    if len(point) > 5:  # Filtrer les points trop courts
                        result[current_section].append(point)
        
        return result
    
    def _calculate_average_credibility(self, summaries: List[DocumentSummary]) -> Optional[float]:
        """Calcule le score de crédibilité moyen."""
        scores = [s.credibility_score for s in summaries if s.credibility_score is not None]
        
        if not scores:
            return None
        
        return sum(scores) / len(scores)

    #fonction summary from content extraction result
    async def process_from_extraction_result(
        self,
        extraction_result: 'ExtractionResult'
    ) -> SummarizationOutput:
        """
        Traite la summarization à partir d'un ExtractionResult.
        
        Args:
            extraction_result: Résultat de l'extraction de contenu
        Returns:
            SummarizationOutput avec tous les résumés
        """
        # Préparer l'input de summarization
        input_data = SummarizationInput(
            documents=extraction_result.documents,
            summary_options={
                'include_sentiment': True,
                'include_citations': True,
                'max_key_points': 5,
                'detailed_analysis': True,
                'chunk_large_docs': True
            }

        )
        
        if not self.validate_input(input_data):
            self.logger.error("Input ExtractionResult invalide pour la summarization")
            raise ValueError("Input ExtractionResult invalide pour la summarization")
        
        # Appeler le processus principal de summarization
        return await self.process(input_data)
      



# Exemple d'utilisation
if __name__ == "__main__":
    import asyncio
    import json
    from src.models.document_models import ExtractionResult

    def save_summarization_output(output, filename=None):
        """Sauvegarde un SummarizationOutput au format JSON."""
        from datetime import datetime
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summarization_output_{len(output.summaries)}docs_{timestamp}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
            return filename
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return None

    async def summarize_from_extraction_file():
        # Charger ExtractionResult
        extraction_file = "extraction_result_2docs_20251116_141527.json"
        try:
            with open(extraction_file, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
            extraction_result = ExtractionResult(**extraction_data)
        except Exception as e:
            print(f"Erreur chargement ExtractionResult: {e}")
            return

        

        summarizer = SummarizerAgent()

        output = await summarizer.process_from_extraction_result(extraction_result)

        # Sauvegarde
        filename = save_summarization_output(output)
        if filename:
            print(f"✅ Résumés sauvegardés dans: {filename}")
        else:
            print("❌ Erreur lors de la sauvegarde du résumé.")

        # Affichage rapide
        for summary in output.summaries:
            print(f"\nRésumé pour {summary.title}:")
            print(f"Résumé exécutif: {summary.executive_summary[:200]}...")
            print(f"Points clés: {[kp.title for kp in summary.key_points]}")
            print(f"Sentiment: {summary.sentiment}")
            print(f"Score de crédibilité: {summary.credibility_score}")
        print(f"Temps total de traitement: {output.total_processing_time:.2f}s")
        print(f"Score de crédibilité moyen: {output.average_credibility}")

    asyncio.run(summarize_from_extraction_file())