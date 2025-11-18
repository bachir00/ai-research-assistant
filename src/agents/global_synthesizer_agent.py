"""
Agent Global Synthesizer - Synthèse finale et génération de rapport.
Prend les résumés de l'agent Summarizer et génère un rapport final structuré.
"""

import asyncio
import hashlib
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.models.synthesis_models import (
    GlobalSynthesisInput, GlobalSynthesisOutput, FinalReport,
    ExecutiveSummary, ReportSection, SourceReference, Methodology,
    ReportType, ReportFormat
)
from src.models.document_models import DocumentSummary, SummarizationOutput
from src.models.state_models import AgentType
from src.services.llm_service import LLMManager, LLMError
from src.core.logging import setup_logger
from config.prompts import GLOBAL_SYNTHESIZER_PROMPTS, SYSTEM_PROMPTS


class GlobalSynthesizerAgent(BaseAgent[GlobalSynthesisInput, GlobalSynthesisOutput]):
    """
    Agent responsable de la synthèse finale et de la génération de rapport.
    
    Fonctionnalités:
    - Synthèse de multiples résumés de documents
    - Génération de rapport final structuré
    - Analyse transversale et identification de patterns
    - Évaluation de qualité et méthodologie
    - Support de différents formats de rapport
    - Génération de résumé exécutif
    """
    
    def __init__(
        self,
        max_retries: int = 2,
        timeout: float = 300.0  # 5 minutes pour la synthèse finale
    ):
        super().__init__(
            agent_type=AgentType.WRITER,
            name="global_synthesizer",
            max_retries=max_retries,
            timeout=timeout
        )
        
        # Services
        self.llm_manager = LLMManager()
        
        # Configuration
        self.max_concurrent_synthesis = 3  # Nombre de tâches parallèles max
        self.min_sources_for_analysis = 1  # Minimum de sources pour une analyse
        
    def validate_input(self, input_data: GlobalSynthesisInput) -> bool:
        """
        Valide les données d'entrée pour la synthèse globale.
        
        Args:
            input_data: Input contenant les résumés à synthétiser
            
        Returns:
            True si les données sont valides
        """
        if not input_data.summarization_output:
            self.logger.error("Aucune sortie de summarization fournie")
            return False
        
        if not input_data.summarization_output.summaries:
            self.logger.error("Aucun résumé disponible pour la synthèse")
            return False
        
        if len(input_data.summarization_output.summaries) < self.min_sources_for_analysis:
            self.logger.error(f"Minimum {self.min_sources_for_analysis} résumé(s) requis")
            return False
        
        if not input_data.original_topic or len(input_data.original_topic.strip()) < 3:
            self.logger.error("Sujet original manquant ou trop court")
            return False
        
        return True
    
    async def process(self, input_data: GlobalSynthesisInput) -> GlobalSynthesisOutput:
        """
        Traite la synthèse globale et génère le rapport final.
        
        Args:
            input_data: Input contenant les résumés et options de synthèse
            
        Returns:
            GlobalSynthesisOutput avec le rapport final
        """
        start_time = datetime.now()
        self.logger.info(f"Début synthèse globale pour: '{input_data.original_topic}'")
        self.logger.info(f"Nombre de résumés à synthétiser: {len(input_data.summarization_output.summaries)}")
        
        try:
            # Étape 1: Préparation des données
            prepared_data = self._prepare_synthesis_data(input_data)
            
            # Étape 2: Génération des sections du rapport en parallèle
            report_sections = await self._generate_report_sections(prepared_data, input_data)
            
            # Étape 3: Génération du résumé exécutif
            executive_summary = await self._generate_executive_summary(prepared_data, input_data)
            
            # Étape 4: Création de la méthodologie
            methodology = self._create_methodology(input_data)
            
            # Étape 5: Création des références de sources
            source_references = self._create_source_references(input_data.summarization_output.summaries)
            
            # Étape 6: Évaluation de qualité
            quality_scores = await self._assess_quality(input_data, report_sections)
            
            # Étape 7: Assemblage du rapport final
            final_report = self._assemble_final_report(
                input_data, 
                executive_summary,
                report_sections,
                methodology,
                source_references,
                quality_scores
            )
            
            # Étape 8: Génération des formats alternatifs
            formatted_outputs = await self._generate_formatted_outputs(final_report, input_data)
            
            # Calcul du temps de traitement
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Création du résultat
            result = GlobalSynthesisOutput(
                final_report=final_report,
                synthesis_metadata={
                    "synthesis_approach": "comprehensive",
                    "llm_model_used": "groq/llama-3.1-8b-instant",
                    "quality_checks_passed": quality_scores["confidence_score"] > 0.6
                },
                processing_stats={
                    "input_summaries": len(input_data.summarization_output.summaries),
                    "synthesis_time": processing_time,
                    "final_report_words": final_report.word_count,
                    "sections_generated": len(report_sections)
                },
                formatted_outputs=formatted_outputs
            )
            
            self.logger.info(f"Synthèse globale terminée en {processing_time:.2f}s")
            self.logger.info(f"Rapport final: {final_report.word_count} mots, {len(report_sections)} sections")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la synthèse globale: {str(e)}")
            raise
    
    def _prepare_synthesis_data(self, input_data: GlobalSynthesisInput) -> Dict[str, Any]:
        """Prépare les données nécessaires pour la synthèse."""
        summaries = input_data.summarization_output.summaries
        
        # Compilation des résumés
        document_summaries = []
        for i, summary in enumerate(summaries, 1):
            doc_summary = f"""
Document {i}: {summary.title}
URL: {summary.url}
Résumé exécutif: {summary.executive_summary}
Résumé détaillé: {summary.detailed_summary}
Sentiment: {summary.sentiment}
Crédibilité: {summary.credibility_score}
Points clés: {[kp.title for kp in summary.key_points]}
"""
            document_summaries.append(doc_summary.strip())
        
        return {
            "topic": input_data.original_topic,
            "document_summaries": "\n\n".join(document_summaries),
            "common_themes": input_data.summarization_output.common_themes,
            "consensus_points": input_data.summarization_output.consensus_points,
            "conflicting_views": input_data.summarization_output.conflicting_views,
            "summaries_count": len(summaries),
            "average_credibility": input_data.summarization_output.average_credibility
        }
    
    async def _generate_report_sections(
        self, 
        prepared_data: Dict[str, Any], 
        input_data: GlobalSynthesisInput
    ) -> List[ReportSection]:
        """Génère les sections principales du rapport."""
        
        # Tâches parallèles pour différentes sections
        tasks = []
        
        # 1. Synthèse principale
        main_synthesis_prompt = GLOBAL_SYNTHESIZER_PROMPTS['final_synthesis'].format(**prepared_data)
        tasks.append(self._get_llm_response(main_synthesis_prompt, "main_synthesis"))
        
        # 2. Analyse thématique
        thematic_prompt = GLOBAL_SYNTHESIZER_PROMPTS['thematic_analysis'].format(
            topic=prepared_data["topic"],
            summaries=prepared_data["document_summaries"]
        )
        tasks.append(self._get_llm_response(thematic_prompt, "thematic_analysis"))
        
        # Exécution des tâches en parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Traitement des résultats
        sections = []
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Erreur génération section: {result}")
                continue
            
            section_type, content = result
            
            if section_type == "main_synthesis":
                # Parser la synthèse principale en sections
                parsed_sections = self._parse_main_synthesis(content)
                sections.extend(parsed_sections)
            
            elif section_type == "thematic_analysis":
                # Ajouter l'analyse thématique comme section
                thematic_section = ReportSection(
                    title="Analyse Thématique Détaillée",
                    content=content,
                    order=2
                )
                sections.append(thematic_section)
        
        # Trier les sections par ordre
        sections.sort(key=lambda x: x.order)
        
        return sections
    
    async def _generate_executive_summary(
        self, 
        prepared_data: Dict[str, Any], 
        input_data: GlobalSynthesisInput
    ) -> ExecutiveSummary:
        """Génère le résumé exécutif."""
        
        # Préparation des données pour le résumé exécutif
        analysis_data = {
            "summaries": prepared_data["document_summaries"],
            "themes": prepared_data["common_themes"],
            "consensus": prepared_data["consensus_points"],
            "conflicts": prepared_data["conflicting_views"],
            "credibility": prepared_data["average_credibility"]
        }
        
        prompt = GLOBAL_SYNTHESIZER_PROMPTS['executive_summary'].format(
            topic=prepared_data["topic"],
            analysis_data=str(analysis_data)
        )
        
        try:
            response = await self.llm_manager.get_completion(
                prompt,
                system_prompt=SYSTEM_PROMPTS['global_synthesizer'],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Parser la réponse pour extraire les composants
            return self._parse_executive_summary(response)
            
        except Exception as e:
            self.logger.error(f"Erreur génération résumé exécutif: {e}")
            # Fallback: créer un résumé simple
            return self._create_fallback_executive_summary(prepared_data)
    
    def _create_methodology(self, input_data: GlobalSynthesisInput) -> Methodology:
        """Crée la description de la méthodologie utilisée."""
        
        analysis_methods = [
            "Extraction automatique de contenu web",
            "Analyse et résumé par intelligence artificielle",
            "Synthèse thématique transversale",
            "Évaluation de crédibilité des sources"
        ]
        
        limitations = [
            "Analyse limitée aux sources web accessibles publiquement",
            "Évaluation de crédibilité basée sur des critères automatisés", 
            "Synthèse générée par IA - vérification humaine recommandée"
        ]
        
        summaries_count = len(input_data.summarization_output.summaries)
        avg_credibility = input_data.summarization_output.average_credibility or 0.5
        
        quality_assessment = f"""
Qualité des données évaluée sur {summaries_count} sources analysées.
Score de crédibilité moyen: {avg_credibility:.2f}/1.0.
Sources diversifiées avec analyse automatisée de sentiment et biais.
"""
        
        return Methodology(
            research_approach="Recherche web automatisée avec synthèse par IA",
            sources_count=summaries_count,
            analysis_methods=analysis_methods,
            limitations=limitations,
            data_quality_assessment=quality_assessment.strip()
        )
    
    def _create_source_references(self, summaries: List[DocumentSummary]) -> List[SourceReference]:
        """Crée les références bibliographiques des sources."""
        
        references = []
        
        for summary in summaries:
            reference = SourceReference(
                title=summary.title,
                url=str(summary.url),
                author=getattr(summary, 'author', None),
                publication_date=getattr(summary, 'published_date', None),
                credibility_score=summary.credibility_score,
                citation_count=1  # Par défaut, chaque source est citée au moins une fois
            )
            references.append(reference)
        
        return references
    
    async def _assess_quality(
        self, 
        input_data: GlobalSynthesisInput, 
        sections: List[ReportSection]
    ) -> Dict[str, float]:
        """Évalue la qualité de l'analyse et du rapport."""
        
        summaries = input_data.summarization_output.summaries
        credibility_scores = [s.credibility_score for s in summaries if s.credibility_score]
        
        # Métriques de base
        completeness_score = min(len(summaries) / 5.0, 1.0)  # Optimal à 5+ sources
        
        if credibility_scores:
            reliability_score = sum(credibility_scores) / len(credibility_scores)
        else:
            reliability_score = 0.5
        
        coherence_score = min(len(sections) / 3.0, 1.0)  # Optimal à 3+ sections
        
        # Score de confiance global
        confidence_score = (completeness_score * 0.4 + 
                          reliability_score * 0.4 + 
                          coherence_score * 0.2)
        
        return {
            "confidence_score": confidence_score,
            "completeness_score": completeness_score,
            "reliability_score": reliability_score,
            "coherence_score": coherence_score
        }
    
    def _assemble_final_report(
        self,
        input_data: GlobalSynthesisInput,
        executive_summary: ExecutiveSummary,
        sections: List[ReportSection],
        methodology: Methodology,
        source_references: List[SourceReference],
        quality_scores: Dict[str, float]
    ) -> FinalReport:
        """Assemble le rapport final complet."""
        
        # Génération de l'ID du rapport
        report_id = self._generate_report_id(input_data.original_topic)
        
        # Titre du rapport
        title = f"Analyse de Recherche: {input_data.original_topic.title()}"
        
        # Introduction générique
        introduction = f"""
            Ce rapport présente une analyse complète du sujet "{input_data.original_topic}" 
            basée sur l'analyse de {len(source_references)} sources documentaires.

            L'analyse a été réalisée par un système d'intelligence artificielle utilisant des 
            méthodes d'extraction automatique de contenu, de résumé intelligent et de synthèse 
            thématique transversale.
        """.strip()
        
        # Conclusion générique
        conclusion = f"""
            Cette analyse de "{input_data.original_topic}" révèle des insights importants 
            basés sur {len(source_references)} sources analysées. 

            Les résultats présentés dans ce rapport offrent une perspective complète sur 
            les différents aspects du sujet, avec un score de confiance global de 
            {quality_scores['confidence_score']:.2f}/1.0.

            Pour des décisions importantes, il est recommandé de compléter cette analyse 
            par une vérification humaine et des sources supplémentaires si nécessaire.
        """.strip()
        
        # Calcul du nombre de mots (approximatif)
        word_count = (
            len(introduction.split()) +
            len(conclusion.split()) +
            len(executive_summary.summary_text.split()) +
            sum(len(section.content.split()) for section in sections)
        )
        
        # Extraction des thèmes et tendances
        summarization_output = input_data.summarization_output
        
        return FinalReport(
            report_id=report_id,
            title=title,
            topic=input_data.original_topic,
            report_type=input_data.report_type,
            report_format=input_data.report_format,
            
            executive_summary=executive_summary,
            introduction=introduction,
            main_sections=sections,
            conclusion=conclusion,
            
            key_themes=summarization_output.common_themes[:10],
            consensus_points=summarization_output.consensus_points[:10],
            conflicting_viewpoints=summarization_output.conflicting_views[:10],
            emerging_trends=[],  # À améliorer avec analyse spécifique
            
            methodology=methodology,
            sources=source_references,
            
            confidence_score=quality_scores["confidence_score"],
            completeness_score=quality_scores["completeness_score"],
            
            total_sources_analyzed=len(source_references),
            processing_time=0.0,  # Sera mis à jour par le processus principal
            word_count=word_count
        )
    
    async def _generate_formatted_outputs(
        self, 
        final_report: FinalReport, 
        input_data: GlobalSynthesisInput
    ) -> Dict[str, str]:
        """Génère le rapport dans différents formats."""
        
        formatted_outputs = {}
        
        # Format Markdown (par défaut)
        markdown_content = self._format_as_markdown(final_report)
        formatted_outputs["markdown"] = markdown_content
        
        # Format texte simple
        text_content = self._format_as_text(final_report)
        formatted_outputs["text"] = text_content
        
        # Format HTML (basique)
        html_content = self._format_as_html(final_report)
        formatted_outputs["html"] = html_content
        
        return formatted_outputs
    
    def _format_as_markdown(self, report: FinalReport) -> str:
        """Formate le rapport en Markdown."""
        
        content = f"""# {report.title}

**Sujet:** {report.topic}  
**Date de génération:** {report.generated_at.strftime('%d/%m/%Y %H:%M')}  
**ID du rapport:** {report.report_id}

---

## Résumé Exécutif

{report.executive_summary.summary_text}

### Conclusions Principales
{chr(10).join(f"- {finding}" for finding in report.executive_summary.key_findings)}

### Insights Clés
{chr(10).join(f"- {insight}" for insight in report.executive_summary.main_insights)}

### Recommandations
{chr(10).join(f"- {rec}" for rec in report.executive_summary.recommendations)}

---

## Introduction

{report.introduction}

---

"""
        
        # Ajout des sections principales
        for section in report.main_sections:
            content += f"## {section.title}\n\n{section.content}\n\n---\n\n"
        
        # Thèmes et analyses
        if report.key_themes:
            content += "## Thèmes Principaux\n\n"
            content += "\n".join(f"- {theme}" for theme in report.key_themes[:5])
            content += "\n\n---\n\n"
        
        # Conclusion
        content += f"## Conclusion\n\n{report.conclusion}\n\n---\n\n"
        
        # Méthodologie
        content += f"""## Méthodologie

            **Approche:** {report.methodology.research_approach}  
            **Sources analysées:** {report.methodology.sources_count}  
            **Score de confiance:** {report.confidence_score:.2f}/1.0

            ### Méthodes d'Analyse
            {chr(10).join(f"- {method}" for method in report.methodology.analysis_methods)}

            ### Limitations
            {chr(10).join(f"- {limitation}" for limitation in report.methodology.limitations)}

            ---

            ## Sources

        """
        
        # Sources
        for i, source in enumerate(report.sources, 1):
            content += f"{i}. **{source.title}**  \n"
            content += f"   URL: {source.url}  \n"
            if source.credibility_score:
                content += f"   Crédibilité: {source.credibility_score:.2f}/1.0  \n"
            content += "\n"
        
        return content
    
    def _format_as_text(self, report: FinalReport) -> str:
        """Formate le rapport en texte simple."""
        content = f"""
            {report.title}
            {'=' * len(report.title)}

            Sujet: {report.topic}
            Date: {report.generated_at.strftime('%d/%m/%Y %H:%M')}
            ID: {report.report_id}

            RÉSUMÉ EXÉCUTIF
            {'-' * 20}

            {report.executive_summary.summary_text}

            CONCLUSIONS PRINCIPALES:
            {chr(10).join(f"• {finding}" for finding in report.executive_summary.key_findings)}

            INTRODUCTION
            {'-' * 15}

            {report.introduction}

        """
        
        # Sections principales
        for section in report.main_sections:
            content += f"\n{section.title.upper()}\n"
            content += "-" * len(section.title) + "\n\n"
            content += section.content + "\n\n"
        
        # Conclusion
        content += f"CONCLUSION\n{'-' * 10}\n\n{report.conclusion}\n\n"
        
        return content
    
    def _format_as_html(self, report: FinalReport) -> str:
        """Formate le rapport en HTML basique."""
        
        html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report.title}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #333; border-bottom: 2px solid #333; }}
                    h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
                    .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    ul {{ margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h1>{report.title}</h1>
                
                <div class="metadata">
                    <strong>Sujet:</strong> {report.topic}<br>
                    <strong>Date:</strong> {report.generated_at.strftime('%d/%m/%Y %H:%M')}<br>
                    <strong>Score de confiance:</strong> {report.confidence_score:.2f}/1.0
                </div>
                
                <h2>Résumé Exécutif</h2>
                <p>{report.executive_summary.summary_text}</p>
                
                <h3>Conclusions Principales</h3>
                <ul>
                    {''.join(f"<li>{finding}</li>" for finding in report.executive_summary.key_findings)}
                </ul>
                
                <h2>Introduction</h2>
                <p>{report.introduction}</p>
        """
        
        # Sections principales
        for section in report.main_sections:
            html += f"""
                <h2>{section.title}</h2>
                <div class="section">
                    {section.content.replace(chr(10), '<br>')}
                </div>
            """
        
        # Conclusion
        html += f"""
            <h2>Conclusion</h2>
            <p>{report.conclusion}</p>
            
            <h2>Sources</h2>
            <ol>
        """
        
        for source in report.sources:
            html += f"""
                    <li>
                        <strong>{source.title}</strong><br>
                        <a href="{source.url}">{source.url}</a>
                        {f"<br>Crédibilité: {source.credibility_score:.2f}/1.0" if source.credibility_score else ""}
                    </li>
            """
        
        html += """
            </ol>
        </body>
        </html>
        """
        
        return html
    
    # Méthodes utilitaires
    
    async def _get_llm_response(self, prompt: str, task_type: str) -> tuple:
        """Obtient une réponse LLM pour une tâche spécifique."""
        try:
            response = await self.llm_manager.get_completion(
                prompt,
                system_prompt=SYSTEM_PROMPTS['global_synthesizer'],
                temperature=0.3,
                max_tokens=3000
            )
            return task_type, response
        except Exception as e:
            self.logger.error(f"Erreur LLM pour {task_type}: {e}")
            return task_type, f"Erreur: {str(e)}"
    
    def _parse_main_synthesis(self, content: str) -> List[ReportSection]:
        """Parse la synthèse principale en sections structurées."""
        
        sections = []
        
        # Recherche des sections avec titres
        section_pattern = r'##\s+(.+?)\n(.*?)(?=##|\Z)'
        matches = re.findall(section_pattern, content, re.DOTALL)
        
        for i, (title, section_content) in enumerate(matches):
            section = ReportSection(
                title=title.strip(),
                content=section_content.strip(),
                order=i + 1
            )
            sections.append(section)
        
        # Si aucune section trouvée, créer une section générale
        if not sections:
            sections.append(ReportSection(
                title="Analyse Générale",
                content=content,
                order=1
            ))
        
        return sections
    
    def _parse_executive_summary(self, content: str) -> ExecutiveSummary:
        """Parse le contenu du résumé exécutif."""
        
        # Extraction simplifiée - à améliorer selon le format LLM
        lines = content.split('\n')
        
        key_findings = []
        main_insights = []
        recommendations = []
        summary_text = content
        
        # Recherche des sections spécifiques
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if 'conclusion' in line.lower() or 'finding' in line.lower():
                current_section = 'findings'
            elif 'insight' in line.lower() or 'découverte' in line.lower():
                current_section = 'insights'
            elif 'recommandation' in line.lower() or 'recommendation' in line.lower():
                current_section = 'recommendations'
            elif line.startswith('-') or line.startswith('•'):
                point = line[1:].strip()
                if len(point) > 10:
                    if current_section == 'findings':
                        key_findings.append(point)
                    elif current_section == 'insights':
                        main_insights.append(point)
                    elif current_section == 'recommendations':
                        recommendations.append(point)
        
        # Fallback: extraire les premières phrases comme findings
        if not key_findings:
            sentences = content.split('.')[:3]
            key_findings = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
        
        return ExecutiveSummary(
            key_findings=key_findings[:5],
            main_insights=main_insights[:5],
            recommendations=recommendations[:5],
            summary_text=summary_text[:500] + "..." if len(summary_text) > 500 else summary_text
        )
    
    def _create_fallback_executive_summary(self, prepared_data: Dict[str, Any]) -> ExecutiveSummary:
        """Crée un résumé exécutif de fallback."""
        
        return ExecutiveSummary(
            key_findings=[
                f"Analyse basée sur {prepared_data['summaries_count']} sources documentaires",
                f"Score de crédibilité moyen: {prepared_data.get('average_credibility', 0.5):.2f}/1.0",
                "Synthèse générée automatiquement par IA"
            ],
            main_insights=[
                "Analyse transversale des différentes perspectives sur le sujet",
                "Identification des thèmes récurrents et des consensus",
                "Évaluation critique des sources et de leur fiabilité"
            ],
            recommendations=[
                "Vérification humaine recommandée pour les décisions importantes",
                "Complément par des sources supplémentaires si nécessaire",
                "Mise à jour régulière de l'analyse"
            ],
            summary_text=f"Cette analyse du sujet '{prepared_data['topic']}' synthétise {prepared_data['summaries_count']} sources documentaires pour fournir une vue d'ensemble complète et objective."
        )
    
    def _generate_report_id(self, topic: str) -> str:
        """Génère un ID unique pour le rapport."""
        
        # Hash du sujet + timestamp
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        return f"rpt_{timestamp}_{topic_hash}"
    
    # #fonction global_summary from summarization output
    async def process_from_summarization_output(self, summarization_output: SummarizationOutput) -> GlobalSynthesisOutput:
        """Génère une synthèse globale à partir de la sortie du summarizer."""
        # Utilise le topic du fichier si non fourni
        topic_val =  (getattr(summarization_output, "topic", None) or "Sujet de synthèse")
        synthesis_input = GlobalSynthesisInput(
            summarization_output=summarization_output,
            original_topic=topic_val
        )
        if not self.validate_input(synthesis_input):
            self.logger.error("Entrée de synthèse invalide. Abandon.")
            raise ValueError("Invalid synthesis input")
        
        self.logger.info(f"Démarrage de la synthèse globale pour '{synthesis_input.original_topic}'...")
        output = await self.process(synthesis_input)
        return output
     


# Exemple d'utilisation
if __name__ == "__main__":
    import asyncio
    from src.models.document_models import Document, DocumentSummary, SummarizationOutput, KeyPoint
    
    import argparse
    import json
    import os
    import sys
    from pathlib import Path

    logger = setup_logger("global_synthesizer_cli")

    def load_summarization_output(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SummarizationOutput(**data)

    async def run_synthesis(input_json, topic=None, output_json=None):
        summarization_output = load_summarization_output(input_json)
        
        agent = GlobalSynthesizerAgent()
        output = await agent.process_from_summarization_output(summarization_output)
        # Génération du nom de fichier si non fourni
        if not output_json:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_json = f"synthesis_output_{ts}.json"
        # Correction : model_dump_json n'accepte pas ensure_ascii
        with open(output_json, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))
        logger.info(f"Synthèse sauvegardée dans {output_json}")
        print(f"\nSynthèse globale terminée. Rapport sauvegardé dans: {output_json}")

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Global Synthesizer Agent CLI")
        parser.add_argument("--input", required=True, help="Chemin du fichier JSON de sortie du summarizer")
        parser.add_argument("--topic", required=False, help="Sujet de recherche (optionnel)")
        parser.add_argument("--output", required=False, help="Chemin du fichier de sortie JSON (optionnel)")
        args = parser.parse_args()
        asyncio.run(run_synthesis(args.input, args.topic, args.output))