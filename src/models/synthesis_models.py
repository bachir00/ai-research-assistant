"""
Modèles Pydantic pour l'agent Global Synthesizer.
Définit les structures de données pour la synthèse finale et le rapport global.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from src.models.document_models import DocumentSummary, SummarizationOutput


class ReportType(str, Enum):
    """Types de rapports de synthèse finale."""
    EXECUTIVE = "executive"  # Rapport exécutif court
    DETAILED = "detailed"   # Rapport détaillé complet
    ACADEMIC = "academic"   # Rapport de style académique
    BUSINESS = "business"   # Rapport orienté business


class ReportFormat(str, Enum):
    """Formats de sortie du rapport."""
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


class GlobalSynthesisInput(BaseModel):
    """
    Input pour l'agent Global Synthesizer.
    """
    summarization_output: SummarizationOutput = Field(
        ..., 
        description="Sortie complète de l'agent Summarizer avec tous les résumés"
    )
    original_topic: str = Field(
        ..., 
        description="Sujet de recherche original"
    )
    synthesis_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Options de configuration pour la synthèse"
    )
    
    # Options configurables avec valeurs par défaut
    report_type: ReportType = Field(
        default=ReportType.DETAILED,
        description="Type de rapport à générer"
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.MARKDOWN,
        description="Format de sortie du rapport"
    )
    include_methodology: bool = Field(
        default=True,
        description="Inclure la section méthodologie"
    )
    include_sources: bool = Field(
        default=True,
        description="Inclure les références des sources"
    )
    include_limitations: bool = Field(
        default=True,
        description="Inclure les limitations de l'analyse"
    )
    max_report_length: int = Field(
        default=5000,
        description="Longueur maximale du rapport en mots"
    )
    target_audience: str = Field(
        default="general",
        description="Audience cible (general, business, academic, policy_makers)"
    )
    
    def __init__(self, **data):
        # Extraire les options de synthesis_options si présentes
        synthesis_options = data.get('synthesis_options', {})
        
        # Appliquer les options aux champs correspondants
        if 'report_type' in synthesis_options:
            data['report_type'] = synthesis_options['report_type']
        if 'report_format' in synthesis_options:
            data['report_format'] = synthesis_options['report_format']
        if 'include_methodology' in synthesis_options:
            data['include_methodology'] = synthesis_options['include_methodology']
        if 'include_sources' in synthesis_options:
            data['include_sources'] = synthesis_options['include_sources']
        if 'include_limitations' in synthesis_options:
            data['include_limitations'] = synthesis_options['include_limitations']
        if 'max_report_length' in synthesis_options:
            data['max_report_length'] = synthesis_options['max_report_length']
        if 'target_audience' in synthesis_options:
            data['target_audience'] = synthesis_options['target_audience']
        
        super().__init__(**data)
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_topic": "impact de l'intelligence artificielle sur l'emploi",
                "synthesis_options": {
                    "report_type": "detailed",
                    "report_format": "markdown",
                    "include_methodology": True,
                    "include_sources": True,
                    "target_audience": "business"
                }
            }
        }


class ExecutiveSummary(BaseModel):
    """Résumé exécutif du rapport final."""
    
    key_findings: List[str] = Field(
        default_factory=list,
        description="3-5 conclusions principales"
    )
    main_insights: List[str] = Field(
        default_factory=list,
        description="Insights et découvertes principales"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommandations basées sur l'analyse"
    )
    summary_text: str = Field(
        ...,
        description="Texte de synthèse exécutive (2-3 paragraphes)"
    )


class ReportSection(BaseModel):
    """Section individuelle du rapport."""
    
    title: str = Field(..., description="Titre de la section")
    content: str = Field(..., description="Contenu de la section")
    subsections: List['ReportSection'] = Field(
        default_factory=list,
        description="Sous-sections"
    )
    order: int = Field(default=0, description="Ordre d'affichage")


class SourceReference(BaseModel):
    """Référence bibliographique d'une source."""
    
    title: str = Field(..., description="Titre du document source")
    url: str = Field(..., description="URL du document")
    author: Optional[str] = Field(default=None, description="Auteur")
    publication_date: Optional[datetime] = Field(default=None, description="Date de publication")
    credibility_score: Optional[float] = Field(default=None, description="Score de crédibilité")
    citation_count: int = Field(default=0, description="Nombre de fois citée dans le rapport")


class Methodology(BaseModel):
    """Description de la méthodologie utilisée."""
    
    research_approach: str = Field(..., description="Approche de recherche utilisée")
    sources_count: int = Field(..., description="Nombre de sources analysées")
    analysis_methods: List[str] = Field(
        default_factory=list,
        description="Méthodes d'analyse utilisées"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Limitations de l'étude"
    )
    data_quality_assessment: str = Field(
        ...,
        description="Évaluation de la qualité des données"
    )


class FinalReport(BaseModel):
    """
    Modèle pour le rapport final de synthèse globale.
    """
    
    # Métadonnées du rapport
    report_id: str = Field(..., description="Identifiant unique du rapport")
    title: str = Field(..., description="Titre du rapport")
    topic: str = Field(..., description="Sujet de recherche original")
    generated_at: datetime = Field(default_factory=datetime.now, description="Date de génération")
    report_type: ReportType = Field(default=ReportType.DETAILED, description="Type de rapport")
    report_format: ReportFormat = Field(default=ReportFormat.MARKDOWN, description="Format du rapport")
    
    # Contenu principal
    executive_summary: ExecutiveSummary = Field(..., description="Résumé exécutif")
    introduction: str = Field(..., description="Introduction du rapport")
    main_sections: List[ReportSection] = Field(
        default_factory=list,
        description="Sections principales du rapport"
    )
    conclusion: str = Field(..., description="Conclusion du rapport")
    
    # Analyses transversales
    key_themes: List[str] = Field(
        default_factory=list,
        description="Thèmes principaux identifiés"
    )
    consensus_points: List[str] = Field(
        default_factory=list,
        description="Points de consensus entre les sources"
    )
    conflicting_viewpoints: List[str] = Field(
        default_factory=list,
        description="Points de vue contradictoires"
    )
    emerging_trends: List[str] = Field(
        default_factory=list,
        description="Tendances émergentes identifiées"
    )
    
    # Métadonnées d'analyse
    methodology: Methodology = Field(..., description="Méthodologie utilisée")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Sources utilisées avec références"
    )
    
    # Métriques de qualité
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score de confiance global (0-1)"
    )
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score de complétude de l'analyse (0-1)"
    )
    
    # Statistiques de traitement
    total_sources_analyzed: int = Field(default=0, description="Nombre total de sources analysées")
    processing_time: float = Field(default=0.0, description="Temps de traitement en secondes")
    word_count: int = Field(default=0, description="Nombre de mots du rapport")
    
    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "rpt_20241115_001",
                "title": "Impact de l'Intelligence Artificielle sur l'Emploi - Rapport de Synthèse",
                "topic": "impact de l'intelligence artificielle sur l'emploi",
                "report_type": "detailed",
                "executive_summary": {
                    "key_findings": [
                        "L'IA transformera 60% des emplois d'ici 2030",
                        "Nouveaux emplois créés dans la tech et supervision IA"
                    ],
                    "summary_text": "Analyse complète de l'impact de l'IA..."
                },
                "confidence_score": 0.85,
                "total_sources_analyzed": 5
            }
        }


class GlobalSynthesisOutput(BaseModel):
    """
    Modèle pour l'output de l'agent Global Synthesizer.
    """
    
    final_report: FinalReport = Field(..., description="Rapport final de synthèse")
    synthesis_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées sur le processus de synthèse"
    )
    processing_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistiques de traitement"
    )
    
    # Formats alternatifs du rapport
    formatted_outputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Rapport formaté dans différents formats (markdown, html, etc.)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Horodatage de la synthèse"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "synthesis_metadata": {
                    "llm_model_used": "groq/llama-3.1-8b-instant",
                    "synthesis_strategy": "comprehensive",
                    "quality_checks_passed": True
                },
                "processing_stats": {
                    "input_summaries": 5,
                    "synthesis_time": 15.3,
                    "final_report_words": 2500
                }
            }
        }


# Configuration forward reference pour les modèles imbriqués
ReportSection.model_rebuild()