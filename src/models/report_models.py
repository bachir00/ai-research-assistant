"""
Modèles Pydantic pour l'agent Writer/Reporter.
Définit les structures de données pour la génération de rapports.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ReportFormat(str, Enum):
    """Formats de rapport supportés."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"


class SectionType(str, Enum):
    """Types de sections dans un rapport."""
    INTRODUCTION = "introduction"
    EXECUTIVE_SUMMARY = "executive_summary"
    MAIN_FINDINGS = "main_findings"
    DETAILED_ANALYSIS = "detailed_analysis"
    CONCLUSIONS = "conclusions"
    RECOMMENDATIONS = "recommendations"
    BIBLIOGRAPHY = "bibliography"
    APPENDIX = "appendix"


class Reference(BaseModel):
    """
    Modèle pour une référence bibliographique.
    """
    title: str = Field(..., description="Titre de la source")
    url: str = Field(..., description="URL de la source")
    author: Optional[str] = Field(default=None, description="Auteur de la source")
    published_date: Optional[datetime] = Field(default=None, description="Date de publication")
    source: Optional[str] = Field(default=None, description="Publication ou site source")
    accessed_date: datetime = Field(default_factory=datetime.now, description="Date d'accès")
    
    def to_citation(self, style: str = "apa") -> str:
        """
        Génère une citation formatée selon le style demandé.
        """
        if style.lower() == "apa":
            parts = []
            if self.author:
                parts.append(f"{self.author}")
            if self.published_date:
                parts.append(f"({self.published_date.year})")
            parts.append(f"{self.title}")
            if self.source:
                parts.append(f"{self.source}")
            parts.append(f"Récupéré de {self.url}")
            return ". ".join(parts) + "."
        return f"{self.title} - {self.url}"
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "L'IA et l'emploi : défis et opportunités",
                "url": "https://example.com/article",
                "author": "Dr. Marie Dubois",
                "published_date": "2024-01-10T00:00:00Z",
                "source": "Revue Technologique",
                "accessed_date": "2024-01-15T10:00:00Z"
            }
        }


class ReportSection(BaseModel):
    """
    Modèle pour une section de rapport.
    """
    title: str = Field(..., description="Titre de la section")
    content: str = Field(..., description="Contenu de la section en markdown")
    section_type: SectionType = Field(..., description="Type de section")
    subsections: List['ReportSection'] = Field(default_factory=list, description="Sous-sections")
    references: List[Reference] = Field(default_factory=list, description="Références citées dans cette section")
    order: int = Field(default=0, description="Ordre d'affichage de la section")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Introduction",
                "content": "L'intelligence artificielle transforme rapidement...",
                "section_type": "introduction",
                "subsections": [],
                "references": [],
                "order": 1
            }
        }


class ReportMetadata(BaseModel):
    """
    Métadonnées du rapport.
    """
    title: str = Field(..., description="Titre du rapport")
    subtitle: Optional[str] = Field(default=None, description="Sous-titre du rapport")
    author: str = Field(default="AI Research Assistant", description="Auteur du rapport")
    creation_date: datetime = Field(default_factory=datetime.now, description="Date de création")
    version: str = Field(default="1.0", description="Version du rapport")
    
    # Informations sur la recherche
    research_topic: str = Field(..., description="Sujet de recherche original")
    sources_count: int = Field(default=0, ge=0, description="Nombre de sources utilisées")
    
    # Tags et classification
    keywords: List[str] = Field(default_factory=list, description="Mots-clés du rapport")
    categories: List[str] = Field(default_factory=list, description="Catégories du rapport")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Impact de l'Intelligence Artificielle sur l'Emploi",
                "subtitle": "Analyse des tendances actuelles et perspectives d'avenir",
                "author": "AI Research Assistant",
                "research_topic": "impact de l'IA sur l'emploi",
                "sources_count": 8,
                "keywords": ["IA", "emploi", "automatisation"],
                "categories": ["technologie", "économie"]
            }
        }


class Report(BaseModel):
    """
    Modèle complet pour un rapport de recherche.
    """
    metadata: ReportMetadata = Field(..., description="Métadonnées du rapport")
    sections: List[ReportSection] = Field(..., description="Sections du rapport")
    bibliography: List[Reference] = Field(..., description="Bibliographie complète")
    
    # Configuration de formatage
    format_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration de formatage spécifique au format de sortie"
    )
    
    # Statistiques du rapport
    word_count: int = Field(default=0, ge=0, description="Nombre de mots total")
    reading_time_minutes: int = Field(default=0, ge=0, description="Temps de lecture estimé en minutes")
    
    def calculate_word_count(self) -> int:
        """Calcule le nombre de mots total du rapport."""
        total_words = 0
        for section in self.sections:
            total_words += len(section.content.split())
            # Récursif pour les sous-sections
            def count_subsection_words(subsections):
                words = 0
                for subsection in subsections:
                    words += len(subsection.content.split())
                    words += count_subsection_words(subsection.subsections)
                return words
            total_words += count_subsection_words(section.subsections)
        return total_words
    
    def calculate_reading_time(self, words_per_minute: int = 200) -> int:
        """Calcule le temps de lecture estimé."""
        if self.word_count == 0:
            self.word_count = self.calculate_word_count()
        return max(1, self.word_count // words_per_minute)
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "title": "Impact de l'IA sur l'Emploi",
                    "research_topic": "impact de l'IA sur l'emploi",
                    "sources_count": 5
                },
                "sections": [],
                "bibliography": [],
                "word_count": 2500,
                "reading_time_minutes": 12
            }
        }


class ReportOutput(BaseModel):
    """
    Modèle pour l'output de l'agent Writer/Reporter.
    """
    report: Report = Field(..., description="Le rapport généré")
    output_format: ReportFormat = Field(..., description="Format de sortie demandé")
    file_path: Optional[str] = Field(default=None, description="Chemin du fichier généré")
    
    # Informations de génération
    generation_time: float = Field(default=0.0, ge=0, description="Temps de génération en secondes")
    llm_calls: int = Field(default=0, ge=0, description="Nombre d'appels au LLM")
    
    # Qualité du rapport
    quality_score: Optional[float] = Field(default=None, ge=0, le=1, description="Score de qualité estimé")
    completeness_score: Optional[float] = Field(default=None, ge=0, le=1, description="Score de complétude")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Horodatage de la génération")
    
    class Config:
        json_schema_extra = {
            "example": {
                "report": {
                    "metadata": {
                        "title": "Impact de l'IA sur l'Emploi"
                    }
                },
                "output_format": "markdown",
                "file_path": "./output/rapport_ia_emploi.md",
                "generation_time": 15.3,
                "llm_calls": 3,
                "quality_score": 0.85,
                "timestamp": "2024-01-15T11:00:00Z"
            }
        }


# Mise à jour des références pour éviter les erreurs de forward reference
ReportSection.model_rebuild()