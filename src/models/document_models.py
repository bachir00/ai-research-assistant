"""
Modèles Pydantic pour l'agent Reader/Summarizer.
Définit les structures de données pour les documents et leurs résumés.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum


class DocumentType(str, Enum):
    """Types de documents supportés."""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    ACADEMIC_PAPER = "academic_paper"
    NEWS = "news"
    REPORT = "report"
    OTHER = "other"


class Document(BaseModel):
    """
    Modèle pour un document à analyser.
    """
    title: str = Field(..., description="Titre du document")
    url: HttpUrl = Field(..., description="URL source du document")
    content: str = Field(..., description="Contenu textuel complet du document")
    doc_type: DocumentType = Field(default=DocumentType.ARTICLE, description="Type de document")
    author: Optional[str] = Field(default=None, description="Auteur du document")
    published_date: Optional[datetime] = Field(default=None, description="Date de publication")
    source: Optional[str] = Field(default=None, description="Site ou publication source")
    word_count: int = Field(default=0, ge=0, description="Nombre de mots dans le document")
    language: str = Field(default="fr", description="Langue du document (code ISO)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "L'impact de l'IA sur le futur du travail",
                "url": "https://example.com/article-ia-travail",
                "content": "L'intelligence artificielle transforme rapidement...",
                "doc_type": "article",
                "author": "Marie Martin",
                "published_date": "2024-01-15T09:30:00Z",
                "source": "TechMag",
                "word_count": 1500,
                "language": "fr"
            }
        }


class ExtractionInput(BaseModel):
    """
    Input pour l'agent Content Extractor.
    """
    urls: List[str] = Field(..., description="Liste des URLs à extraire", min_items=1)
    content_filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Filtres à appliquer au contenu extrait"
    )
    extraction_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Options d'extraction spécifiques"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": [
                    "https://example.com/article1",
                    "https://example.com/article2.pdf"
                ],
                "content_filters": {
                    "min_content_length": 100,
                    "max_content_length": 10000,
                    "language": "fr",
                    "required_keywords": ["intelligence artificielle"]
                },
                "extraction_options": {
                    "timeout": 30,
                    "max_retries": 2
                }
            }
        }


class ExtractionResult(BaseModel):
    """
    Résultat de l'extraction de contenu.
    """
    documents: List[Document] = Field(..., description="Documents extraits avec succès")
    total_urls: int = Field(..., ge=0, description="Nombre total d'URLs traitées")
    successful_extractions: int = Field(..., ge=0, description="Nombre d'extractions réussies")
    failed_extractions: int = Field(..., ge=0, description="Nombre d'extractions échouées")
    failed_urls: List[str] = Field(default_factory=list, description="URLs qui ont échoué lors de l'extraction")
    execution_time: float = Field(..., ge=0, description="Temps d'exécution en secondes")
    extraction_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistiques détaillées de l'extraction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [],
                "total_urls": 5,
                "successful_extractions": 4,
                "failed_extractions": 1,
                "execution_time": 12.5,
                "extraction_stats": {
                    "total_words": 5000,
                    "average_words_per_doc": 1250,
                    "doc_types": {"article": 3, "pdf": 1},
                    "languages": {"fr": 4}
                }
            }
        }


class KeyPoint(BaseModel):
    """
    Modèle pour un point clé extrait d'un document.
    """
    title: str = Field(..., description="Titre du point clé")
    content: str = Field(..., description="Contenu détaillé du point")
    importance: float = Field(..., ge=0, le=1, description="Score d'importance (0-1)")
    category: Optional[str] = Field(default=None, description="Catégorie du point clé")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Automatisation des tâches répétitives",
                "content": "L'IA permet d'automatiser 30% des tâches actuelles...",
                "importance": 0.9,
                "category": "automatisation"
            }
        }


class Citation(BaseModel):
    """
    Modèle pour une citation importante extraite du document.
    """
    text: str = Field(..., description="Texte de la citation")
    author: Optional[str] = Field(default=None, description="Auteur de la citation")
    context: Optional[str] = Field(default=None, description="Contexte de la citation")
    page_number: Optional[int] = Field(default=None, description="Numéro de page (si applicable)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "L'IA ne remplacera pas les humains, elle augmentera leurs capacités",
                "author": "Dr. Jean Dupont",
                "context": "Conclusion de l'étude sur l'IA et l'emploi",
                "page_number": None
            }
        }


class DocumentSummary(BaseModel):
    """
    Modèle pour le résumé d'un document.
    """
    document_id: str = Field(..., description="Identifiant unique du document")
    title: str = Field(..., description="Titre du document original")
    url: HttpUrl = Field(..., description="URL du document original")
    
    # Résumé principal
    executive_summary: str = Field(..., description="Résumé exécutif (2-3 phrases)")
    detailed_summary: str = Field(..., description="Résumé détaillé (1-2 paragraphes)")
    
    # Points clés
    key_points: List[KeyPoint] = Field(default_factory=list, description="Points clés extraits")
    main_arguments: List[str] = Field(default_factory=list, description="Arguments principaux")
    
    # Citations et données
    important_citations: List[Citation] = Field(default_factory=list, description="Citations importantes")
    statistics: List[str] = Field(default_factory=list, description="Statistiques mentionnées")
    
    # Métadonnées d'analyse
    sentiment: Optional[str] = Field(default=None, description="Sentiment général (positif/neutre/négatif)")
    bias_assessment: Optional[str] = Field(default=None, description="Évaluation des biais potentiels")
    credibility_score: Optional[float] = Field(default=None, ge=0, le=1, description="Score de crédibilité (0-1)")
    
    # Informations de traitement
    processed_at: datetime = Field(default_factory=datetime.now, description="Horodatage du traitement")
    processing_time: float = Field(default=0.0, ge=0, description="Temps de traitement en secondes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "title": "L'impact de l'IA sur le futur du travail",
                "url": "https://example.com/article",
                "executive_summary": "L'IA transformera 60% des emplois d'ici 2030...",
                "detailed_summary": "Cette étude approfondie examine...",
                "key_points": [],
                "sentiment": "neutre",
                "credibility_score": 0.8,
                "processed_at": "2024-01-15T10:15:00Z",
                "processing_time": 5.2
            }
        }


class SummarizationOutput(BaseModel):
    """
    Modèle pour l'output complet de l'agent Reader/Summarizer.
    """
    summaries: List[DocumentSummary] = Field(..., description="Liste des résumés de documents")
    total_documents: int = Field(..., ge=0, description="Nombre total de documents traités")
    total_processing_time: float = Field(..., ge=0, description="Temps total de traitement")
    average_credibility: Optional[float] = Field(default=None, ge=0, le=1, description="Score de crédibilité moyen")
    
    # Analyse globale
    common_themes: List[str] = Field(default_factory=list, description="Thèmes récurrents identifiés")
    consensus_points: List[str] = Field(default_factory=list, description="Points de consensus entre les sources")
    conflicting_views: List[str] = Field(default_factory=list, description="Points de vue conflictuels")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Horodatage de l'analyse")
    
    class Config:
        json_schema_extra = {
            "example": {
                "summaries": [],
                "total_documents": 5,
                "total_processing_time": 25.6,
                "average_credibility": 0.75,
                "common_themes": ["automatisation", "formation", "adaptation"],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }