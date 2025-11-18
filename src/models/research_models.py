"""
Modèles Pydantic pour l'agent Researcher.
Définit les structures de données pour les requêtes de recherche et les résultats.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


#Passer par llm --> to Retreive keywords
class ResearchQuery(BaseModel):
    """
    Modèle pour une requête de recherche.
    """
    topic: str = Field(..., description="Le sujet de recherche principal")
    keywords: List[str] = Field(default_factory=list, description="Mots-clés spécifiques à rechercher")
    max_results: int = Field(default=5, ge=1, le=20, description="Nombre maximum de résultats à retourner")
    search_depth: str = Field(default="basic", description="Profondeur de la recherche: 'basic' ou 'advanced'")
    date_range: Optional[str] = Field(default=None, description="Période de recherche (ex: 'last_year', 'last_month')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "impact de l'intelligence artificielle sur l'emploi",
                "keywords": ["IA", "automatisation", "marché du travail"],
                "max_results": 5,
                "search_depth": "basic",
                "date_range": "last_year" # Faire l'intégration de year also in the research agent
            }
        }


class SearchResult(BaseModel):
    """
    Modèle pour un résultat de recherche individuel.
    """
    title: str = Field(..., description="Titre de l'article ou de la page")
    url: HttpUrl = Field(..., description="URL de la source")
    snippet: str = Field(..., description="Extrait ou résumé court du contenu")
    published_date: Optional[datetime] = Field(default=None, description="Date de publication")
    author: Optional[str] = Field(default=None, description="Auteur de l'article")
    source: Optional[str] = Field(default=None, description="Site source (ex: 'lemonde.fr')")
    score: Optional[float] = Field(default=None, ge=0, le=1, description="Score de pertinence (0-1)")
    tags: List[str] = Field(default_factory=list, description="Tags ou catégories associées")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "L'IA transforme le marché de l'emploi",
                "url": "https://example.com/article",
                "snippet": "Une étude récente montre que l'intelligence artificielle...",
                "published_date": "2024-01-15T10:00:00Z",
                "author": "Jean Dupont",
                "source": "example.com",
                "score": 0.85,
                "tags": ["technologie", "emploi"]
            }
        }


class ResearchOutput(BaseModel):
    """
    Modèle pour l'output complet de l'agent Researcher.
    """
    query: ResearchQuery = Field(..., description="La requête originale")
    results: List[SearchResult] = Field(..., description="Liste des résultats trouvés")
    total_found: int = Field(..., ge=0, description="Nombre total de résultats trouvés")
    search_time: float = Field(..., ge=0, description="Temps de recherche en secondes")
    search_engine: str = Field(..., description="Moteur de recherche utilisé (ex: 'tavily', 'serper')")
    timestamp: datetime = Field(default_factory=datetime.now, description="Horodatage de la recherche")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": {
                    "topic": "impact de l'IA sur l'emploi",
                    "max_results": 5
                },
                "results": [],
                "total_found": 15,
                "search_time": 2.3,
                "search_engine": "tavily",
                "timestamp": "2024-01-15T10:00:00Z"
            }
        }