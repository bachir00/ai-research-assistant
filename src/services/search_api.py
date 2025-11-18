"""
Services d'API pour la recherche web.
Intègre les APIs Tavily et Serper pour la recherche d'informations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests
import asyncio
import aiohttp
from datetime import datetime
import json

from src.core.logging import setup_logger
from src.models.research_models import SearchResult

# Import sécurisé de la configuration
try:
    from config.settings import api_config
except Exception as e:
    print(f"Erreur lors de l'import de la configuration: {e}")
    api_config = None


class SearchAPIError(Exception):
    """Exception pour les erreurs d'API de recherche."""
    pass


class BaseSearchAPI(ABC):
    """Interface de base pour les APIs de recherche."""
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Effectue une recherche.
        
        Args:
            query: Requête de recherche
            max_results: Nombre maximum de résultats
            **kwargs: Paramètres spécifiques à l'API
            
        Returns:
            Liste des résultats de recherche
        """
        pass


class TavilySearchAPI(BaseSearchAPI):
    """
    Client pour l'API Tavily.
    Documentation: https://docs.tavily.com/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Accès sécurisé à la configuration
        if api_config:
            self.api_key = api_key or getattr(api_config, 'TAVILY_API_KEY', '')
        else:
            self.api_key = api_key or ''
        self.base_url = "https://api.tavily.com"
        self.logger = setup_logger("tavily_api")
        
        if not self.api_key:
            raise SearchAPIError("Clé API Tavily manquante")
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        search_depth: str = "basic",
        include_images: bool = False,
        include_answer: bool = True,
        **kwargs
    ) -> List[SearchResult]:
        """
        Recherche avec l'API Tavily.
        
        Args:
            query: Requête de recherche
            max_results: Nombre de résultats (max 20)
            search_depth: "basic" ou "advanced"
            include_images: Inclure les images
            include_answer: Inclure une réponse IA
            
        Returns:
            Liste des résultats
        """
        self.logger.info(f"Recherche Tavily: '{query}' (max: {max_results})")
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": min(max_results, 20),
            "include_images": include_images,
            "include_answer": include_answer,
            "include_raw_content": False
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/search",
                    json=payload,
                    timeout=30
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise SearchAPIError(f"Erreur Tavily {response.status}: {error_text}")
                    
                    data = await response.json()
                    return self._parse_tavily_results(data)
                    
            except aiohttp.ClientTimeout:
                raise SearchAPIError("Timeout lors de la requête Tavily")
            except aiohttp.ClientError as e:
                raise SearchAPIError(f"Erreur de connexion Tavily: {str(e)}")
    
    def _parse_tavily_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse les résultats de l'API Tavily."""
        results = []
        
        for item in data.get("results", []):
            try:
                # Parsing de la date de publication si disponible
                published_date = None
                if "published_date" in item and item["published_date"]:
                    try:
                        published_date = datetime.fromisoformat(item["published_date"].replace('Z', '+00:00'))
                    except:
                        pass
                
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    published_date=published_date,
                    source=item.get("source", ""),
                    score=item.get("score", 0.0)
                )
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Erreur parsing résultat Tavily: {e}")
                continue
        
        self.logger.info(f"Tavily: {len(results)} résultats parsés")
        return results


class SerperSearchAPI(BaseSearchAPI):
    """
    Client pour l'API Serper (Google Search).
    Documentation: https://serper.dev/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Accès sécurisé à la configuration  
        if api_config:
            self.api_key = api_key or getattr(api_config, 'SERPER_API_KEY', '')
        else:
            self.api_key = api_key or ''
        self.base_url = "https://google.serper.dev"
        self.logger = setup_logger("serper_api")
        
        if not self.api_key:
            raise SearchAPIError("Clé API Serper manquante")
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        country: str = "fr",
        language: str = "fr",
        search_type: str = "search",
        **kwargs
    ) -> List[SearchResult]:
        """
        Recherche avec l'API Serper.
        
        Args:
            query: Requête de recherche
            max_results: Nombre de résultats (max 100)
            country: Code pays (ex: "fr", "us")
            language: Code langue (ex: "fr", "en")
            search_type: Type de recherche ("search", "news", "images")
            
        Returns:
            Liste des résultats
        """
        self.logger.info(f"Recherche Serper: '{query}' (max: {max_results})")
        
        payload = {
            "q": query,
            "num": min(max_results, 100),
            "gl": country,
            "hl": language
        }
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        endpoint = f"{self.base_url}/{search_type}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=30
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise SearchAPIError(f"Erreur Serper {response.status}: {error_text}")
                    
                    data = await response.json()
                    return self._parse_serper_results(data, search_type)
                    
            except aiohttp.ClientTimeout:
                raise SearchAPIError("Timeout lors de la requête Serper")
            except aiohttp.ClientError as e:
                raise SearchAPIError(f"Erreur de connexion Serper: {str(e)}")
    
    def _parse_serper_results(self, data: Dict[str, Any], search_type: str) -> List[SearchResult]:
        """Parse les résultats de l'API Serper."""
        results = []
        
        # Les résultats sont dans différentes clés selon le type de recherche
        items_key = "organic" if search_type == "search" else "news" if search_type == "news" else "images"
        items = data.get(items_key, [])
        
        for item in items:
            try:
                # Parsing de la date pour les news
                published_date = None
                if "date" in item:
                    try:
                        published_date = datetime.fromisoformat(item["date"])
                    except:
                        pass
                
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    published_date=published_date,
                    source=item.get("source", ""),
                    score=item.get("position", 0) / 100.0  # Position convertie en score
                )
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Erreur parsing résultat Serper: {e}")
                continue
        
        self.logger.info(f"Serper: {len(results)} résultats parsés")
        return results


class SearchAPIManager:
    """
    Gestionnaire des APIs de recherche.
    Permet de basculer entre les APIs et de gérer les fallbacks.
    """
    
    def __init__(self):
        self.apis = {}
        self.logger = setup_logger("search_manager")
        
        # Initialisation des APIs disponibles
        try:
            if api_config and getattr(api_config, 'TAVILY_API_KEY', ''):
                self.apis["tavily"] = TavilySearchAPI()
                self.logger.info("API Tavily initialisée")
        except Exception as e:
            self.logger.warning(f"Impossible d'initialiser Tavily: {e}")
        
        try:
            if api_config and getattr(api_config, 'SERPER_API_KEY', ''):
                self.apis["serper"] = SerperSearchAPI()
                self.logger.info("API Serper initialisée")
        except Exception as e:
            self.logger.warning(f"Impossible d'initialiser Serper: {e}")
        
        if not self.apis:
            raise SearchAPIError("Aucune API de recherche disponible")
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        preferred_api: str = "tavily",
        **kwargs
    ) -> List[SearchResult]:
        """
        Effectue une recherche avec fallback entre APIs.
        
        Args:
            query: Requête de recherche
            max_results: Nombre de résultats
            preferred_api: API préférée ("tavily" ou "serper")
            
        Returns:
            Liste des résultats
        """
        # Ordre de priorité des APIs
        api_order = [preferred_api] + [api for api in self.apis.keys() if api != preferred_api]
        
        for api_name in api_order:
            if api_name not in self.apis:
                continue
                
            try:
                self.logger.info(f"Tentative de recherche avec {api_name}")
                results = await self.apis[api_name].search(query, max_results, **kwargs)
                
                if results:
                    self.logger.info(f"Recherche réussie avec {api_name}: {len(results)} résultats")
                    return results
                else:
                    self.logger.warning(f"Aucun résultat avec {api_name}")
                    
            except Exception as e:
                self.logger.warning(f"Erreur avec {api_name}: {e}")
                continue
        
        # Aucune API n'a fonctionné
        raise SearchAPIError(f"Échec de recherche avec toutes les APIs pour: {query}")
    
    def get_available_apis(self) -> List[str]:
        """Retourne la liste des APIs disponibles."""
        return list(self.apis.keys())
    
    def is_api_available(self, api_name: str) -> bool:
        """Vérifie si une API est disponible."""
        return api_name in self.apis