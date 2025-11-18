"""
Configuration du projet AI Research Assistant.
Ce fichier contient les configurations par défaut qui peuvent être surchargées
par les variables d'environnement.
"""

from pydantic_settings import BaseSettings
from typing import Dict, Optional, List


class APIConfig(BaseSettings):
    """Configuration des clés API et des paramètres associés"""
    # LLM API (REQUIS)
    GROQ_API_KEY: str = ""
    
    # APIs de Recherche (Au moins une REQUISE)
    SERPER_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    BRAVE_API_KEY: str = ""
    
    # Configuration des modèles
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4000
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Limites de recherche
    MAX_SOURCES: int = 20
    MAX_SUMMARY_LENGTH: int = 500
    SEARCH_TIMEOUT: int = 30
    
    # Performance et sécurité
    API_RATE_LIMIT: int = 100
    MAX_CONCURRENT_REQUESTS: int = 10
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class DatabaseConfig(BaseSettings):
    """Configuration de la base de données"""
    DATABASE_URL: str = "sqlite:///data/research.db"
    CHROMA_PERSIST_DIRECTORY: str = "data/chroma"
    CHROMA_COLLECTION_NAME: str = "research_documents"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class PathConfig(BaseSettings):
    """Configuration des chemins et répertoires"""
    DATA_DIR: str = "data"
    REPORTS_DIR: str = "data/reports"
    CACHE_DIR: str = "data/cache"
    LOGS_DIR: str = "logs"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class FeatureConfig(BaseSettings):
    """Configuration des fonctionnalités"""
    ENABLE_CACHING: bool = True
    ENABLE_VECTOR_STORE: bool = True
    ENABLE_RATE_LIMITING: bool = True
    CACHE_TTL: int = 3600
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class LoggingConfig(BaseSettings):
    """Configuration du logging"""
    LOG_LEVEL: str = "INFO"
    ENABLE_FILE_LOGGING: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class ExportConfig(BaseSettings):
    """Configuration d'export et rapports"""
    DEFAULT_EXPORT_FORMAT: str = "markdown"
    PDF_PAGE_SIZE: str = "A4"
    INCLUDE_CITATIONS: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class DevelopmentConfig(BaseSettings):
    """Configuration de développement"""
    DEBUG: bool = False
    DEVELOPMENT_MODE: bool = False
    WORKER_THREADS: int = 4
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Instanciation des configurations
try:
    api_config = APIConfig()
    database_config = DatabaseConfig()
    path_config = PathConfig()
    feature_config = FeatureConfig()
    logging_config = LoggingConfig()
    export_config = ExportConfig()
    development_config = DevelopmentConfig()
except Exception as e:
    print(f"Erreur lors du chargement de la configuration: {e}")
    # Configuration par défaut en cas d'erreur
    api_config = None