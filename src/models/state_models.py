"""
Modèles d'état pour l'orchestration LangGraph.
Définit l'état global du système et les états des agents.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from .research_models import ResearchQuery, ResearchOutput
from .document_models import SummarizationOutput
from .report_models import ReportOutput


class AgentType(str, Enum):
    """Types d'agents dans le système."""
    RESEARCHER = "researcher"
    CONTENT_EXTRACTOR = "content_extractor"
    READER = "reader"
    WRITER = "writer"


class AgentStatus(str, Enum):
    """Statuts possibles d'un agent."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class ProcessingStep(str, Enum):
    """Étapes du processus de recherche."""
    INIT = "init"
    RESEARCH = "research"
    READING = "reading" 
    WRITING = "writing"
    COMPLETED = "completed"
    ERROR = "error"


class AgentState(BaseModel):
    """
    État individuel d'un agent.
    """
    agent_type: AgentType = Field(..., description="Type de l'agent")
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Statut actuel")
    
    # Informations de timing
    start_time: Optional[datetime] = Field(default=None, description="Heure de début d'exécution")
    end_time: Optional[datetime] = Field(default=None, description="Heure de fin d'exécution")
    duration: Optional[float] = Field(default=None, description="Durée d'exécution en secondes")
    
    # Gestion des erreurs
    error_message: Optional[str] = Field(default=None, description="Message d'erreur si applicable")
    retry_count: int = Field(default=0, ge=0, description="Nombre de tentatives")
    max_retries: int = Field(default=3, ge=0, description="Nombre maximum de tentatives")
    
    # Métadonnées spécifiques à l'agent
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Données spécifiques à l'agent")
    
    def start_execution(self):
        """Marque le début de l'exécution."""
        self.status = AgentStatus.WORKING
        self.start_time = datetime.now()
        self.end_time = None
    
    def complete_execution(self):
        """Marque la fin réussie de l'exécution."""
        self.status = AgentStatus.COMPLETED
        self.end_time = datetime.now()
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
    
    def mark_error(self, error_message: str):
        """Marque l'agent en erreur."""
        self.status = AgentStatus.ERROR
        self.error_message = error_message
        self.end_time = datetime.now()
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_type": "researcher",
                "status": "completed",
                "start_time": "2024-01-15T10:00:00Z",
                "end_time": "2024-01-15T10:02:30Z",
                "duration": 150.0,
                "retry_count": 0,
                "metadata": {"search_engine": "tavily"}
            }
        }


class GraphState(BaseModel):
    """
    État global du graph LangGraph.
    Contient toutes les données partagées entre les agents.
    """
    # Identification de la session
    session_id: str = Field(..., description="Identifiant unique de la session")
    current_step: ProcessingStep = Field(default=ProcessingStep.INIT, description="Étape actuelle du processus")
    
    # Requête initiale
    original_query: Optional[ResearchQuery] = Field(default=None, description="Requête de recherche originale")
    
    # États des agents
    agents: Dict[AgentType, AgentState] = Field(
        default_factory=lambda: {
            AgentType.RESEARCHER: AgentState(agent_type=AgentType.RESEARCHER),
            AgentType.READER: AgentState(agent_type=AgentType.READER),
            AgentType.WRITER: AgentState(agent_type=AgentType.WRITER)
        },
        description="État de chaque agent"
    )
    
    # Données partagées entre agents
    research_output: Optional[ResearchOutput] = Field(default=None, description="Résultats de recherche")
    summarization_output: Optional[SummarizationOutput] = Field(default=None, description="Résultats de synthèse")
    report_output: Optional[ReportOutput] = Field(default=None, description="Rapport final")
    
    # Métadonnées globales
    start_time: datetime = Field(default_factory=datetime.now, description="Heure de début du processus")
    end_time: Optional[datetime] = Field(default=None, description="Heure de fin du processus")
    total_duration: Optional[float] = Field(default=None, description="Durée totale en secondes")
    
    # Configuration et paramètres
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration du processus")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="Préférences utilisateur")
    
    # Gestion des erreurs globales
    global_errors: List[str] = Field(default_factory=list, description="Erreurs globales du processus")
    is_successful: bool = Field(default=False, description="Indique si le processus s'est terminé avec succès")
    
    # Informations de débogage
    debug_info: Dict[str, Any] = Field(default_factory=dict, description="Informations de débogage")
    
    def get_current_agent(self) -> Optional[AgentType]:
        """Retourne l'agent actuellement en cours d'exécution."""
        for agent_type, agent_state in self.agents.items():
            if agent_state.status == AgentStatus.WORKING:
                return agent_type
        return None
    
    def is_agent_completed(self, agent_type: AgentType) -> bool:
        """Vérifie si un agent a terminé son exécution."""
        return self.agents[agent_type].status == AgentStatus.COMPLETED
    
    def all_agents_completed(self) -> bool:
        """Vérifie si tous les agents ont terminé."""
        return all(
            agent.status == AgentStatus.COMPLETED 
            for agent in self.agents.values()
        )
    
    def has_errors(self) -> bool:
        """Vérifie s'il y a des erreurs dans le processus."""
        return (
            len(self.global_errors) > 0 or
            any(agent.status == AgentStatus.ERROR for agent in self.agents.values())
        )
    
    def complete_process(self):
        """Marque le processus comme terminé."""
        self.end_time = datetime.now()
        self.total_duration = (self.end_time - self.start_time).total_seconds()
        self.current_step = ProcessingStep.COMPLETED
        self.is_successful = not self.has_errors()
    
    def add_global_error(self, error_message: str):
        """Ajoute une erreur globale."""
        self.global_errors.append(error_message)
        self.current_step = ProcessingStep.ERROR
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "current_step": "research",
                "original_query": {
                    "topic": "impact de l'IA sur l'emploi"
                },
                "start_time": "2024-01-15T10:00:00Z",
                "is_successful": False,
                "global_errors": []
            }
        }


class WorkflowEvent(BaseModel):
    """
    Événement dans le workflow LangGraph.
    """
    event_id: str = Field(..., description="Identifiant unique de l'événement")
    event_type: str = Field(..., description="Type d'événement")
    agent_type: Optional[AgentType] = Field(default=None, description="Agent concerné")
    timestamp: datetime = Field(default_factory=datetime.now, description="Horodatage de l'événement")
    data: Dict[str, Any] = Field(default_factory=dict, description="Données associées à l'événement")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_001",
                "event_type": "agent_started",
                "agent_type": "researcher",
                "timestamp": "2024-01-15T10:00:00Z",
                "data": {"query": "impact IA emploi"}
            }
        }