"""
Classe de base pour tous les agents du système.
Définit l'interface commune et les fonctionnalités partagées.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic
from datetime import datetime
import asyncio
import uuid

from src.core.logging import setup_logger
from src.models.state_models import AgentState, AgentStatus, AgentType


# Type générique pour les inputs et outputs des agents
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class BaseAgent(ABC, Generic[InputType, OutputType]):
    """
    Classe de base abstraite pour tous les agents du système.
    
    Fournit les fonctionnalités communes :
    - Gestion de l'état
    - Logging
    - Gestion des erreurs et retry
    - Métriques de performance
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        name: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 300.0  # 5 minutes par défaut
    ):
        """
        Initialise l'agent de base.
        
        Args:
            agent_type: Type de l'agent
            name: Nom personnalisé de l'agent
            max_retries: Nombre maximum de tentatives en cas d'erreur
            timeout: Timeout en secondes pour l'exécution
        """
        self.agent_type = agent_type
        self.name = name or f"{agent_type.value}_agent"
        self.agent_id = str(uuid.uuid4())
        
        # Configuration
        self.max_retries = max_retries
        self.timeout = timeout
        
        # État de l'agent
        self.state = AgentState(
            agent_type=agent_type,
            max_retries=max_retries
        )
        
        # Logger spécifique à l'agent
        self.logger = setup_logger(f"agent_{self.name}")
        
        # Métriques
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        
        self.logger.info(f"Agent {self.name} initialisé (ID: {self.agent_id})")
    
    @abstractmethod
    async def process(self, input_data: InputType) -> OutputType:
        """
        Méthode principale de traitement de l'agent.
        Doit être implémentée par chaque agent concret.
        
        Args:
            input_data: Données d'entrée spécifiques à l'agent
            
        Returns:
            Données de sortie spécifiques à l'agent
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: InputType) -> bool:
        """
        Valide les données d'entrée.
        
        Args:
            input_data: Données à valider
            
        Returns:
            True si les données sont valides
        """
        pass
    
    async def execute(self, input_data: InputType) -> OutputType:
        """
        Exécute l'agent avec gestion des erreurs et retry.
        
        Args:
            input_data: Données d'entrée
            
        Returns:
            Résultat de l'exécution
            
        Raises:
            Exception: Si l'exécution échoue après tous les retry
        """
        self.logger.info(f"Début d'exécution de l'agent {self.name}")
        self.state.start_execution()
        self.metrics["total_executions"] += 1
        
        # Validation des données d'entrée
        if not self.validate_input(input_data):
            error_msg = f"Données d'entrée invalides pour l'agent {self.name}"
            self.logger.error(error_msg)
            self.state.mark_error(error_msg)
            self.metrics["failed_executions"] += 1
            raise ValueError(error_msg)
        
        # Tentatives d'exécution avec retry
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(f"Tentative {attempt + 1}/{self.max_retries + 1}")
                
                # Exécution avec timeout
                result = await asyncio.wait_for(
                    self.process(input_data),
                    timeout=self.timeout
                )
                
                # Succès
                self.state.complete_execution()
                self.metrics["successful_executions"] += 1
                self._update_processing_time()
                
                self.logger.info(f"Agent {self.name} terminé avec succès")
                return result
                
            except asyncio.TimeoutError as e:
                error_msg = f"Timeout atteint pour l'agent {self.name} (>{self.timeout}s)"
                self.logger.warning(error_msg)
                last_exception = e
                self.state.retry_count += 1
                
            except Exception as e:
                error_msg = f"Erreur dans l'agent {self.name}: {str(e)}"
                self.logger.warning(error_msg)
                last_exception = e
                self.state.retry_count += 1
                
                # Attendre avant la prochaine tentative (backoff exponentiel)
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s, etc.
                    self.logger.info(f"Attente de {wait_time}s avant la prochaine tentative")
                    await asyncio.sleep(wait_time)
        
        # Toutes les tentatives ont échoué
        final_error = f"Agent {self.name} a échoué après {self.max_retries + 1} tentatives"
        self.logger.error(final_error)
        self.state.mark_error(final_error)
        self.metrics["failed_executions"] += 1
        
        raise Exception(final_error) from last_exception
    
    def _update_processing_time(self):
        """Met à jour les métriques de temps de traitement."""
        if self.state.duration:
            self.metrics["total_processing_time"] += self.state.duration
            self.metrics["average_processing_time"] = (
                self.metrics["total_processing_time"] / 
                self.metrics["successful_executions"]
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retourne le statut actuel de l'agent.
        
        Returns:
            Dictionnaire avec les informations de statut
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type.value,
            "status": self.state.status.value,
            "retry_count": self.state.retry_count,
            "duration": self.state.duration,
            "error_message": self.state.error_message,
            "metrics": self.metrics,
            "last_execution": self.state.end_time.isoformat() if self.state.end_time else None
        }
    
    def reset(self):
        """Remet l'agent à zéro pour une nouvelle exécution."""
        self.state = AgentState(
            agent_type=self.agent_type,
            max_retries=self.max_retries
        )
        self.logger.info(f"Agent {self.name} remis à zéro")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, status={self.state.status.value})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.agent_id}, "
                f"type={self.agent_type.value}, status={self.state.status.value})")


class AgentError(Exception):
    """Exception personnalisée pour les erreurs d'agents."""
    
    def __init__(self, message: str, agent_name: str, agent_id: str):
        self.agent_name = agent_name
        self.agent_id = agent_id
        super().__init__(f"Agent {agent_name} ({agent_id}): {message}")


class AgentTimeoutError(AgentError):
    """Exception pour les timeouts d'agents."""
    pass


class AgentValidationError(AgentError):
    """Exception pour les erreurs de validation d'agents."""
    pass