"""
Modèles de données Pydantic pour le système multi-agents.
"""

from .research_models import (
    ResearchQuery,
    SearchResult,
    ResearchOutput
)
from .document_models import (
    Document,
    DocumentSummary,
    SummarizationOutput,
    KeyPoint,
    Citation,
    DocumentType
)
from .report_models import (
    ReportSection,
    Report,
    ReportOutput,
    Reference,
    ReportFormat,
    ReportMetadata
)
from .state_models import (
    AgentState,
    GraphState,
    AgentType,
    AgentStatus,
    ProcessingStep,
    WorkflowEvent
)

__all__ = [
    # Research models
    "ResearchQuery",
    "SearchResult", 
    "ResearchOutput",
    
    # Document models
    "Document",
    "DocumentSummary",
    "SummarizationOutput",
    "KeyPoint",
    "Citation", 
    "DocumentType",
    
    # Report models
    "ReportSection",
    "Report",
    "ReportOutput",
    "Reference",
    "ReportFormat",
    "ReportMetadata",
    
    # State models
    "AgentState",
    "GraphState",
    "AgentType",
    "AgentStatus",
    "ProcessingStep",
    "WorkflowEvent"
]