from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


# Enum defining game states
class State(Enum):
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


# Enum defining scenario types
class ScenarioType(Enum):
    MEDICAL_QA = "Medical Q&A"
    COURT_DEBATE = "Court Debate"
    PERSONA_CHAT = "Persona Chat"


# Dataclass defining interaction message
@dataclass
class InteractionMessage:
    role: str
    content: str
    round: int
    strategy: str
    answer: Optional[str] = None


# Abstract base class: Participant interface
class IParticipant(ABC):
    """Abstract interface for participants, defining behaviors that all participants must implement"""

    @abstractmethod
    def generate_response(self, context: Dict[str, Any]) -> str:
        """Generate response content"""
        pass

    @abstractmethod
    def update_belief(self, new_belief: Any) -> None:
        """Update belief state"""
        pass

    @abstractmethod
    def get_current_strategy(self) -> str:
        """Get current strategy"""
        pass
