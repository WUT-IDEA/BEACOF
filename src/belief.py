from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from utils import load_config

config = load_config(str(Path(__file__).parent / "config.yaml"))


class IBelief(ABC):
    # Attributes that all subclasses must define
    _VALID_ATTRIBUTES: dict[str, float] = {}

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []
        # init all keys with default value
        self._belief_values: dict[str, float] = {}
        self._confidence_values: dict[str, float] = {}

        # Initialize default values for all attributes
        for attr, default_value in self._VALID_ATTRIBUTES.items():
            self._belief_values[attr] = default_value
            self._confidence_values[attr] = default_value

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history

    @property
    def belief(self) -> dict[str, float]:
        """get current belief"""
        return self._belief_values.copy()

    @property
    def confidence(self) -> dict[str, float]:
        """get current confidence"""
        return self._confidence_values.copy()

    @abstractmethod
    def update_belief(self, update_data: dict[str, Any]) -> None:
        """Update belief - to be implemented by subclasses"""
        pass

    def _update_belief_impl(self, update_data: dict[str, Any]) -> None:
        """
        Common implementation of belief update using Bayesian method
        """
        # Implementation details omitted for reproducibility
        pass

    def _validate_value(self, key: str, value: Any) -> float:
        """validate the effectiveness of the value"""
        if key not in self._VALID_ATTRIBUTES:
            raise ValueError(f"Invalid key: {key}")

        try:
            val: float = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"key of {key} must be a number")

        if not 0 <= val <= 1:
            raise ValueError(
                f"the range of key: {key} must be in range [0,1], current value: {val}"
            )

        return val


class MedQABelief(IBelief):
    # Allowed keys of belief for MedQA
    _VALID_ATTRIBUTES: dict[str, float] = {
        "evidence_strength": 0.5,  # Overall strength of opponent's evidence [0,1]
        "medical_knowledge": 0.5,  # Depth and accuracy of opponent's medical knowledge [0,1]
        "credibility": 0.5,  # Overall credibility of opponent [0,1]
        "diagnostic_accuracy": 0.5,  # Accuracy of opponent's diagnostic reasoning and conclusions [0,1]
        "consensus_prob": 0.5,  # Probability of reaching consensus based on this statement [0,1]
    }

    def update_belief(self, update_data: dict[str, Any]) -> None:
        """Update belief using common Bayesian implementation"""
        self._update_belief_impl(update_data)


class CourtDebateBelief(IBelief):
    # Allowed keys of belief for Court Debate
    _VALID_ATTRIBUTES: dict[str, float] = {
        "evidence_strength": 0.5,  # Strength of opponent's evidence [0,1]
        "legal_position": 0.5,  # Opponent's legal position [0,1]
        "credibility": 0.5,  # Credibility of opponent [0,1]
        "strategic_competence": 0.5,  # Strategic competence of opponent [0,1]
        "winning_prob": 0.5,  # Probability of opponent winning [0,1]
    }

    def update_belief(self, update_data: dict[str, Any]) -> None:
        """Update belief using common Bayesian implementation"""
        self._update_belief_impl(update_data)

    def reset_to_defaults(self) -> None:
        """reset the belief to default"""
        for attr, default_value in self._VALID_ATTRIBUTES.items():
            self._belief_values[attr] = default_value
            self._confidence_values[attr] = default_value


class PersonaChatBelief(IBelief):
    # Allowed keys of belief for Persona Chat
    _VALID_ATTRIBUTES: dict[str, float] = {
        "personality_consistency": 0.5,  # Consistency of opponent's personality [0,1]
        "communication_skill": 0.5,  # Communication skill of opponent [0,1]
        "emotional_intelligence": 0.5,  # Emotional intelligence of opponent [0,1]
        "engagement_level": 0.5,  # Engagement level of opponent [0,1]
        "relationship_building": 0.5,  # Relationship building ability of opponent [0,1]
    }

    def update_belief(self, update_data: dict[str, Any]) -> None:
        """Update belief using common Bayesian implementation"""
        self._update_belief_impl(update_data)

    def reset_to_defaults(self) -> None:
        """reset the belief to default"""
        for attr, default_value in self._VALID_ATTRIBUTES.items():
            self._belief_values[attr] = default_value
            self._confidence_values[attr] = default_value
