from typing import Any, TYPE_CHECKING
from belief import IBelief
from base import ScenarioType

if TYPE_CHECKING:
    from role import LLMParticipant


def speech_prompt(
    scenario: ScenarioType,
    role: str,
    belief: dict[str, IBelief],
    profile: str,
    context: dict[str, Any],
) -> str:
    """
    prompt of generating interaction content for Medical Q&A
    """
    # Implementation details omitted for reproducibility
    return "Placeholder prompt"


def get_payoff_prompt(
    scenario: ScenarioType,
    participants: list["LLMParticipant"],
    context: dict[str, Any],
) -> str:
    # Implementation details omitted for reproducibility
    return "Placeholder prompt"


def get_action_predict_prompt(
    scenario: ScenarioType,
    participants: list["LLMParticipant"],
    context: dict[str, Any],
) -> str:
    # Implementation details omitted for reproducibility
    return "Placeholder prompt"


def get_evaluate_speech_prompt(
    scenario: ScenarioType, role: str, content: str, context: dict[str, Any]
) -> str:
    # Implementation details omitted for reproducibility
    return "Placeholder prompt"


def get_mad_prompts():
    """MAD prompts for medical QA debate"""
    # Implementation details omitted for reproducibility
    return {}
