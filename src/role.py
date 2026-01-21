from loguru import logger
from typing import Any, Dict, Optional
from pathlib import Path

from prompts.medical_prompts import speech_prompt as medical_speech_prompt
from prompts.court_prompts import speech_prompt as court_speech_prompt
from prompts.persona_prompts import speech_prompt as persona_speech_prompt
from ollama import Ollama
from base import IParticipant
from utils import load_config
from belief import IBelief
from base import ScenarioType


config = load_config(str(Path(__file__).parent / "config.yaml"))

# define generics of belief class
# T = TypeVar("T", bound=IBelief)


class LLMParticipant(IParticipant):
    """Role of LLM participant"""

    def __init__(
        self,
        role: str,
        profile: str,
        llm_chat: Ollama,
        beliefs: Optional[dict[str, IBelief]] = None,
    ) -> None:
        self._role: str = role
        self._profile: str = profile
        self._interact: Ollama = llm_chat
        self._strategy_history: list[str] = []
        self._beliefs: Optional[dict[str, IBelief]] = beliefs

    @property
    def role(self) -> str:
        return self._role

    @property
    def belief(self) -> dict[str, IBelief]:
        if self._beliefs is None:
            return {}
        return self._beliefs

    def update_belief(self, new_belief: dict[str, dict[str, Any]]) -> None:
        if self._beliefs is not None:
            for role, belief in self._beliefs.items():
                if role in new_belief:
                    belief.update_belief(new_belief[role])

    async def generate_response(
        self,
        scenario: ScenarioType,
        context: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Generate LLM response"""
        try:
            # Build prompt
            prompt: str = self._build_prompt(scenario, context)
            history.append({"role": "user", "content": prompt})

            # logger.debug(f"Interaction Prompt: {prompt}")

            response, consumption = await self._interact._chat(messages=history)

            return response, consumption
        except Exception as e:
            logger.error(f"Error: generation error: {e}")
            return "Error in generation."

    def get_current_strategy(self) -> str:
        """get current strategy"""
        return self._strategy_history[-1]

    def _build_prompt(self, scenario: ScenarioType, context: Dict[str, Any]) -> str:
        """Set generate speech prompt"""

        if scenario == ScenarioType.COURT_DEBATE:
            prompt_func = court_speech_prompt
        elif scenario == ScenarioType.MEDICAL_QA:
            prompt_func = medical_speech_prompt
        elif scenario == ScenarioType.PERSONA_CHAT:
            prompt_func = persona_speech_prompt
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")

        return prompt_func(
            scenario=scenario,
            role=self.role,
            belief=self.belief if self.belief else None,
            profile=self._profile,
            context=context,
        )
