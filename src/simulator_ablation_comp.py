from loguru import logger
from typing import Any

from utils import extract_json_from_text, cal_consensus_distance, load_config
from belief import IBelief
from prompts.medical_prompts_ablation import get_payoff_prompt as medical_payoff
from prompts.court_prompts_ablation import get_payoff_prompt as court_payoff
from prompts.persona_prompts_ablation import get_payoff_prompt as persona_payoff
from prompts.medical_prompts_ablation import get_action_predict_prompt as medical_action
from prompts.court_prompts_ablation import get_action_predict_prompt as court_action
from prompts.persona_prompts_ablation import get_action_predict_prompt as persona_action
from prompts.medical_prompts_ablation import get_evaluate_speech_prompt as medical_eval
from prompts.court_prompts_ablation import get_evaluate_speech_prompt as court_eval
from prompts.persona_prompts_ablation import get_evaluate_speech_prompt as persona_eval
from base import State, InteractionMessage, ScenarioType
from ollama import Ollama
from role import LLMParticipant
from pathlib import Path
import json
import numpy as np

config = load_config(str(Path(__file__).parent / "config.yaml"))


class InteractiveGameEngine:
    """Interactive Simulation Engine - Ablation Experiment: Fixed Collaboration Type to Competition"""

    def __init__(
        self,
        participants: list[LLMParticipant],
        util: Ollama,
        scenario_type: ScenarioType,
        max_rounds: int = 10,
    ) -> None:
        self._scenario_type: ScenarioType = scenario_type
        self._max_rounds = max_rounds
        self._participants: list[LLMParticipant] = participants
        self._util: Ollama = util

        self._state = State.INITIALIZING
        self._current_round = 0

        self._interaction_history: list[InteractionMessage] = []
        self._payoff: list[dict[str, Any]] = []
        self._action_predict: list[dict[str, Any]] = []
        self._evaluation: list[dict[str, Any]] = []
        self._tokens: dict[str, list] = {}
        self._game_results: dict[str, Any] = {}
        self._belief_histories: dict[str, list[np.ndarray]] = {}

    def add_participant(self, participant: LLMParticipant) -> None:
        """add participant"""
        self._participants.append(participant)

    async def simulate(self, info: str, id: str, file_path: str) -> None:
        """Start Simulation"""
        try:
            if len(self._participants) < 2:
                raise ValueError("The number of participants must be greater than 2.")
            logger.info(f"{'Start Simulation':=^80}")

            self._info: str = info
            self._id: str = id
            self._state = State.IN_PROGRESS
            self._file_path: str = file_path
            # process of interaction
            await self._interaction_loop()

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            self._end_game(error=True)
            raise
        finally:
            self._reset_simulation_state()

    def _reset_simulation_state(self) -> None:
        """reset state for new simulation"""
        self._state = State.INITIALIZING
        self._current_round = 0
        self._interaction_history: list[InteractionMessage] = []
        self._payoff: list[dict[str, Any]] = []
        self._action_predict: list[dict[str, Any]] = []
        self._evaluation: list[dict[str, Any]] = []
        self._tokens: dict[str, list] = {}
        self._game_results: dict[str, Any] = {}
        self._belief_histories: dict[str, list[np.ndarray]] = {}

    def _get_payoff_prompt(self, scenario, participants, context):
        if scenario == ScenarioType.COURT_DEBATE:
            return court_payoff(scenario, participants, context)
        elif scenario == ScenarioType.MEDICAL_QA:
            return medical_payoff(scenario, participants, context)
        elif scenario == ScenarioType.PERSONA_CHAT:
            return persona_payoff(scenario, participants, context)
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")

    def _get_action_predict_prompt(self, scenario, participants, context):
        if scenario == ScenarioType.COURT_DEBATE:
            return court_action(scenario, participants, context)
        elif scenario == ScenarioType.MEDICAL_QA:
            return medical_action(scenario, participants, context)
        elif scenario == ScenarioType.PERSONA_CHAT:
            return persona_action(scenario, participants, context)
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")

    async def _interaction_loop(self) -> None:
        """Main interaction loop"""
        belief_state = []
        # Initialize belief history
        for p in self._participants:
            if hasattr(p, "_beliefs") and p._beliefs:
                # Each participant has only one belief object
                belief_obj = list(p._beliefs.values())[0]
                initial_vector = np.array(list(belief_obj.belief.values()))
                self._belief_histories[p.role] = [initial_vector]
        try:
            while (
                self._current_round < self._max_rounds
                and self._state == State.IN_PROGRESS
            ):

                # execute one round of interaction
                await self._execute_round()

                # collect belief state for a round
                for p in self._participants:
                    if hasattr(p, "_beliefs") and p._beliefs:
                        participant_beliefs = {}
                        for target_role, belief in p._beliefs.items():
                            if hasattr(belief, "belief"):
                                participant_beliefs[target_role] = belief.belief
                        if participant_beliefs:
                            belief_state.append({p.role: participant_beliefs})
                            # Append to belief history
                            belief_obj = list(p._beliefs.values())[0]
                            current_vector = np.array(list(belief_obj.belief.values()))
                            self._belief_histories[p.role].append(current_vector)

                self._current_round += 1

                # Check termination conditions
                if self.is_terminal_condition(
                    {
                        "round_count": self._current_round,
                        "max_rounds": self._max_rounds,
                        "interactions": self._interaction_history,
                    }
                ):
                    self._state = State.FINISHED

            self._end_game()

            result = {
                "id": self._id,
                "interaction_history": [
                    message.__dict__ for message in self._interaction_history
                ],
                "payoff": self._payoff,
                "predicted_actions": self._action_predict,
                "belief_state": belief_state,
                "evaluation": self._evaluation,
                "token": self._tokens,
            }

            self._save_result(self._file_path, result)

        except Exception as e:
            logger.error(f"Simulation execution error: {e}")
            self._state = State.ERROR
            raise e

    async def _execute_round(self) -> None:
        """Execute one round of interaction"""
        try:
            logger.info(f"{'='*20} Round {self._current_round + 1} {'='*20}")
            # generate payoff matrix of current round
            await self._generate_payoff()

            # generate strategy probabilities of each participant
            await self._predict_action()

            # interaction
            for participant in self._participants:
                try:
                    context: dict[str, Any] = {
                        "scenario_type": self._scenario_type.value,
                        "current_round": self._current_round + 1,
                        "max_rounds": self._max_rounds,
                        "payoff": (
                            self._payoff[-1][participant.role] if self._payoff else {}
                        ),
                        "action_predict": (
                            self._action_predict[-1] if self._action_predict else {}
                        ),
                        "info": self._info,
                        "history": self._interaction_history,
                    }

                    formatted_history = []
                    if len(self._interaction_history) > 0:
                        for i, message in enumerate(self._interaction_history):
                            # first role is user, second is assistant, etc.
                            role = "user" if i % 2 == 0 else "assistant"

                            # construct messages for chat history
                            formatted_history.append(
                                {"role": role, "content": message.content}
                            )

                    logger.info(
                        f"{'='*20} Round {self._current_round + 1} - {participant.role} Speaking {'='*20}"
                    )

                    # Ablation: Use forced strategy prompt
                    prompt = self._build_prompt_ablation(
                        scenario=self._scenario_type,
                        role=participant.role,
                        belief=(
                            participant.belief
                            if hasattr(participant, "belief")
                            else None
                        ),
                        profile=getattr(participant, "_profile", ""),
                        context=context,
                        forced_strategy="competition" if self._scenario_type == ScenarioType.COURT_DEBATE else "competition",
                    )
                    history = formatted_history + [{"role": "user", "content": prompt}]

                    response, consumption = await participant._interact._chat(
                        messages=history
                    )

                    if "interaction_token" not in self._tokens:
                        self._tokens["interaction_token"] = []
                    self._tokens["interaction_token"].append(
                        {"role": participant.role, "token": consumption["eval_count"]}
                    )

                    logger.info(
                        f"{participant.role + ' Speech Content':=^80}\n {response}"
                    )
                    speech: dict[str, Any] = extract_json_from_text(response)

                    # Validate speech structure
                    if self._scenario_type == ScenarioType.MEDICAL_QA:
                        required_keys = ["strategy", "answer", "content"]
                    else:
                        required_keys = ["strategy", "content"]
                    missing_keys = [key for key in required_keys if key not in speech]
                    if missing_keys:
                        raise ValueError(
                            f"Missing keys in speech JSON: {missing_keys}. Speech: {speech}"
                        )
                    if self._scenario_type == ScenarioType.MEDICAL_QA:
                        message: InteractionMessage = InteractionMessage(
                            role=participant.role,
                            content=speech["content"],
                            round=self._current_round + 1,
                            strategy=speech["strategy"],
                            answer=speech["answer"],
                        )
                    else:
                        message: InteractionMessage = InteractionMessage(
                            role=participant.role,
                            content=speech["content"],
                            round=self._current_round + 1,
                            strategy=speech["strategy"],
                        )

                    self._interaction_history.append(message)

                    # update others' belief
                    await self._update_participants_beliefs(new_message=message)

                except Exception as e:
                    logger.error(
                        f"Error in round {self._current_round + 1} for participant {participant.role}: {type(e).__name__}: {e}"
                    )
                    logger.error(
                        f"Response content (first 500 chars): {response[:500] if 'response' in locals() else 'Response not available'}"
                    )
                    logger.error(
                        f"Extracted speech: {speech if 'speech' in locals() else 'Speech not extracted'}"
                    )
                    raise

        except Exception as e:
            logger.error(f"Round {self._current_round + 1} Interaction error: {e}")
            raise

    def is_terminal_condition(self, conditions: dict) -> bool:
        # Belief early stopping mechanism: Terminate debate if belief change amplitude is below threshold for 3 consecutive rounds
        # Terminate as soon as one party meets the early stopping condition!
        for agent in self._participants:
            if agent.role in self._belief_histories and len(
                self._belief_histories[agent.role]
            ) >= (
            >= (config["belief"]["early_stop_round"] + 1)
            ):  # Initial + 3 rounds
                recent_changes = [
                    cal_consensus_distance(
                        self._belief_histories[agent.role][i],
                        self._belief_histories[agent.role][i - 1],
                    )
                    for i in range(-1, -(config["belief"]["early_stop_round"] + 1), -1)
                ]
                if all(
                    change < config["belief"]["belief_change_threshold"]
                    for change in recent_changes
                ):  # belief_change_threshold
                    logger.info(
                        f"{agent.role}'s belief has not changed significantly in the last 3 rounds, interaction ends"
                    )
                    return True
        return False

    async def _update_participants_beliefs(
        self, new_message: InteractionMessage
    ) -> None:
        """evaluate the speech and update belief"""
        logger.info(f"{'='*20} Evaluating Speech and Updating Beliefs {'='*20}")
        await self._evaluate_belief(new_message)
        others = [p for p in self._participants if p.role != new_message.role]
        for person in others:
            if hasattr(person, "_beliefs") and person._beliefs:
                speaker_role = new_message.role
            if speaker_role in person._beliefs:
                belief = person._beliefs[speaker_role]
                if hasattr(belief, "update_belief"):
                    belief.update_belief(self._evaluation[-1])
            # if person.belief is not None:
            #     person.update_belief(self._evaluation[-1])

    async def _generate_payoff(self) -> None:
        """generate payoff matrix of current round"""
        # if system_prompt is not None:
        #     self._util._set_system_prompt(system_prompt)
        logger.info(f"{'='*20} Generating Payoff Matrix {'='*20}")
        prompt: str = self._get_payoff_prompt(
            scenario=self._scenario_type,
            participants=self._participants,
            context={
                "current_round": self._current_round + 1,
                "max_rounds": self._max_rounds,
                "history": self._interaction_history,
                "info": self._info,
            },
        )

        # logger.debug(f"Payoff Prompt: {prompt}")

        response, consumption = await self._util._chat(
            [{"role": "user", "content": prompt}]
        )
        logger.info(f"Payoff LLM response:\n {response}")
        payoff: dict[str, Any] = extract_json_from_text(response)
        logger.info(f"Payoff extracted: {payoff}")
        # Map keys if necessary
        # if 'Participant A' in payoff and 'Participant B' in payoff:
        #     payoff['Doctor1'] = payoff.pop('Participant A')
        #     payoff['Doctor2'] = payoff.pop('Participant B')
        self._payoff.append(payoff)

        if "payoff_token" not in self._tokens:
            self._tokens["payoff_token"] = []
        self._tokens["payoff_token"].append(consumption["eval_count"])

    async def _predict_action(self):
        """predict all participants passible action in the next round."""
        logger.info(f"{'='*20} Predicting Actions {'='*20}")
        prompt: str = self._get_action_predict_prompt(
            scenario=self._scenario_type,
            participants=self._participants,
            context={
                "payoff": self._payoff[-1],
                "history": self._interaction_history,
                "info": self._info,
                "current_round": self._current_round + 1,
                "max_rounds": self._max_rounds,
            },
        )

        # logger.debug(f"Action Predict Prompt: {prompt}")

        response, consumption = await self._util._chat(
            [{"role": "user", "content": prompt}]
        )

        predict_action: dict[str, Any] = extract_json_from_text(response)
        logger.debug(f"Predicted Actions extracted: {predict_action}")
        self._action_predict.append(predict_action)

        if "action_predict_token" not in self._tokens:
            self._tokens["action_predict_token"] = []
        self._tokens["action_predict_token"].append(consumption["eval_count"])

    def _get_evaluate_speech_prompt(
        self, scenario: ScenarioType, role: str, content: str, context: dict[str, Any]
    ) -> str:
        """Get the appropriate evaluate speech prompt based on scenario"""
        if scenario == ScenarioType.MEDICAL_QA:
            return medical_eval(scenario, role, content, context)
        elif scenario == ScenarioType.COURT_DEBATE:
            return court_eval(scenario, role, content, context)
        elif scenario == ScenarioType.PERSONA_CHAT:
            return persona_eval(scenario, role, content, context)
        else:
            raise ValueError(f"Unsupported scenario type: {scenario}")

    def _build_prompt_ablation(
        self,
        scenario: ScenarioType,
        role: str,
        belief: dict[str, IBelief],
        profile: str,
        context: dict[str, Any],
        forced_strategy: str,
    ) -> str:
        """Build prompt with forced strategy for ablation experiments"""
        from prompts.medical_prompts_ablation import speech_prompt as medical_speech
        from prompts.court_prompts_ablation import speech_prompt as court_speech
        from prompts.persona_prompts_ablation import speech_prompt as persona_speech

        if scenario == ScenarioType.COURT_DEBATE:
            prompt_func = court_speech
        elif scenario == ScenarioType.MEDICAL_QA:
            prompt_func = medical_speech
        elif scenario == ScenarioType.PERSONA_CHAT:
            prompt_func = persona_speech
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")

        return prompt_func(
            scenario=scenario,
            role=role,
            belief=belief if belief else None,
            profile=profile,
            context=context,
            forced_strategy=forced_strategy,
        )

    async def _evaluate_belief(self, message: InteractionMessage) -> None:
        """evaluate the speech"""
        prompt: str = self._get_evaluate_speech_prompt(
            scenario=self._scenario_type,
            role=message.role,
            content=message.content,
            context={
                "current_round": self._current_round + 1,
                "max_rounds": self._max_rounds,
                "history": self._interaction_history,
                "info": self._info,
            },
        )

        # logger.debug(f"Evaluate Speech Prompt: {prompt}")

        response, consumption = await self._util._chat(
            [{"role": "user", "content": prompt}]
        )
        raw_eval: dict[str, Any] = extract_json_from_text(response)
        # Enrich with metadata while maintaining required keys 'belief' and 'confidence'
        if not isinstance(raw_eval, dict):
            raw_eval = {}
        evaluation: dict[str, Any] = {
            "speaker": message.role,
            "round": self._current_round + 1,
            **raw_eval,
        }
        self._evaluation.append(evaluation)
        if "belief_update_token" not in self._tokens:
            self._tokens["belief_update_token"] = []
        self._tokens["belief_update_token"].append(consumption["eval_count"])

    def _end_game(self, error: bool = False) -> None:
        """End the game"""
        if error:
            self._state = State.ERROR
        else:
            self._state = State.FINISHED

        # Calculate final results
        self._game_results = {
            "total_rounds": self._current_round + 1,
            "total_interactions": len(self._interaction_history),
            "scenario_type": self._scenario_type.value,
        }

    def get_results(self) -> dict[str, Any]:
        """Get game results"""
        return self._game_results.copy()

    def get_interaction_history(self) -> list[InteractionMessage]:
        """Get interaction history"""
        return self._interaction_history.copy()

    def _save_result(self, path: str, data: dict) -> None:
        """save result to file"""

        def _convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_numpy(item) for item in obj]
            return obj

        try:
            file_path = Path(path)

            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.suffix != ".jsonl":
                file_path = file_path.with_suffix(".jsonl")

            data = _convert_numpy(data)

            with open(file_path, "a", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")

            logger.info(f"result has saved in : {file_path}")

        except Exception as e:
            logger.error(f"save result error: {e}")
            raise
