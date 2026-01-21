import json
from tqdm import tqdm
from datetime import datetime
from loguru import logger
from pathlib import Path
import asyncio
import os
from utils import (
    calculate_metrics_local,
    extract_json_from_text,
    load_json,
    load_config,
)
from belief import MedQABelief, CourtDebateBelief, PersonaChatBelief, IBelief
from base import ScenarioType
from ollama import Ollama
from role import LLMParticipant
from simulator import InteractiveGameEngine
from prompts.role_system import get_system_prompt


def setup_global_logging():
    """Set up global logging configuration"""
    # Create log directory
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Remove default configuration
    logger.remove()

    # Console output
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # File output
    logger.add(
        sink=log_dir / f"{timestamp}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm} | {level} | {name}:{function}:{line} | {message}",
        encoding="utf-8",
    )


async def determine_specialties(question: str, util_llm: Ollama) -> tuple[str, str]:
    """
    Use LLM to dynamically determine the specialties for the two doctors based on the question content.

    Args:
        question: The medical question text.
        util_llm: The utility LLM for analysis.

    Returns:
        Tuple of (specialty1, specialty2) for the two doctors.
    """
    prompt = f"""
Analyze the following medical question and determine two complementary medical specialties that would be most relevant for discussing and diagnosing this case. The specialties should be different and provide diverse perspectives.

Question: {question}

Output the two specialties in JSON format:
```json
{{
    "specialty1": "Specialty Name 1",
    "specialty2": "Specialty Name 2"
}}
```
Ensure the specialties are real medical fields (e.g., Cardiology, Radiology, Infectious Diseases, etc.).
"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response, _ = await util_llm._chat(messages)
        result = extract_json_from_text(response)
        if result and "specialty1" in result and "specialty2" in result:
            specialty1 = result["specialty1"]
            specialty2 = result["specialty2"]
            return specialty1, specialty2
        else:
            raise ValueError("Failed to parse specialties from LLM response.")
    except Exception as e:
        print(f"Error in determine_specialties: {e}")
        return "General Medicine", "Internal Medicine"  # Default value


async def run(scenario: str = "medqa", retry_cases: list[str | int] = []) -> None:
    # Load configuration
    config = load_config(str(Path(__file__).parent / "config.yaml"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine scenario type and data
    if scenario == "medqa":
        id_key = "qid"
        scenario_type = ScenarioType.MEDICAL_QA
        data_path = str(Path(__file__).parent.parent / "data" / "medqa_mc_test.json")
        result_dir = str(
            Path(__file__).parent.parent
            / "results"
            / "medical_qa"
            / f"{config['role_llm']['model_name']}_{timestamp}"
        )
    elif scenario == "court":
        id_key = "case_id"
        scenario_type = ScenarioType.COURT_DEBATE
        data_path = str(Path(__file__).parent.parent / "data" / "criminal_cases.json")
        result_dir = str(
            Path(__file__).parent.parent
            / "results"
            / "court_debate"
            / f"{config['role_llm']['model_name']}_{timestamp}"
        )
    elif scenario == "persona":
        id_key = "id"
        scenario_type = ScenarioType.PERSONA_CHAT
        data_path = str(
            Path(__file__).parent.parent / "data" / "Synthetic_persona_chat_test.json"
        )
        result_dir = str(
            Path(__file__).parent.parent
            / "results"
            / "persona_chat"
            / f"{config['role_llm']['model_name']}_{timestamp}"
        )
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    # file path of result
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, "interaction_result.jsonl")

    llm: Ollama = Ollama(model_name=config["role_llm"]["model_name"])
    llm._set_system_prompt(system_prompt=get_system_prompt(scenario_type))

    util_llm: Ollama = Ollama(model_name=config["util_llm"]["model_name"])

    cases: list = load_json(data_path)

    error_list: list[str] = []


    if len(retry_cases) > 0:
        cases = [case for case in cases if case[id_key] in retry_cases]

    for case in tqdm(cases, desc=f"Simulating {scenario}"):
        id = case.get("qid", case.get("id", case.get("case_id", "unknown")))
        try:
            logger.info(f"Simulating case ID: {id})")

            # Create participants based on scenario
            if scenario == "medqa":
                # Dynamically determine specialties based on the question using LLM
                print(
                    f"================================ 生成角色职业================================"
                )
                specialty1, specialty2 = await determine_specialties(
                    case["question"], util_llm
                )

                # Update doctor profiles dynamically
                doctor1_profile = (
                    f"A {specialty1} specialist, skilled in related diagnoses"
                )
                doctor2_profile = (
                    f"A {specialty2} specialist, skilled in related diagnoses"
                )

                role1 = "Dr. Elaine Chen"
                role2 = "Dr. Marcus Li"
                profile1 = doctor1_profile
                profile2 = doctor2_profile

            elif scenario == "court":
                role1 = "原告"
                role2 = "被告"
                profile1 = "一名原告律师为控方提交证据并进行论证"
                profile2 = "一名被告律师在为指控进行辩护并提出反驳论点"

            elif scenario == "persona":
                # Assume data has persona info
                role1 = "Alex"
                role2 = "Frank"
                profile1 = " ".join(case.get("user_1_persona", [""]))
                profile2 = " ".join(case.get("user_2_persona", [""]))

            # Create appropriate belief objects based on scenario
            if scenario == "medqa":
                belief_p: IBelief = MedQABelief()
                belief_d: IBelief = MedQABelief()
            elif scenario == "court":
                belief_p: IBelief = CourtDebateBelief()
                belief_d: IBelief = CourtDebateBelief()
            elif scenario == "persona":
                belief_p: IBelief = PersonaChatBelief()
                belief_d: IBelief = PersonaChatBelief()
            else:
                raise ValueError(
                    f"Unsupported scenario for belief creation: {scenario}"
                )

            participant1: LLMParticipant = LLMParticipant(
                role=role1,
                profile=profile1,
                llm_chat=llm,
                beliefs={role2: belief_p},
            )

            participant2: LLMParticipant = LLMParticipant(
                role=role2,
                profile=profile2,
                llm_chat=llm,
                beliefs={role1: belief_d},
            )

            # Define discussion object for this case
            discussion: InteractiveGameEngine = InteractiveGameEngine(
                participants=[participant1, participant2],
                util=util_llm,
                scenario_type=scenario_type,
                max_rounds=config["max_rounds"],
            )

            # Prepare info based on scenario
            if scenario == "medqa":
                info = f"""Question: {case["question"]}\nOptions: {case.get("options", "None")}\nPlease discuss and provide a final diagnosis."""
            elif scenario == "court":
                info = f"""案件名: {case.get("case_name", "")}\n原告：{case.get("plaintiff", "")}\n被告：{case.get("defendant", "")}\n被告相关信息：{case.get("defendant_info","")}\n原告诉称：{case.get("plaintiff_claim","")}\n被告辩称：{case.get("defendant_claim","")}\n案件基本事实: {case.get("facts", "")}\n核心争议点: {case.get("key_issues", "")}\n"""
            elif scenario == "persona":
                utterances = case.get("utterances", [])
                history_str = "\n".join(
                    utterances[: config["real_chat_turns"]]
                )  # Use first few utterances as history
                info = f"""Previous conversation:\n{history_str}\nPlease continue the conversation as your persona."""

            await discussion.simulate(info=info, id=id, file_path=file_path)
        except Exception as e:
            logger.error(f"Error: {e}")
            error_list.append(id)

    logger.error(f"Error cases: {error_list}")


if __name__ == "__main__":
    import sys

    setup_global_logging()

    # Get scenario from command line argument(medqa, court, persona), default to medqa
    scenario = sys.argv[1] if len(sys.argv) > 1 else "medqa"

    # retry cases of ids
    retry_cases: list[str | int] = [
    ]
    # Simulation
    asyncio.run(run(scenario, retry_cases=retry_cases))
