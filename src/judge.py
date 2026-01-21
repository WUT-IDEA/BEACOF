import os
import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from ollama import Ollama
from utils import load_config, load_json, extract_json_from_text, calculate_metrics_local
from prompts.judge import judge_system_prompt, generate_verdict_prompt, extract_judge_result

config = load_config(str(Path(__file__).parent / "config.yaml"))
model_name = "qwen3:30b-a3b"

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
        sink=log_dir / f"judge_{timestamp}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm} | {level} | {name}:{function}:{line} | {message}",
        encoding="utf-8",
    )


async def judge(result_file_path: str, retry_list: list) -> None:
    # create judge
    judge: Ollama = Ollama(model_name=model_name)
    # judge: Ollama = Ollama(model_name=config["util_llm"]["model_name"])
    judge._set_system_prompt(system_prompt=judge_system_prompt())

    # collect data from judge
    result_dir = os.path.dirname(result_file_path)
    judge_result_file_path = os.path.join(result_dir, "judge_verdict.jsonl")

    error_list: list[str] = []

    # judge has no belief attribute
    interactions = []
    with open(result_file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            interactions.append(record)

    if len(retry_list) > 0:
        interactions = [rec for rec in interactions if rec["id"] in retry_list]

    raw_data: list = load_json(config["data"]["court_path"])

    for record in tqdm(interactions, desc="Judging cases"):
        try:
            # raw case information
            case = next((c for c in raw_data if c["case_id"] == record["id"]), None)
            if case is None:
                logger.error(f"Case data not found for ID: {record['id']}")
                continue

            debate_summary = []
            for speech in record["interaction_history"]:
                debate_summary.append(
                    f"第{speech['round']}轮 {speech['role']}发言: {speech['content']}"
                )

            prompt: str = generate_verdict_prompt(
                case_info={
                    "plaintiff": case["plaintiff"],
                    "defendant": case["defendant"],
                    "defendant_info": case["defendant_info"],
                    "case_type": case["case_type"],
                    "facts": case["facts"],
                },
                debate_summary=debate_summary,
            )
            response, consumption = await judge._chat(
                [{"role": "user", "content": prompt}]
            )
            verdict_data = {
                "id": record["id"],
                "verdict": response,
                "token": consumption["eval_count"],
            }
            logger.info(f"Judge verdict for case ID {record['id']}: {response}")

            with open(judge_result_file_path, "a", encoding="utf-8") as f:
                json.dump(verdict_data, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            logger.error(f"Error in judging case ID {record['id']}: {e}")
            error_list.append(record["id"])
    logger.error(f"Error cases: {error_list}")


if __name__ == "__main__":
    import asyncio

    setup_global_logging()
    result_file = ""

    retry_list = [
    ]
    asyncio.run(judge(result_file, retry_list))
