import asyncio
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from ollama import Ollama
from openai import OpenAI
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prompts.eval_coherence import evaluate_coherence_prompt
import torch
from collections import Counter
import statistics
import math
from utils import extract_json_from_text, calculate_metrics_local, load_config
from prompts.judge import (
    extract_judge_result,
)

config = load_config(str(Path(__file__).parent / "config.yaml"))

class GPTClient:
    def __init__(self, gpt_config):
        self.config = gpt_config
        self.client = OpenAI(
            api_key=gpt_config.api_key,
            base_url=gpt_config.base_url,
        )

    def generate_response(self, messages, temperature=None, max_tokens=None):
        """生成响应"""
        if temperature is None:
            temperature = self.config.temperature
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        try:
            # print(
            #     f"GPT Messages: {json.dumps(messages, indent=2, ensure_ascii=False)}\n"
            # )
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"GPT调用失败: {e}")
            return None


def setup_global_logging():
    """设置全局日志配置"""
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
        sink=log_dir / f"evaluate_{timestamp}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm} | {level} | {name}:{function}:{line} | {message}",
        encoding="utf-8",
    )


def evaluate_court(
    interaction_file_path: str, calculation_result_file_path: str
) -> None:
    """Statistics metrics from calculation results"""

    # Interaction results
    with open(interaction_file_path, "r", encoding="utf-8") as f:
        interaction_results: list = [json.loads(line) for line in f]

    with open(calculation_result_file_path, "r", encoding="utf-8") as f:
        calculation_results: list = [json.loads(line) for line in f]

    assert len(interaction_results) == len(
        calculation_results
    ), "Interaction results and calculation results must have the same number of cases."

    logger.info(f"Total cases evaluated: {len(calculation_results)}")

    interaction_total_cases = len(interaction_results)
    if interaction_total_cases == 0:
        logger.error("No interaction results found.")
        return

    # Statistics of average interaction tokens
    avg_payoff_tokens = (
        sum(
            sum(item["token"]["payoff_token"]) / len(item["token"]["payoff_token"])
            for item in interaction_results
        )
        / interaction_total_cases
    )
    logger.info(f"Average interaction tokens: {avg_payoff_tokens:.2f}")

    # Statistics of average action prediction tokens
    avg_action_predict_tokens = (
        sum(
            sum(item["token"]["action_predict_token"])
            / len(item["token"]["action_predict_token"])
            for item in interaction_results
        )
        / interaction_total_cases
    )
    logger.info(f"Average action prediction tokens: {avg_action_predict_tokens:.2f}")

    # Statistics of average interaction rounds
    avg_rounds = (
        sum(item["interaction_history"][-1]["round"] for item in interaction_results)
        / interaction_total_cases
    )
    logger.info(f"Average interaction rounds: {avg_rounds:.2f}")

    # Statistics of average interaction strategy types
    roles = set(item["role"] for item in interaction_results[0]["interaction_history"])
    role_strategy_totals = {role: {"合作": 0, "竞争": 0, "合竞": 0} for role in roles}

    for interaction_result in interaction_results:
        # calculate strategy proportion for each role in a single interaction
        role_strategy_counts = {
            role: {"合作": 0, "竞争": 0, "合竞": 0} for role in roles
        }

        # count strategy occurrences
        for interaction in interaction_result["interaction_history"]:
            role_strategy_counts[interaction["role"]][interaction["strategy"]] += 1

        for role in roles:
            total_interactions_for_role = sum(role_strategy_counts[role].values())
            if total_interactions_for_role > 0:
                for strategy in ["合作", "竞争", "合竞"]:
                    strategy_proportion = (
                        role_strategy_counts[role][strategy]
                        / total_interactions_for_role
                    )
                    role_strategy_totals[role][strategy] += strategy_proportion

    logger.info("Average strategy proportions across all interactions:")
    for role in roles:
        avg_cooperation = role_strategy_totals[role]["合作"] / interaction_total_cases
        avg_competition = role_strategy_totals[role]["竞争"] / interaction_total_cases
        avg_coopetition = role_strategy_totals[role]["合竞"] / interaction_total_cases

        logger.info(f"Role: {role}")
        logger.info(f"  - 合作 (Cooperation): {avg_cooperation:.2f}")
        logger.info(f"  - 竞争 (Competition): {avg_competition:.2f}")
        logger.info(f"  - 合竞 (Coopetition): {avg_coopetition:.2f}")

    # Statistics from calculation results
    total_cases = len(calculation_results)
    if total_cases == 0:
        logger.error("No calculation results found.")
        return

    avg_precision = (
        sum(
            item["metrics_calculation"]["legal_citations_metrics"]["precision"]
            for item in calculation_results
        )
        / total_cases
    )
    avg_recall = (
        sum(
            item["metrics_calculation"]["legal_citations_metrics"]["recall"]
            for item in calculation_results
        )
        / total_cases
    )
    avg_f1 = (
        sum(
            item["metrics_calculation"]["legal_citations_metrics"]["f1-score"]
            for item in calculation_results
        )
        / total_cases
    )

    avg_charge = (
        sum(
            item["metrics_calculation"]["verdict_result_metrics"]["charge"]
            for item in calculation_results
        )
        / total_cases
    )
    avg_prison_term = (
        sum(
            item["metrics_calculation"]["verdict_result_metrics"]["prison_term"]
            for item in calculation_results
        )
        / total_cases
    )
    avg_fine = (
        sum(
            item["metrics_calculation"]["verdict_result_metrics"]["fine"]
            for item in calculation_results
        )
        / total_cases
    )

    logger.info("Average Legal Citation Metrics:")
    logger.info(f"- Precision: {avg_precision:.3f}")
    logger.info(f"- Recall: {avg_recall:.3f}")
    logger.info(f"- F1 Score: {avg_f1:.3f}")

    logger.info("Average Verdict Result Metrics:")
    logger.info(f"- Charge Accuracy: {avg_charge:.3f}")
    logger.info(f"- Prison Term Accuracy: {avg_prison_term:.3f}")
    logger.info(f"- Fine Accuracy: {avg_fine:.3f}")


def statistic_court(file_path):
    """Statistics metrics from judge results"""
    with open(file_path, "r", encoding="utf-8") as f:
        judge_results: list = [json.loads(line) for line in f]

    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    charge: list[float] = []
    prison: list[float] = []
    fine: list[float] = []

    for judge_item in judge_results:
        precision.append(
            judge_item["metrics_calculation"]["legal_citations_metrics"]["precision"]
        )
        recall.append(
            judge_item["metrics_calculation"]["legal_citations_metrics"]["recall"]
        )
        f1.append(
            judge_item["metrics_calculation"]["legal_citations_metrics"]["f1-score"]
        )
        charge.append(
            judge_item["metrics_calculation"]["verdict_result_metrics"]["charge"]
        )
        prison.append(
            judge_item["metrics_calculation"]["verdict_result_metrics"]["prison_term"]
        )
        fine.append(judge_item["metrics_calculation"]["verdict_result_metrics"]["fine"])

    logger.info("Statistics Average Metrics:")
    logger.info(f"- Precision: {sum(precision) / len(precision):.3f}")
    logger.info(f"- Recall: {sum(recall) / len(recall):.3f}")
    logger.info(f"- F1 Score: {sum(f1) / len(f1):.3f}")
    logger.info(f"- Charge Accuracy: {sum(charge) / len(charge):.3f}")
    logger.info(f"- Prison Term Accuracy: {sum(prison) / len(prison):.3f}")
    logger.info(f"- Fine Accuracy: {sum(fine) / len(fine):.3f}")


async def evaluate_law_metrics(
    judge_result_file_path: str, retry_list: list = [], model_name: str = "qwen3:30b-a3b"
) -> None:
    """Evaluate legal metrics based on judge results"""

    with open(config["data"]["court_path"], "r", encoding="utf-8") as f:
        cases_data: list = json.load(f)

    with open(judge_result_file_path, "r", encoding="utf-8") as f:
        judge_results: list = [json.loads(line) for line in f]

    # collect data from judge
    result_dir = os.path.dirname(judge_result_file_path)
    calculation_result_file_path = os.path.join(result_dir, "calculation_result.jsonl")

    llm = Ollama(model_name=model_name)

    if len(retry_list) > 0:
        judge_results = [item for item in judge_results if item["id"] in retry_list]

    error_list: list[str] = []

    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    charge: list[float] = []
    prison: list[float] = []
    fine: list[float] = []

    for judge_item in tqdm(judge_results, desc="Evaluating legal metrics"):
        try:
            if len(judge_item["verdict"]) == 0:
                logger.error(
                    f"Content of verdict is empty，case_id: {judge_item['id']}"
                )
                error_list.append(judge_item["id"])
                continue

            case = next(
                (c for c in cases_data if c["case_id"] == judge_item["id"]), None
            )

            prompt = extract_judge_result(final_verdict=judge_item["verdict"])
            response, consumption = await llm._chat(
                messages=[{"role": "user", "content": prompt}]
            )

            logger.info(f"Judge result extraction response: {response}")

            extracted_info = extract_json_from_text(response)

            # remove invalid spaces
            for idx, legal in enumerate(extracted_info["legal_citations"]):
                extracted_info["legal_citations"][idx] = legal.replace(" ", "")
            for verdict in extracted_info["verdict_result"].keys():
                extracted_info["verdict_result"][verdict] = extracted_info[
                    "verdict_result"
                ][verdict].replace(" ", "")

            metrics_calculation = calculate_metrics_local(
                extracted_info,
                {
                    "legal_citations": case["legal_citations"],
                    "charge": case["charge"],
                    "sentence_num": case["sentence_num"],
                    "probation": case["probation"],
                    "fine_num": case["fine_num"],
                },
            )

            logger.info(f"法律推理F1分数计算完成:")
            logger.info(
                f"Metrics of law citation\n- Precision: {metrics_calculation['legal_citations_metrics']['precision']:.3f}, Recall: {metrics_calculation['legal_citations_metrics']['recall']:.3f}, F1: {metrics_calculation['legal_citations_metrics']['f1-score']:.3f}"
            )
            logger.info(
                f"Metrics of verdict\n- charge: {metrics_calculation['verdict_result_metrics']['charge']:.3f}, prison_term: {metrics_calculation['verdict_result_metrics']['prison_term']:.3f}, fine: {metrics_calculation['verdict_result_metrics']['fine']:.3f}"
            )

            precision.append(
                metrics_calculation["legal_citations_metrics"]["precision"]
            )
            recall.append(metrics_calculation["legal_citations_metrics"]["recall"])
            f1.append(metrics_calculation["legal_citations_metrics"]["f1-score"])
            charge.append(metrics_calculation["verdict_result_metrics"]["charge"])
            prison.append(metrics_calculation["verdict_result_metrics"]["prison_term"])
            fine.append(metrics_calculation["verdict_result_metrics"]["fine"])

            with open(calculation_result_file_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "id": judge_item["id"],
                        "extracted_info": extracted_info,
                        "metrics_calculation": metrics_calculation,
                        "token": consumption["eval_count"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

        except Exception as e:
            logger.error(f"Error in judging case ID {judge_item['id']}: {e}")
            error_list.append(judge_item["id"])

    logger.info("Overall Average Metrics:")
    if precision:
        logger.info(f"- Avg Precision: {sum(precision) / len(precision):.3f}")
    if recall:
        logger.info(f"- Avg Recall: {sum(recall) / len(recall):.3f}")
    if f1:
        logger.info(f"- Avg F1 Score: {sum(f1) / len(f1):.3f}")
    if charge:
        logger.info(f"- Avg Charge Accuracy: {sum(charge) / len(charge):.3f}")
    if prison:
        logger.info(f"- Avg Prison Term Accuracy: {sum(prison) / len(prison):.3f}")
    if fine:
        logger.info(f"- Avg Fine Accuracy: {sum(fine) / len(fine):.3f}")

    logger.error(f"Error cases: {error_list}")


def evaluate_medqa(interaction_file_path: str, ground_truth_file_path: str, toggle=True) -> None:
    """Evaluate Medical QA interaction results"""

    # Interaction results
    with open(interaction_file_path, "r", encoding="utf-8") as f:
        interaction_results: list = [json.loads(line) for line in f]

    with open(ground_truth_file_path, "r", encoding="utf-8") as f:
        raw_data: list = json.load(f)

    # assert len(interaction_results) == len(
    #     raw_data
    # ), "Interaction results length and raw data length must have the same number of cases."

    logger.info(f"Total cases evaluated: {len(interaction_results)}")

    total_cases = len(interaction_results)
    if total_cases == 0:
        logger.error("No interaction results found.")
        return

    correct_answers = 0

    error_ids = []

    for idx, item in enumerate(interaction_results):
        # assert (
        #     item["interaction_history"][-2]["answer"]
        #     == item["interaction_history"][-1]["answer"]
        # ), "Answer inconsistency in interaction history of last two interactions!"

        ground_truth: str = [d["answer"] for d in raw_data if d["qid"] == item["id"]][0]
        if toggle:
            final_answer = item["interaction_history"][-1]["answer"]
        else:
            try:
                res = item["interaction_history"][-1]["content"].strip('"')
                final_answer = extract_json_from_text(res)
                final_answer = final_answer["answer"]
            except Exception as e:
                # logger.error(f"Error parsing response for item {item["id"]}: {e}")
                error_ids.append(item["id"])
                continue

        if final_answer.strip().lower() == ground_truth.strip().lower():
            correct_answers += 1
    accuracy = correct_answers / total_cases

    logger.info(f"Error parsing response for items, length: {len(error_ids)}:\n{error_ids}")
    logger.info(f"Medical QA Accuracy: {accuracy:.3f}")


def evaluate_chat(interaction_file_path: str) -> None:
    """Evaluate Persona Chat interaction results"""

    # Interaction results
    with open(interaction_file_path, "r", encoding="utf-8") as f:
        interaction_results: list = [json.loads(line) for line in f]

    total_cases = len(interaction_results)
    if total_cases == 0:
        logger.error("No interaction results found.")
        return

    logger.info(f"Total chat cases evaluated: {total_cases}")

    # Load Persona data
    persona_path = config["data"]["persona_path"]
    with open(persona_path, "r", encoding="utf-8") as f:
        persona_data = json.load(f)

    # Create persona mapping by id
    persona_mapping = {}
    for item in persona_data:
        item_id = item["id"]
        user1_personas = item.get("user_1_persona", [])
        user2_personas = item.get("user_2_persona", [])
        persona_mapping[item_id] = {"user_1": user1_personas, "user_2": user2_personas}

    logger.info(f"Loaded persona mapping for {len(persona_mapping)} cases")

    # Determine role mapping for each interaction
    role_mapping = {}
    for interaction in interaction_results:
        interaction_id = interaction["id"]
        turns = interaction["interaction_history"]
        if turns:
            first_role = turns[0]["role"]
            second_role = turns[1]["role"]
            if second_role:
                role_mapping[interaction_id] = {
                    first_role: "user_1",
                    second_role: "user_2",
                }
            else:
                # Only one role, assume it's user_1
                role_mapping[interaction_id] = {first_role: "user_1"}

    logger.info(f"Determined role mappings for {len(role_mapping)} interactions")

    # Load NLI model
    model_path = config["nli_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Download NLTK punkt if not available
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Evaluate each interaction
    all_entail = []
    all_contr = []
    all_neutral = []
    all_consistency = []

    for interaction in tqdm(interaction_results[:1], desc="Evaluating NLI"):
        interaction_id = interaction["id"]
        turns = interaction["interaction_history"]
        turn_entail = []
        turn_contr = []
        turn_neutral = []
        turn_consistency = []

        for turn in turns:
            if "content" in turn:
                response = turn["content"]
                role = turn["role"]

                # Get personas for this role
                if interaction_id in persona_mapping and interaction_id in role_mapping:
                    user_type = role_mapping[interaction_id].get(role)
                    if user_type:
                        personas = persona_mapping[interaction_id][user_type]
                    else:
                        personas = []
                else:
                    personas = []

                entail_u, contr_u, neutral_u, consistency_u = evaluate_turn_nli(
                    response, personas, tokenizer, model, device
                )
                turn_entail.append(entail_u)
                turn_contr.append(contr_u)
                turn_neutral.append(neutral_u)
                turn_consistency.append(consistency_u)

        if turn_entail:
            avg_entail = sum(turn_entail) / len(turn_entail)
            avg_contr = sum(turn_contr) / len(turn_contr)
            avg_neutral = sum(turn_neutral) / len(turn_neutral)
            avg_consistency = sum(turn_consistency) / len(turn_consistency)

            all_entail.append(avg_entail)
            all_contr.append(avg_contr)
            all_neutral.append(avg_neutral)
            all_consistency.append(avg_consistency)

    if all_entail:
        final_avg_entail = sum(all_entail) / len(all_entail)
        final_avg_contr = sum(all_contr) / len(all_contr)
        final_avg_neutral = sum(all_neutral) / len(all_neutral)
        final_avg_consistency = sum(all_consistency) / len(all_consistency)

        logger.info("NLI Evaluation Results:")
        logger.info(f"- AvgEntail: {final_avg_entail:.4f}")
        logger.info(f"- AvgNeutral: {final_avg_neutral:.4f}")
        logger.info(f"- AvgContr: {final_avg_contr:.4f}")
        logger.info(f"- AvgConsistency: {final_avg_consistency:.4f}")
    else:
        logger.warning("No valid turns found for NLI evaluation")


def evaluate_diversity(interaction_file_path: str) -> None:
    """Evaluate diversity metrics for persona chat interactions.

    Metrics computed:
    - distinct-1: unique unigrams / total unigrams
    - distinct-2: unique bigrams / total bigrams
    - avg_response_length (tokens) and std
    - token entropy across all responses
    Metrics are reported overall and per role when possible.
    """

    # Interaction results
    with open(interaction_file_path, "r", encoding="utf-8") as f:
        interaction_results: list = [json.loads(line) for line in f]

    total_cases = len(interaction_results)
    if total_cases == 0:
        logger.error("No interaction results found for diversity evaluation.")
        return

    # Ensure nltk punkt available for tokenization
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    from nltk import word_tokenize

    # Collect tokens and response lengths
    all_tokens = []
    all_bigrams = []
    response_lengths = []
    role_tokens = {}

    for interaction in interaction_results:
        interaction_id = interaction["id"]
        turns = interaction.get("interaction_history", [])
        for turn in turns:
            # support fields: content or answer
            text = turn.get("content")
            if not text or not text.strip():
                continue
            tokens = word_tokenize(text)
            tokens = [t.lower() for t in tokens if t.strip()]
            if not tokens:
                continue

            all_tokens.extend(tokens)
            response_lengths.append(len(tokens))
            bigrams = list(zip(tokens, tokens[1:]))
            all_bigrams.extend(bigrams)

            role = turn.get("role")
            if role:
                role_tokens.setdefault(role, []).extend(tokens)

    # Compute distinct metrics
    total_unigrams = len(all_tokens)
    total_bigrams = len(all_bigrams)
    unique_unigrams = len(set(all_tokens))
    unique_bigrams = len(set(all_bigrams))

    distinct_1 = unique_unigrams / total_unigrams if total_unigrams > 0 else 0.0
    distinct_2 = unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0

    avg_len = statistics.mean(response_lengths) if response_lengths else 0.0
    std_len = statistics.pstdev(response_lengths) if response_lengths else 0.0

    # Token entropy
    token_counts = Counter(all_tokens)

    def entropy_from_counts(counter: Counter) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0.0
        return -sum(
            (c / total) * math.log2(c / total) for c in counter.values() if c > 0
        )

    token_entropy = entropy_from_counts(token_counts)

    logger.info("Diversity Metrics (overall):")
    logger.info(f"- Total responses (cases): {total_cases}")
    logger.info(f"- Distinct-1 (unique unigrams / total): {distinct_1:.4f}")
    logger.info(f"- Distinct-2 (unique bigrams / total): {distinct_2:.4f}")
    logger.info(f"- Avg response length (tokens): {avg_len:.2f}")
    logger.info(f"- Response length stddev: {std_len:.2f}")
    logger.info(f"- Token entropy (bits): {token_entropy:.4f}")

    # Compute a normalized entropy in [0,1] by dividing by max possible entropy log2(V)
    norm_entropy = 0.0
    if unique_unigrams > 1:
        norm_entropy = token_entropy / math.log2(unique_unigrams)

    # Composite diversity score: average of distinct-1, distinct-2 and normalized entropy
    diversity_score = (distinct_1 + distinct_2 + norm_entropy) / 3.0
    logger.info(f"- Normalized entropy: {norm_entropy:.4f}")
    logger.info(f"- Composite diversity score: {diversity_score:.4f}")

    # Per-role breakdown
    if role_tokens:
        logger.info("Diversity Metrics (per role):")
        for role, tokens in role_tokens.items():
            tot = len(tokens)
            uniq = len(set(tokens))
            # bigrams for role
            role_bigrams = list(zip(tokens, tokens[1:]))
            role_tot_bi = len(role_bigrams)
            role_uniq_bi = len(set(role_bigrams))
            role_dist1 = uniq / tot if tot > 0 else 0.0
            role_dist2 = role_uniq_bi / role_tot_bi if role_tot_bi > 0 else 0.0
            role_entropy = entropy_from_counts(Counter(tokens))
            role_norm_entropy = 0.0
            role_uniq = len(set(tokens))
            if role_uniq > 1:
                role_norm_entropy = role_entropy / math.log2(role_uniq)
            role_diversity = (role_dist1 + role_dist2 + role_norm_entropy) / 3.0
            logger.info(
                f"Role: {role} - Dist1: {role_dist1:.4f}, Dist2: {role_dist2:.4f}, Entropy: {role_entropy:.4f}, NormEntropy: {role_norm_entropy:.4f}, Diversity: {role_diversity:.4f}"
            )


def evaluate_turn_nli(response: str, personas: list, tokenizer, model, device) -> tuple:
    """Evaluate a single turn response using NLI"""
    # Split into sentences
    sentences = nltk.sent_tokenize(response)

    if not sentences:
        return 0.0, 0.0, 0.0, 0.0

    sentence_entail = []
    sentence_contr = []
    sentence_neutral = []

    # Per-sentence NLI
    for sentence in sentences:
        entail_probs = []
        contr_probs = []
        neutral_probs = []

        for persona in personas:
            # Get NLI scores
            inputs = tokenizer(
                persona, sentence, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                # outputs.logits is (batch_size, num_labels)
                # Here batch_size=1, num_labels=3
                probs = torch.softmax(outputs.logits, dim=-1)

            # Assuming label order: entailment, neutral, contradiction
            entail_prob = probs[0][0].item()  # Index 0 is entailment
            neutral_prob = probs[0][1].item()  # Index 1 is neutral
            contr_prob = probs[0][2].item()  # Index 2 is contradiction

            entail_probs.append(entail_prob)
            contr_probs.append(contr_prob)
            neutral_probs.append(neutral_prob)

        avg_entail = sum(entail_probs) / len(entail_probs) if entail_probs else 0.0
        avg_contr = sum(contr_probs) / len(contr_probs) if contr_probs else 0.0
        avg_neutral = sum(neutral_probs) / len(neutral_probs) if neutral_probs else 0.0
        
        sentence_entail.append(avg_entail)
        sentence_contr.append(avg_contr)
        sentence_neutral.append(avg_neutral)

    # Per-turn aggregation
    contr_u = sum(sentence_contr) / len(sentence_contr) if sentence_contr else 0.0
    entail_u = sum(sentence_entail) / len(sentence_entail) if sentence_entail else 0.0
    neutral_u = sum(sentence_neutral) / len(sentence_neutral) if sentence_neutral else 0.0
    consistency_u = entail_u - contr_u

    return entail_u, contr_u, neutral_u, consistency_u

if __name__ == "__main__":

    eval_type = sys.argv[1] if len(sys.argv) > 1 else "court"

    if eval_type == "court":
        retry_list = []
        asyncio.run(
            evaluate_law_metrics(
                "results/CAMEL/court_debate/gemma3:12b_20251122_104225_retest1/judge_verdict.jsonl",
                retry_list,
                model_name="qwen3:30b-a3b",
            )
        )
        # statistic_court(
        #     "results/court_debate/gemma3:12b_20251119_163743_retest2/calculation_result.jsonl"
        # )
    elif eval_type == "medqa":
        evaluate_medqa(
            "results/CAMEL/medqa/gemma3:12b_20251120_005410_retest2/interaction_result.jsonl",
            "data/medqa_mc_test.json",
            toggle=True,  # medqa -> True, mad -> False
        )
    elif eval_type == "persona":
        persona_path = "results/CAMEL/persona/gemma3:12b_20251122_103231_retest2/interaction_result.jsonl"
        evaluate_chat(persona_path)
        evaluate_diversity(persona_path)
