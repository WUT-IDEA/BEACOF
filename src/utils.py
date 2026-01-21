import bisect
from typing import Any
import aiohttp
import json
from loguru import logger
import yaml
import re
import numpy as np


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_json(json_path: str) -> Any:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cal_consensus_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate belief distance"""
    # Implementation details omitted for reproducibility
    return 0.0


async def call_ollama_api(
    model: str,
    prompt: str,
    base_url: str,
    temperature: float = 0,
    max_tokens: int = 4096,
    top_p: float = 0.9,
    top_k: int = 40,
    min_p: float = 0.0,
    repeat_penalty: float = 1.2,
    max_retries: int = 3,
) -> tuple:
    """Asynchronous call to Ollama API"""

    url = f"{base_url}/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": True,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status != 200:
                raise Exception(f"API call failed: {response.status}")

            full_response = ""
            final_context = {}

            # Process streaming response line by line
            async for line in response.content:
                if not line:
                    continue
                try:
                    # Decode and parse JSON
                    chunk = json.loads(line.decode("utf-8"))

                    # Concatenate model generated content
                    if "response" in chunk:
                        content_piece = chunk["response"]
                        full_response += content_piece

                        # Print to console in real time
                        print(content_piece, end="", flush=True)

                    # If it's the last part, save context information
                    if chunk.get("done", False):
                        final_context = chunk
                        break

                except json.JSONDecodeError:
                    logger.error(f"Unparseable JSON line: {line}")

            # After streaming output ends, print newline
            print("\n")

            # Extract resource consumption information
            resource_consumption = {
                "total_duration": final_context.get("total_duration", 0),
                "prompt_eval_count": final_context.get("prompt_eval_count", 0),
                "prompt_eval_duration": final_context.get("prompt_eval_duration", 0),
                "eval_count": final_context.get("eval_count", 0),
                "eval_duration": final_context.get("eval_duration", 0),
            }

            logger.info(
                f"Ollama API call completed. Model: {model}, Time: {resource_consumption['total_duration']/1e9:.2f}s, Tokens: {resource_consumption['eval_count']}"
            )

            # Remove think tags and return
            result_text = remove_think_tags(full_response.strip())
            return result_text, resource_consumption


async def call_ollama_api_chat(
    model: str,
    messages: list[dict[str, str]],
    base_url: str,
    temperature: float = 0,
    max_tokens: int = 4096,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0,
    print_stream: bool = True,
) -> tuple:
    """Asynchronous call to Ollama Chat API, supports streaming output"""

    url = f"{base_url}/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": True,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status != 200:
                raise Exception(f"API call failed: {response.status}")

            full_response = ""
            final_context = {}

            # Process streaming response line by line
            async for line in response.content:
                if not line:
                    continue

                try:
                    chunk = json.loads(line.decode("utf-8"))

                    # Chat API response format: { message: { role, content }, done: false }
                    content_piece = ""
                    if isinstance(chunk.get("message"), dict):
                        content_piece = chunk["message"].get("content", "")
                    elif "response" in chunk:  # Compatible with generate format
                        content_piece = chunk.get("response", "")

                    if content_piece:
                        full_response += content_piece
                        if print_stream:
                            print(content_piece, end="", flush=True)

                    if chunk.get("done", False):
                        final_context = chunk
                        break

                except json.JSONDecodeError:
                    logger.error(f"Unparseable JSON line: {line}")

            if print_stream:
                print("\n")

            resource_consumption = {
                "total_duration": final_context.get("total_duration", 0),
                "prompt_eval_count": final_context.get("prompt_eval_count", 0),
                "prompt_eval_duration": final_context.get("prompt_eval_duration", 0),
                "eval_count": final_context.get("eval_count", 0),
                "eval_duration": final_context.get("eval_duration", 0),
            }

            logger.info(
                f"Ollama Chat call successful - Model: {model}, Time: {resource_consumption['total_duration']/1e9:.2f}s, Tokens: {resource_consumption['eval_count']}"
            )

            # Remove think tags and return
            result_text = remove_think_tags(full_response.strip())
            return result_text, resource_consumption


def remove_think_tags(text: str) -> str:
    """Remove think tags and their content from text"""
    import re

    # Remove <think>...</think> tags and their content
    # Use non-greedy matching, support multi-line content
    think_pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(think_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove <thinking>...</thinking> tags and their content
    thinking_pattern = r"<thinking>.*?</thinking>"
    cleaned_text = re.sub(
        thinking_pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE
    )

    # Clean up extra whitespace characters
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)  # Remove extra blank lines
    cleaned_text = cleaned_text.strip()  # Remove leading and trailing whitespace

    return cleaned_text


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract JSON content from text"""
    try:
        # Remove think tags
        text = remove_think_tags(text)

        # Find all JSON code blocks, take the last one
        json_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
        matches = re.findall(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            json_str = matches[-1].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try ast.literal_eval as alternative
                import ast

                try:
                    return ast.literal_eval(json_str)
                except (ValueError, SyntaxError):
                    pass

        # If no code blocks, try to find the last {...}
        brace_pattern = r"\{.*\}"
        brace_match = re.search(brace_pattern, text, re.DOTALL)
        if brace_match:
            json_str = brace_match.group().strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                import ast

                try:
                    return ast.literal_eval(json_str)
                except (ValueError, SyntaxError):
                    pass

        # Finally try to parse the entire text directly
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            import ast

            try:
                return ast.literal_eval(text.strip())
            except (ValueError, SyntaxError):
                pass

    except Exception as e:
        logger.error(f"JSON extraction failed: {e}")
        return {}


def calculate_metrics_local(
    extracted_info: dict, reference_info: dict
) -> dict[str, Any]:

    fine_bound = [0, 1001, 5001, 10001, 50001, 100001, 500001, 1000001]
    sentence_bound = [0, 7, 13, 37, 61, 121]
    probation_bound = [0, 13, 25, 61]

    extracted_legal_citations = extracted_info["legal_citations"]

    extracted_charge = extracted_info["verdict_result"]["charge"]
    extracted_charge_index = extracted_charge.find("guilty of")
    extracted_charge = extracted_charge[extracted_charge_index:]

    extracted_sentence = extracted_info["verdict_result_num"]["sentence"]
    extracted_probation = extracted_info["verdict_result_num"]["probation"]
    extracted_fine = extracted_info["verdict_result_num"]["fine"]

    reference_legal_citations = reference_info["legal_citations"]

    reference_charge = reference_info["charge"]
    reference_charge_index = reference_charge.find("guilty of")
    reference_charge = reference_charge[reference_charge_index:]

    reference_sentence = reference_info["sentence_num"]
    reference_probation = reference_info["probation"]
    reference_fine = reference_info["fine_num"]

    result = {
        "legal_citations_metrics": {"precision": 0, "recall": 0, "f1-score": 0},
        "verdict_result_metrics": {"charge": 0, "prison_term": 0, "fine": 0},
    }
    l_p = l_r = 0
    for pre_legal_citation in extracted_legal_citations:
        for ref_legal_citation in reference_legal_citations:
            if pre_legal_citation == ref_legal_citation:
                l_p += 1
                l_r += 1
                break
            # Handle substring issues
            elif (
                pre_legal_citation in ref_legal_citation
                or ref_legal_citation in pre_legal_citation
            ):
                l_p += 1
                l_r += 1
                break
    if len(extracted_legal_citations) == 0:
        result["legal_citations_metrics"]["precision"] = 0.0
        result["legal_citations_metrics"]["recall"] = 0.0
        result["legal_citations_metrics"]["f1-score"] = 0.0
    else:
        result["legal_citations_metrics"]["precision"] = l_p / len(
            extracted_legal_citations
        )
        result["legal_citations_metrics"]["recall"] = l_r / len(
            reference_legal_citations
        )
        result["legal_citations_metrics"]["f1-score"] = (
            2
            * (
                result["legal_citations_metrics"]["precision"]
                * result["legal_citations_metrics"]["recall"]
            )
            / (
                result["legal_citations_metrics"]["precision"]
                + result["legal_citations_metrics"]["recall"]
            )
            if (
                result["legal_citations_metrics"]["precision"]
                + result["legal_citations_metrics"]["recall"]
            )
            > 0
            else 0.0
        )

    result["verdict_result_metrics"]["charge"] = (
        1 if extracted_charge == reference_charge else 0
    )
    # Calculate prison term
    if reference_probation == 0:
        if extracted_probation == 0:
            if in_same_range(reference_sentence, extracted_sentence, sentence_bound):
                result["verdict_result_metrics"]["prison_term"] = 1
            else:
                result["verdict_result_metrics"]["prison_term"] = 0
        else:
            result["verdict_result_metrics"]["prison_term"] = 0
    else:
        if in_same_range(
            reference_sentence, extracted_sentence, sentence_bound
        ) and in_same_range(reference_probation, extracted_probation, probation_bound):
            result["verdict_result_metrics"]["prison_term"] = 1
        else:
            result["verdict_result_metrics"]["prison_term"] = 0

    # Calculate fine
    if reference_fine == 0:
        if extracted_fine == 0:
            result["verdict_result_metrics"]["fine"] = 1
        else:
            result["verdict_result_metrics"]["fine"] = 0
    else:
        if in_same_range(reference_fine, extracted_fine, fine_bound):
            result["verdict_result_metrics"]["fine"] = 1
        else:
            result["verdict_result_metrics"]["fine"] = 0

    return result


# return the range of a value given the bounds
def get_range(value, bounds):
    idx = bisect.bisect_right(bounds, value)
    lower = bounds[idx - 1]
    upper = bounds[idx] if idx < len(bounds) else float("inf")
    return (lower, upper)


# compaire if two values are in the same range
def in_same_range(val1, val2, bounds):
    return get_range(val1, bounds) == get_range(val2, bounds)


def append_json_to_jsonl(json_file: str, jsonl_file: str) -> None:
    """
    Append the contents of a JSON file to a JSONL file.
    Assumes the JSON file contains a list of objects or a single object.
    Each object is appended as a separate line in the JSONL file.

    Args:
        json_file (str): Path to the input JSON file.
        jsonl_file (str): Path to the output JSONL file.
    """
    try:
        # Load the JSON data
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Open the JSONL file in append mode
        with open(jsonl_file, "a", encoding="utf-8") as f:
            if isinstance(data, list):
                # If data is a list, append each item as a separate line
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            elif isinstance(data, dict):
                # If data is a single object, append it as one line
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
            else:
                raise ValueError("JSON file must contain a list or a dict.")

        logger.info(f"Successfully appended data from {json_file} to {jsonl_file}")

    except Exception as e:
        logger.error(f"Error appending JSON to JSONL: {e}")
        raise


def sort_jsonl_by_id(jsonl_file: str) -> None:
    """
    Sort the JSONL file by the 'id' field in each JSON object.
    Assumes 'id' is in format like 'case_10', sorts by the numeric part.
    Overwrites the original file with sorted data.

    Args:
        jsonl_file (str): Path to the JSONL file to sort.
    """
    try:
        # Read all lines and parse JSON
        data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        # Sort by 'id' - extract numeric part after '_'
        def sort_key(item):
            id_str = item.get("id", "")
            try:
                # Assume format 'case_10', extract 10 as int
                num_part = id_str.split("_")[-1]
                return int(num_part)
            except (ValueError, IndexError):
                # If not numeric, sort as string
                return id_str

        data_sorted = sorted(data, key=sort_key)

        # Write back to file
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for item in data_sorted:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"Successfully sorted {jsonl_file} by 'id' (numeric order)")

    except Exception as e:
        logger.error(f"Error sorting JSONL by id: {e}")
        raise


def del_cases(path, ids, id_key="id"):
    """
    Delete specific data from jsonl file based on id.

    Args:
        path (str): Path to the jsonl file
        ids (list or str): List of ids to delete, or a single id string
    """
    # Ensure ids is a list
    if isinstance(ids, str):
        ids = [ids]
    elif not isinstance(ids, list):
        raise ValueError("ids must be a string or list of strings")

    # Read all lines
    with open(path, "r", encoding="utf-8") as f:
        interaction_results = [json.loads(line.strip()) for line in f if line.strip()]

    # Filter out entries where id is in ids
    filtered_results = [
        item for item in interaction_results if item.get(id_key) not in ids
    ]

    # Write back to file
    with open(path, "w", encoding="utf-8") as f:
        for item in filtered_results:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    logger.info(
        f"Deleted {len(interaction_results) - len(filtered_results)} cases from {path}"
    )


if __name__ == "__main__":
    pass