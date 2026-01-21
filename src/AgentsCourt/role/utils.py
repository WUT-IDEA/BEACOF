import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from functools import wraps
from typing import Any, List
import traceback

# Explicitly import from the project's messages module to avoid conflicts with system packages
from role.messages import OpenAIMessage
from role.typeC import ModelType


def count_tokens_ollama_chat_models(
    messages: List[OpenAIMessage],
) -> int:
    """使用简化的token计算方法，适用于Ollama模型"""
    num_tokens = 0
    for message in messages:
        # Simplified token calculation: approximately 1 token per word
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(value.split())
    return num_tokens


def num_tokens_from_messages(
    messages: List[OpenAIMessage],
    model: ModelType,
) -> int:
    """Returns the number of tokens used by a list of messages."""
    # Since using Ollama, all model types use the same token calculation method
    return count_tokens_ollama_chat_models(messages)


def get_model_token_limit(model: ModelType) -> int:
    """Returns the maximum number of tokens for a given model."""
    # Since using Ollama's qwen3:30b-a3b model, set a reasonable token limit
    # Qwen models usually support larger context lengths
    return 32768  # Unified to 32k tokens


def ollama_available_required(func):
    """检查Ollama服务是否可用的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import requests
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code == 200:
                return func(*args, **kwargs)
            else:
                raise ValueError('Ollama服务不可用，请确保Ollama正在运行')
        except Exception as e:
            raise ValueError(f'无法连接到Ollama服务: {e}')

    return wrapper


def print_text_animated(text, delay=0.01, end=""):
    for char in text:
        print(char, end=end, flush=True)
        time.sleep(delay)
    print('\n')


class Prompt:
    def __init__(
        self,
        question_prefix: str,
        answer_prefix: str,
        intra_example_sep: str,
        inter_example_sep: str,
        engine: str = None,
        temperature: float = None,
    ) -> None:
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.intra_example_sep = intra_example_sep
        self.inter_example_sep = inter_example_sep
        self.engine = engine
        self.temperature = temperature

    def make_query(self, prompt: str, question: str) -> str:
        return (
            f"{prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        )

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



def retry_parse_fail_prone_cmd(
    func,
    max_retries: int = 3,
    exceptions=(
        ValueError,
        KeyError,
        IndexError,
    ),
):
    def wrapper(*args, **kwargs):
        retries = max_retries
        while retries:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                stack_trace = traceback.format_exc()

                retries -= 1
                print(f"An error occurred: {e}. {stack_trace}. Left retries: {retries}.")
        return None

    return wrapper