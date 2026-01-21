import yaml
from enum import Enum
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Debate large model
model_name = config["model_name"]

# The values should be the same as the prompt file names
class RoleType(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    DEFAULT = "default"


class ModelType(Enum):
    # Keep original names for compatibility, but actually using Ollama models
    GPT_3_5_TURBO = model_name
    GPT_4 = model_name
    GPT_4_32k = model_name
    GPT_3_5_TURBO_1106 = model_name
    GPT_4_1106_preview = model_name


# The values should be the same as the prompt dir names
class TaskType(Enum):
    AI_SOCIETY = "ai_society"
    CODE = "code"
    MISALIGNMENT = "misalignment"
    TRANSLATION = "translation"
    DEFAULT = "default"


__all__ = ['RoleType', 'ModelType', 'TaskType']
