# -*- coding: utf-8 -*-
"""
AgentsCourt Ollama API Tool Module
Unified management of all large language model calls in the project, migrated from OpenAI GPT to Ollama qwen3:30b-a3b

Main Functions:
1. Encapsulate Ollama API call interface
2. Support chat and simple completion modes
3. Provide error handling and retry mechanism
4. Unified output format processing
5. Compatible with original OpenAI interface call method

Authors: AgentsCourt Team
Date: 2024
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import time
from typing import List, Dict, Any, Optional

from role.utils import remove_think_tags
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Debate large model
model_name = config["model_name"]


class OllamaClient:
    """Ollama client class for calling locally deployed qwen3:30b-a3b model"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = model_name):
        self.base_url = base_url
        self.model = model
        self.chat_endpoint = f"{base_url}/api/chat"
        self.version_endpoint = f"{base_url}/api/version"
    
    def check_health(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(self.version_endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_message: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Call Ollama chat completion API
        
        Args:
            messages: Message list, format [{"role": "user/assistant", "content": "..."}]
            system_message: System message (optional)
            stream: Whether to use streaming output
            timeout: Request timeout
            
        Returns:
            Model generated reply content
        """
        if system_message:
            # If system message is provided, add to the beginning of message list
            formatted_messages = [{"role": "system", "content": system_message}] + messages
        else:
            formatted_messages = messages
        
        # formatted_messages[-1]["content"] = messages[-1]["content"] + "/no_think"
        print(f"调用模型：{self.model}\n")
        data = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": stream,
            "options": {
                "temperature": config['temperature'],
                "num_predict": 4096
            }
        }
        
        try:
            response = requests.post(self.chat_endpoint, json=data)
            response.raise_for_status()
            result = response.json()
            # print(result["message"]["content"].strip())
            return remove_think_tags(result["message"]["content"].strip()), result["eval_count"]
        except Exception as e:
            raise Exception(f"调用Ollama API失败: {e}")
    
    def simple_completion(self, prompt: str, system_message: str = None) -> str:
        """
        Simple text completion interface, compatible with original GPT_4 function signature
        
        Args:
            prompt: User input prompt
            system_message: System message
            
        Returns:
            Model generated reply
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, system_message)
    


# Global Ollama client instance
_ollama_client = OllamaClient()


def ollama_chat_completion(
    messages: List[Dict[str, str]], 
    system_message: Optional[str] = None,
    delay: float = 0
) -> str:
    """
    Global Ollama chat completion function
    
    Args:
        messages: Message list
        system_message: System message
        delay: Delay time before call (seconds)
        
    Returns:
        Model reply content
    """
    if delay > 0:
        time.sleep(delay)
    
    return _ollama_client.chat_completion(messages, system_message)


def ollama_simple_completion(prompt: str, system_message: str = None, delay: float = 0) -> str:
    """
    Simple Ollama completion function, for replacing original GPT_4 function
    
    Args:
        prompt: User input
        system_message: System message  
        delay: Delay time before call (seconds)
        
    Returns:
        Model reply content
    """
    if delay > 0:
        time.sleep(delay)
        
    return _ollama_client.simple_completion(prompt, system_message)


def check_ollama_health() -> bool:
    """Check Ollama service health status"""
    return _ollama_client.check_health()


def get_ollama_model_info() -> str:
    """Get current model information"""
    return _ollama_client.model


# Backward compatible function alias
def GPT_4(system_message: str, prompt: str) -> str:
    """
    Ollama call function compatible with original GPT_4 function interface
    Maintain same function signature as original code for seamless migration
    """
    return ollama_simple_completion(prompt, system_message)


def llm(prompt: str) -> str:
    """
    Ollama call function compatible with original llm function interface
    """
    return ollama_simple_completion(prompt)


if __name__ == "__main__":
    # Test Ollama connection
    print("测试Ollama连接...")
    if check_ollama_health():
        print("✓ Ollama服务运行正常")
        print(f"✓ 当前模型: {get_ollama_model_info()}")
        
        # Simple test
        try:
            response = ollama_simple_completion("你好，请简单介绍一下你自己。")
            print(f"✓ 测试成功，模型回复: {response[:100]}...")
        except Exception as e:
            print(f"✗ 测试失败: {e}")
    else:
        print("✗ 无法连接到Ollama服务，请检查:")
        print("  1. Ollama服务是否已启动")
        print("  2. 服务地址是否正确 (默认: http://localhost:11434)")
        print("  3. qwen3:30b-a3b模型是否已下载")