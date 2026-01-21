"""
AgentsCourt角色扮演模块
实现原告与被告之间的智能体对话交互系统

主要功能：
1. 管理多智能体角色扮演会话
2. 协调原告和被告的轮流发言
3. 处理消息格式转换和历史记录
4. 控制对话终止条件

Authors: AgentsCourt Team
Date: 2024
"""

from typing import Dict, List, Optional, Tuple
from role.generators import SystemMessageGenerator
from role.messages import AssistantChatMessage, ChatMessage, UserChatMessage
from role.typeC import ModelType, RoleType
from role.chat_agent import ChatAgent
import os
import requests
import time
import re


def llm(prompt):
    """
    使用Ollama API调用qwen3:30b-a3b模型
    
    Args:
        prompt (str): 输入提示词
        
    Returns:
        str: LLM生成的回复
    """
    import sys
    import os
    # Add AgentsCourt directory to Python path to import ollama_utils
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from ollama_utils import ollama_simple_completion
    return ollama_simple_completion(prompt)


class RolePlaying:
    """
    角色扮演会话管理器
    
    管理原告(user_agent)和被告(assistant_agent)之间的对话交互，
    包括系统消息生成、智能体初始化和对话状态管理。
    """
    
    def __init__(
        self,
        user_role,
        assistant_role,
        case_plaintiff,
        case_defendant,
        case_defendant_detail,
        task_prompt= ""
    ) -> None:
        """
        初始化角色扮演会话
        
        Args:
            user_role (tuple): 用户角色信息 (角色名, 角色提示词)
            assistant_role (tuple): 助手角色信息 (角色名, 角色提示词)
            case_plaintiff (str): 原告名称
            case_defendant (str): 被告名称  
            case_defendant_detail (str): 被告详细信息
            task_prompt (str): 任务描述提示词
        """
        
        # ===== Parse role information =====
        self.task_prompt = task_prompt 
        user_role_name = user_role[0]           # 'plaintiff'
        assistant_role_name = assistant_role[0] # 'defendant'
        plaintiff_prompt = user_role[1]         # Plaintiff's claim content
        defendant_prompt = assistant_role[1]    # Defendant's defense content

        # ===== Generate system messages =====
        sys_msg_generator = SystemMessageGenerator()
        
        # Build template replacement dictionary
        sys_msg_meta_dicts = [{
            "<ASSISTANT_ROLE>": assistant_role_name,
            "<USER_ROLE>": user_role_name,
            "<PLAINTIFF>": plaintiff_prompt,
            "<DEFENDANT>": defendant_prompt,
            "<PLAINTIFF_NAME>": case_plaintiff,
            "<DEFENDANT_NAME>": case_defendant,
            "<DEFENDANT_DETAIL>": case_defendant_detail,
            "<TASK>": task_prompt,
        }] * 2  # Generate the same metadata for two agents
        
        # Generate system messages
        self.assistant_sys_msg, self.user_sys_msg = (
            sys_msg_generator.from_dicts(
                meta_dicts=sys_msg_meta_dicts,
                role_tuples=[
                    (assistant_role_name, RoleType.ASSISTANT),  # Defendant agent
                    (user_role_name, RoleType.USER),            # Plaintiff agent
                ],
            ))

        # ===== Initialize agents =====
        self.assistant_agent = ChatAgent(
            self.assistant_sys_msg  # Defendant agent
        )
        self.user_agent = ChatAgent(
            self.user_sys_msg      # Plaintiff agent
        )

    def init_chat(self):
        """
        初始化对话会话
        
        重置两个智能体的状态，并生成初始消息启动对话。
        
        Returns:
            tuple: (assistant_msg, msgs) 初始助手消息和消息列表
        """
        
        # Reset agent states
        self.assistant_agent.reset()
        self.user_agent.reset()

        # Create initial messages
        assistant_msg = AssistantChatMessage(
            role_name=self.assistant_sys_msg.role_name,
            content=self.task_prompt)

        user_msg = UserChatMessage(
            role_name=self.user_sys_msg.role_name,
            content=self.task_prompt)
        
        # Let assistant agent generate first response
        msgs, _, _ = self.assistant_agent.step(user_msg) 

        return assistant_msg, msgs

    def process_messages(
        self,
        messages: List[ChatMessage],
    ) -> ChatMessage:
        """
        处理消息列表，确保只有一条消息
        
        Args:
            messages (List[ChatMessage]): 输入消息列表
            
        Returns:
            ChatMessage: 处理后的单条消息
            
        Raises:
            ValueError: 当消息列表为空或包含多条消息时抛出异常
        """
        if len(messages) == 0:
            raise ValueError("No messages to process.")
        if len(messages) > 1 :
            raise ValueError("Got more than one message to process. "
                             f"Num of messages: {len(messages)}.")
        else:
            processed_msg = messages[0]

        return processed_msg

    def step(
        self,
        assistant_msg: ChatMessage
    ):
        """
        执行一轮双智能体对话交互
        
        这是核心的对话调度函数，实现原告和被告的轮流发言：
        1. 让原告智能体对助手消息进行响应
        2. 检查原告是否要终止对话
        3. 让被告智能体对原告消息进行响应  
        4. 检查被告是否要终止对话
        5. 更新双方的消息历史
        
        Args:
            assistant_msg (ChatMessage): 被告(助手)的输入消息
            
        Returns:
            tuple: ((assistant_msg, assistant_terminated, assistant_info),
                   (user_msg, user_terminated, user_info))
                   返回两个元组分别代表被告和原告的状态
        """

        # ===== First stage: Plaintiff response =====
        user_msgs, user_terminated, user_info = self.user_agent.step(
            assistant_msg.to_user_chat_message())

        # Check if plaintiff terminates dialogue
        if user_terminated:
            return ((None, None, None), (None, user_terminated, user_info))
        
        # Process plaintiff message and update history
        user_msg = self.process_messages(user_msgs)
        self.user_agent.update_messages(user_msg)

        # ===== Second stage: Defendant response =====
        (assistant_msgs, assistant_terminated,
         assistant_info) = self.assistant_agent.step(
             user_msg.to_user_chat_message())
             
        # Check if defendant terminates dialogue
        if assistant_terminated:
            return ((None, assistant_terminated, assistant_info),
                    (user_msg, user_terminated, user_info))
                    
        # Process defendant message and update history
        assistant_msg = self.process_messages(assistant_msgs)
        self.assistant_agent.update_messages(assistant_msg)
        
        # ===== Return both parties' status =====
        return (
            (assistant_msg, assistant_terminated, assistant_info),
            (user_msg, user_terminated, user_info)
        )
