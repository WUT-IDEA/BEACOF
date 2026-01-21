from typing import Dict, Generator, List, Optional, Set, Tuple
from role.messages import SystemMessage, SystemMessageType
import os
from role.typeC import ModelType, RoleType




def get_system_prompt(role):
    # Get the directory of the current file, then construct the path to the AgentsCourt root directory
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    
    if role == 'plaintiff':
        template_path = os.path.join(parent_dir, 'system_prompt', 'plaintiff_message.txt')
        with open(template_path, "r", encoding='utf-8') as f:
            template = f.read()
    else:
        template_path = os.path.join(parent_dir, 'system_prompt', 'defendant_message.txt')
        with open(template_path, "r", encoding='utf-8') as f:
            template = f.read()
    
    return template


class SystemMessageGenerator:
    """System message generator for agents.
    """

    def __init__(self):
        
        user_prompt_template = get_system_prompt(role='plaintiff')
        assistant_prompt_template = get_system_prompt(role='defendant')

        self.sys_prompts = dict()
        self.sys_prompts[RoleType.USER] = user_prompt_template
        self.sys_prompts[RoleType.ASSISTANT] = assistant_prompt_template
        
    def from_dict(
        self,
        meta_dict,
        role_tuple,
    ):
        # assistant_role_name, RoleType.ASSISTANT
        role_name, role_type = role_tuple
        # assistant_prompt_template
        sys_prompt = self.sys_prompts[role_type]
        # 
        sys_prompt = self.replace_keywords(meta_dict, sys_prompt)
        return SystemMessage(role_name=role_name, role_type=role_type,
                             meta_dict=meta_dict, content=sys_prompt)

    def from_dicts(
        self,
        meta_dicts,
        role_tuples,
    ):
        if len(meta_dicts) != len(role_tuples):
            raise ValueError(
                "The number of meta_dicts and role_types should be the same.")

        return [
            self.from_dict(meta_dict, role_tuple)
            for meta_dict, role_tuple in zip(meta_dicts, role_tuples)
        ]

    @staticmethod
    def replace_keywords(
        meta_dict,
        sys_prompt,
    ):
        for key, value in meta_dict.items():
            sys_prompt = sys_prompt.replace(key, value)
        return sys_prompt


