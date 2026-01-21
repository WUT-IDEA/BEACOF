"""
AgentsCourt法庭辩论主模块
实现原告与被告之间的多轮法庭辩论对话，生成庭审记录

主要功能：
1. 解析案件信息，设置原告和被告角色
2. 执行多轮法庭辩论对话
3. 记录和保存庭审过程
4. 处理异常情况和重试机制

Authors: AgentsCourt Team
Date: 2024
"""

import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colorama import Fore
from role.role_playing import RolePlaying
from role.utils import print_text_animated, remove_think_tags
# from Self_Feedback.task_init import TaskInit
# from Self_Feedback.feedback import Feedback
# from Self_Feedback.task_iterate import TaskIterate
from datetime import datetime

import json
import copy
from tqdm import tqdm
import time 

def court_interact(Task, chat_turn_limit):
    """
    执行法庭辩论交互过程
    
    Args:
        Task (dict): 案件信息字典，包含案件详情、当事人信息等
        chat_turn_limit (int): 最大对话轮数限制
        
    Returns:
        list: 庭审记录列表，包含原告和被告的发言记录
        
    主要流程：
    1. 解析案件信息和当事人身份
    2. 构建角色扮演会话
    3. 执行多轮对话交互
    4. 收集和格式化庭审记录
    """
    # ===== 1. Parse basic case information =====
    case_dict = Task
    case_name = case_dict["案件名"]
    
    # Build case description template
    CASE_TEMPLATE = """基本案情: {case_details}\n案件类型: {case_type}"""
    task_prompt = CASE_TEMPLATE.format(
        case_details=case_dict["基本案情"], 
        case_type=case_dict["类别"]+"案件"
    )
    
    # Extract party information
    case_plaintiff = case_dict["原告（公诉）"]
    case_defendant = case_dict["被告"]
    case_defendant_detail = case_dict["被告基本情况"]
    
    # ===== 2. Set role defense prompts =====
    # Plaintiff's claim
    plaintiff_prompt = case_dict['原告诉请判令（公诉机关指控）']    # Defendant's defense: Prioritize lawyer's defense, then defendant's statement
    if case_dict['被告代理人辩护'] == '无' and case_dict['被告陈述'] == '无':
        defendant_prompt = '无'
    elif case_dict['被告代理人辩护'] != '无' and case_dict['被告陈述'] != '无':
        defendant_prompt = case_dict['被告代理人辩护']
    elif case_dict['被告代理人辩护'] != '无' and case_dict['被告陈述'] == '无':
        defendant_prompt = case_dict['被告代理人辩护']
    elif case_dict['被告代理人辩护'] == '无' and case_dict['被告陈述'] != '无':
        defendant_prompt = case_dict['被告陈述']
        
    # ===== 3. Initialize role-playing session =====
    role_play_session = RolePlaying(
        ('plaintiff', plaintiff_prompt),    # Plaintiff role and claim
        ('defendant', defendant_prompt),    # Defendant role and defense points
        case_plaintiff=case_plaintiff,
        case_defendant=case_defendant,
        case_defendant_detail=case_defendant_detail,
        task_prompt=task_prompt,
        #with_task_specify=True,
    )

    #print(Fore.YELLOW + f"\n当前案号: {case_name}\n{task_prompt}\n\nInteractive simulating...\n")
    
    # ===== 4. Execute court debate dialogue =====
    n = 0  # Current dialogue round counter
    assistant_msg, _ = role_play_session.init_chat()  # Initialize dialogue
    # print(Fore.GREEN + f"Defendant:\n\n{assistant_msg.content}\n")

    court_record = []  # Court record container
    record_token = [] # Speech token record
    while n < chat_turn_limit:
        n += 1
        # Execute one round of dialogue: Plaintiff speaks -> Defendant responds
        assistant_return, user_return = role_play_session.step(assistant_msg)
        
        # Parse return results
        assistant_msg, assistant_terminated, assistant_info = assistant_return
        user_msg, user_terminated, user_info = user_return
        
        # ===== 5. Display and record dialogue content =====
        # user_msg.content = remove_think_tags(user_msg.content)
        # assistant_msg.content = remove_think_tags(assistant_msg.content)
        
        # Console display dialogue process (with color distinction)
        print_text_animated(Fore.BLUE + f"Plaintiff:\n\n{user_msg.content}\n")
        print_text_animated(Fore.GREEN + f"Defendant:\n\n{assistant_msg.content}\n")
        
        # Extract speech content
        user_content = user_msg.content
        assistant_content = assistant_msg.content
        
        # Record valid speeches that match the format
        # if user_content.startswith('[原告控诉]') or user_content.startswith('[原告发言]'):
        court_record.append(user_content)
        record_token.append(user_info['usage']['cord_token'])
        
        # if assistant_content.startswith('[被告辩解]'):
        court_record.append(assistant_content)
        record_token.append(assistant_info['usage']['cord_token'])

    return court_record, record_token


def auto_run(output_path, test_list=[], round_num=4):
    with open("./data/criminal_cases.json","r") as file:
        data_all = json.load(file)
    fail_id = []  # Failed case ID record
    court_record_dict = {}  # Court record dictionary
    court_record_token = {} # Court token consumption
    
    if len(test_list) > 0:
        data_all = [x for x in data_all if x['案号'] in test_list]

    # ===== 批量处理案件 =====
    for case_dict in tqdm(data_all):
        try:
            case_num = case_dict["案号"]
            court_record = []
            iter_num = 1
            
            # ===== Retry mechanism: Ensure sufficient court records are generated =====
            while len(court_record) < (2 * round_num - 1) and iter_num < round_num:
                print('Number of iter: {}'.format(iter_num))
                court_record, record_token = court_interact(Task=case_dict, chat_turn_limit=round_num)
                court_record_dict[case_num] = court_record
                court_record_token[case_num] = record_token
                iter_num += 1
            
            print('Number of Court records: {}'.format(len(court_record)))
            #print('{} Saved!'.format(case_num))
        except Exception as e:
            case_num = case_dict["案号"]
            fail_id.append(case_num)
            print(f"处理案件 {case_num} 时发生错误: {e}")
    
    if len(test_list) != 0:
        with open(f"{output_path}/court_records.json", "r", encoding='utf-8') as file:
            data_record = json.load(file)
        with open(f"{output_path}/court_record_tokens.json", "r", encoding='utf-8') as file:
            data_record_token = json.load(file)

        for key, item in court_record_dict.items():
            if key not in data_record or len(data_record[key]) == 0:
                data_record[key] = item
                
        for key, item in court_record_token.items():
            if key not in data_record_token or len(data_record_token[key]) == 0:
                data_record_token[key] = item

        with open(f"{output_path}/court_records.json", "w", encoding='utf-8') as file:
            json.dump(data_record, file, ensure_ascii=False, indent=4)
        with open(f"{output_path}/court_record_tokens.json", "w", encoding='utf-8') as file:
            json.dump(data_record_token, file, ensure_ascii=False, indent=4)
    else:
        # ===== 保存庭审记录 =====
        with open(f"{output_path}/court_records.json", "w", encoding='utf-8') as file:
            json.dump(court_record_dict, file, ensure_ascii=False, indent=4)
        with open(f"{output_path}/court_record_tokens.json", "w", encoding='utf-8') as file:
            json.dump(court_record_token, file, ensure_ascii=False, indent=4)
    
    print(f"处理完成！成功: {len(court_record_dict)}, 失败: {len(fail_id)}")
    if fail_id:
        print(f"失败案件ID: {fail_id}")


if __name__ == "__main__":
    """
    主程序入口：批量处理案件数据，生成庭审记录
    
    处理流程：
    1. 加载案件数据集
    2. 对每个案件执行法庭辩论模拟
    3. 重试机制确保生成足够的庭审记录
    4. 保存所有庭审记录到JSON文件
    """

    # ===== Load case data =====
    with open("./data/criminal_cases.json","r") as file:
        data_all = json.load(file)
    fail_id = []  # Failed case ID record
    court_record_dict = (
        {}
    )  # Court record dictionary
    court_record_token = {} # Court token consumption
    result_dir = (
        f'results/4-rounds-qwen3-30b-retest2/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(result_dir, exist_ok=True)
    test_list = []

    max_turns = 4

    if len(test_list) > 0:
        data_all = [x for x in data_all if x['案号'] in test_list]

    # ===== 批量处理案件 =====
    for case_dict in tqdm(data_all):
        try:
            case_num = case_dict["案号"]
            court_record = []
            iter_num = 1

            # ===== 重试机制：确保生成足够的庭审记录 =====
            while len(court_record) < max_turns * 2 - 1 and iter_num < max_turns:
                print('Number of iter: {}'.format(iter_num))
                court_record, record_token = court_interact(Task=case_dict, chat_turn_limit=max_turns)
                court_record_dict[case_num] = court_record
                court_record_token[case_num] = record_token
                iter_num += 1

            print('Number of Court records: {}'.format(len(court_record)))
            # print('{} Saved!'.format(case_num))
        except Exception as e:
            case_num = case_dict["案号"]
            fail_id.append(case_num)
            print(f"处理案件 {case_num} 时发生错误: {e}")

    # if len(test_list) != 0:
    #     with open("court_records.json", "r", encoding='utf-8') as file:
    #         data_record = json.load(file)
    #     with open("court_record_tokens.json", "r", encoding='utf-8') as file:
    #         data_record_token = json.load(file)

    #     for key, item in court_record_dict.items():
    #         if key not in data_record or len(data_record[key]) == 0:
    #             data_record[key] = item

    #     for key, item in court_record_token.items():
    #         if key not in data_record_token or len(data_record_token[key]) == 0:
    #             data_record_token[key] = item

    #     with open("court_records.json", "w", encoding='utf-8') as file:
    #         json.dump(data_record, file, ensure_ascii=False, indent=4)
    #     with open("court_record_tokens.json", "w", encoding='utf-8') as file:
    #         json.dump(data_record_token, file, ensure_ascii=False, indent=4)
    # else:
    # ===== 保存庭审记录 =====
    with open(f"{result_dir}/court_records.json", "w", encoding='utf-8') as file:
        json.dump(court_record_dict, file, ensure_ascii=False, indent=4)
    with open(f"{result_dir}/court_record_tokens.json", "w", encoding='utf-8') as file:
        json.dump(court_record_token, file, ensure_ascii=False, indent=4)

    print(f"处理完成！成功: {len(court_record_dict)}, 失败: {len(fail_id)}")
    if fail_id:
        print(f"失败案件ID: {fail_id}")
