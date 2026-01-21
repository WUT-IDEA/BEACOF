import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from tqdm import tqdm

def GPT_4(prompt):
    """Use Ollama API to call qwen3:30b-a3b model"""
    from ollama_utils import ollama_simple_completion
    return ollama_simple_completion(prompt, '你是一个专业的法律助手')

def auto_extract_judge_result(output_path, test_list):
    with open(f"{output_path}/record.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    judge_result_new = {}
    new_error_list = []

    # Process erroneous records
    if len(test_list) > 0:
        for error_case_id in tqdm(test_list):
            # Try re-extraction once
            with open(f"{output_path}/judge_result.json", "r", encoding="utf-8") as f:
                old_data = json.load(f)
            if old_data.get(error_case_id) is None:
                print(f"{error_case_id}处理中...")
                template_path = "system_prompt/crimial_extract_prompt.txt"
                with open(template_path, "r", encoding='utf-8') as f:
                    template = f.read()
                    
                prompt_form = """抽取结果:<result_initial>
                请将以上内容整理为以下格式: 
                抽取结果:{"审判依据":["《中华人民共和国刑法》第一百三十三条之一第一款第(二)项","《中华人民共和国刑法》第六十七条第一款", "《中华人民共和国刑法》第七十二条第一款", "《中华人民共和国刑法》第七十二条第三款"，"《中华人民共和国刑法》第七十三条第一款"，"《中华人民共和国刑法》第七十三条第三款"], "审判结果":{"罪名":"被告人金某某犯危险驾驶罪","刑期":"判处拘役二个月，缓刑二个月","罚金":"罚金人民币四千元"}}
                抽取结果:
                """
                
                prompt_extract = template.replace("<case_detail>", data[error_case_id])
                print("Extracting...")
                try:
                    result_initial, _ = GPT_4(prompt_extract)
                    print(result_initial)
                    dictionary = json.loads(result_initial)
                    print("Right form. Next...")
                    old_data[error_case_id] =  dictionary
                except json.JSONDecodeError:
                    try:
                        print("Revising...")
                        prompt_revise = prompt_form.replace("<result_initial>", result_initial)
                        #print("prompt_revise:", prompt_revise)
                        
                        result_reform, _ = GPT_4(prompt_revise)
                        print("result_reform:", result_reform)
                    
                        dictionary = json.loads(result_reform)
                        print("Revisd!")
                        old_data[error_case_id] =  dictionary

                    except json.JSONDecodeError:
                        # If conversion fails, throw an error
                        new_error_list.append(error_case_id)
                        print("Error {}".format(error_case_id))
                with open(f"{output_path}/judge_result.json", "w", encoding="utf-8") as f:
                    json.dump(old_data, f, ensure_ascii=False, indent=4)
    else:   
        for case_id, judge_result in tqdm(data.items()):
            template_path = "system_prompt/crimial_extract_prompt.txt"
            with open(template_path, "r", encoding='utf-8') as f:
                template = f.read()
                
            prompt_form = """抽取结果:<result_initial>
            请将以上内容整理为以下格式: 
            抽取结果:{"审判依据":["《中华人民共和国刑法》第一百三十三条之一第一款第(二)项","《中华人民共和国刑法》第六十七条第一款", "《中华人民共和国刑法》第七十二条第一款", "《中华人民共和国刑法》第七十二条第三款"，"《中华人民共和国刑法》第七十三条第一款"，"《中华人民共和国刑法》第七十三条第三款"], "审判结果":{"罪名":"被告人金某某犯危险驾驶罪","刑期":"判处拘役一年二个月，缓刑二个月","罚金":"罚金人民币四千元"}}
            抽取结果:
            """
            
            prompt_extract = template.replace("<case_detail>", judge_result)
            print("Extracting...")
            try:
                result_initial, _ = GPT_4(prompt_extract)
                dictionary = json.loads(result_initial)
                print("Right form. Next...")
                judge_result_new[case_id] =  dictionary
            except json.JSONDecodeError:
                try:
                    print("Revising...")
                    prompt_revise = prompt_form.replace("<result_initial>", result_initial)
                    #print("prompt_revise:", prompt_revise)
                    
                    result_reform, _ = GPT_4(prompt_revise)
                    print("result_reform:", result_reform)
                
                    dictionary = json.loads(result_reform)
                    print("Revisd!")
                    judge_result_new[case_id] =  dictionary

                except json.JSONDecodeError:
                    # If conversion fails, throw an error
                    new_error_list.append(case_id)
                    print("判决结果提取重复失败： {}".format(case_id))

            with open(f"{output_path}/judge_result.json", "w", encoding="utf-8") as f:
                json.dump(judge_result_new, f, ensure_ascii=False, indent=4)

    print("失败案件ID：", new_error_list)

if __name__ == "__main__":
    file_path = (
        "results/4-rounds-qwen3-30b/retest-2/record.json"
    )
    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    current_dir = os.path.dirname(file_path)

    judge_result_new = {}
    error_case_number = ['(2021)苏0509刑初1665号']
    new_error_list = []

    # Process erroneous records
    if len(error_case_number) > 0:
        for error_case_id in tqdm(error_case_number):
            # Try re-extraction once
            with open(
                f"{current_dir}/judge_result.json",
                "r",
                encoding="utf-8",
            ) as f:
                old_data = json.load(f)
            if old_data.get(error_case_id) is None:
                print(f"Error_case_id{error_case_id}再次处理...")
                template_path = "system_prompt/crimial_extract_prompt.txt"
                with open(template_path, "r", encoding='utf-8') as f:
                    template = f.read()

                prompt_form = """抽取结果:<result_initial>
                请将以上内容整理为以下格式: 
                抽取结果:{"审判依据":["《中华人民共和国刑法》第一百三十三条之一第一款第(二)项","《中华人民共和国刑法》第六十七条第一款", "《中华人民共和国刑法》第七十二条第一款", "《中华人民共和国刑法》第七十二条第三款"，"《中华人民共和国刑法》第七十三条第一款"，"《中华人民共和国刑法》第七十三条第三款"], "审判结果":{"罪名":"被告人金某某犯危险驾驶罪","刑期":"判处拘役二个月，缓刑二个月","罚金":"罚金人民币四千元"}}
                抽取结果:
                """

                prompt_extract = template.replace("<case_detail>", data[error_case_id])
                print("Extracting...")
                try:
                    result_initial, _ = GPT_4(prompt_extract)
                    print(f"Extracted result: {result_initial}")
                    dictionary = json.loads(result_initial)
                    print("Right form. Next...")
                    old_data[error_case_id] =  dictionary
                except json.JSONDecodeError:
                    try:
                        print("Revising...")
                        prompt_revise = prompt_form.replace("<result_initial>", result_initial)
                        # print("prompt_revise:", prompt_revise)

                        result_reform, _ = GPT_4(prompt_revise)
                        print("result_reform:", result_reform)

                        dictionary = json.loads(result_reform)
                        print("Revisd!")
                        old_data[error_case_id] =  dictionary

                    except json.JSONDecodeError:
                        # If conversion fails, throw an error
                        new_error_list.append(error_case_id)
                        print("Error {}".format(error_case_id))
                with open(
                    f"{current_dir}/judge_result.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(old_data, f, ensure_ascii=False, indent=4)
    else:   
        for case_id, judge_result in tqdm(data.items()):
            template_path = "system_prompt/crimial_extract_prompt.txt"
            with open(template_path, "r", encoding='utf-8') as f:
                template = f.read()

            prompt_form = """抽取结果:<result_initial>
            请将以上内容整理为以下格式: 
            抽取结果:{"审判依据":["《中华人民共和国刑法》第一百三十三条之一第一款第(二)项","《中华人民共和国刑法》第六十七条第一款", "《中华人民共和国刑法》第七十二条第一款", "《中华人民共和国刑法》第七十二条第三款"，"《中华人民共和国刑法》第七十三条第一款"，"《中华人民共和国刑法》第七十三条第三款"], "审判结果":{"罪名":"被告人金某某犯危险驾驶罪","刑期":"判处拘役一年二个月，缓刑二个月","罚金":"罚金人民币四千元", "sentence": 14, "probation": 2, "fine": 4000}}
            抽取结果:
            """

            prompt_extract = template.replace("<case_detail>", judge_result)
            print("Extracting...")
            try:
                result_initial, _ = GPT_4(prompt_extract)
                dictionary = json.loads(result_initial)
                print("Right form. Next...")
                judge_result_new[case_id] =  dictionary
            except json.JSONDecodeError:
                try:
                    print("Revising...")
                    prompt_revise = prompt_form.replace("<result_initial>", result_initial)
                    # print("prompt_revise:", prompt_revise)

                    result_reform, _ = GPT_4(prompt_revise)
                    print("result_reform:", result_reform)

                    dictionary = json.loads(result_reform)
                    print("Revisd!")
                    judge_result_new[case_id] =  dictionary

                except json.JSONDecodeError:
                    # If conversion fails, throw an error
                    new_error_list.append(case_id)
                    print("Error {}".format(case_id))

            with open(f"{current_dir}/judge_result.json", "w", encoding="utf-8") as f:
                json.dump(judge_result_new, f, ensure_ascii=False, indent=4)

    print(new_error_list)
