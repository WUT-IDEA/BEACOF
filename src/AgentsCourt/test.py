import bisect
import json, requests
from typing import Any, Dict

from tqdm import tqdm

from role.utils import remove_think_tags

def calculate_token(
    prompt: str,
    model: str = "deepseek-r1:32b",
    base_url: str = "http://localhost:11434",
    ) -> int:
    # Calculate token count corresponding to text
        url = f"{base_url}/api/embed"
        data = {
            "model": model,
            "input": prompt,
        }

        response = requests.post(url, json=data)
        # response.raise_for_status()
        response = response.json()
        return response["prompt_eval_count"]

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON content from text"""
    try:
        # Try to find JSON code block
        import re

        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            return json.loads(matches[0])

        # Try to parse the entire text directly
        return json.loads(text)

    except Exception as e:
        print(f"JSON extraction failed: {e}")
        return {}

def llm_api(model:str, prompt:str) -> str:
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    response = requests.post(url, json=data, stream=True)
    response.raise_for_status()

    full_response = ""
    final_context = {}

    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))

            # Compatible with streaming fields of /api/chat and /api/generate
            # /api/chat common: { message: { role, content }, done: false }
            # /api/generate common: { response: "...", done: false }
            content_piece = ""
            if isinstance(chunk.get("message"), dict):
                content_piece = chunk["message"].get("content", "")
            elif "response" in chunk:
                content_piece = chunk.get("response", "")
            elif "delta" in chunk:
                content_piece = chunk.get("delta", "")

            if content_piece:
                full_response += content_piece
                print(content_piece, end="", flush=True)

            if chunk.get("done", False):
                final_context = chunk
                break
        except json.JSONDecodeError:
            print(f"Unparseable JSON line: {line}")
    
    return remove_think_tags(full_response.strip())

def get_range(value, bounds):
    """返回value所在的区间 (lower, upper)"""
    idx = bisect.bisect_right(bounds, value)
    lower = bounds[idx-1]
    upper = bounds[idx] if idx < len(bounds) else float('inf')
    return (lower, upper)

# Compare if two values are in the same range
def in_same_range(val1, val2, bounds):
    return get_range(val1, bounds) == get_range(val2, bounds)


# Merge legal data
def merge_laws():
    all_laws = {}

    pahts = ["./data/laws_1.json", "./data/laws_2.json", "./data/laws_3.json"]

    for path in pahts:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(len(data))
            for key, value in data.items():
                if key not in all_laws:
                    all_laws[key] = value

    print(len(all_laws))
    with open("./data/all_laws.json", "w", encoding="utf-8") as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=4)

# Statistics token consumption - no think
def statistics_total(output_path):
    with open(f"{output_path}/court_records.json", "r", encoding='utf-8') as f:
        debate_data = json.load(f)
    with open(f"{output_path}/record.json", "r", encoding='utf-8') as f:
        judge_data = json.load(f)
        
    debate_token_count = []
    judge_token_count = []
    print(len(debate_data))
    print(len(judge_data))
    for case_id, item in tqdm(debate_data.items()):
        for text in item:
            debate_token_count.append(calculate_token(prompt=text))

    for case_id, item in tqdm(judge_data.items()):
        judge_token_count.append(calculate_token(prompt=item))
    print(f"不含think")
    print(f"平均发言token消耗：{round(sum(debate_token_count) / len(debate_token_count), 3)}")
    print(f"平均判决token消耗：{round(sum(judge_token_count) / len(judge_token_count), 3)}")
    
    with open(f"{output_path}/token_no_thinking.txt", "w", encoding="utf-8") as f:
        f.write(f"平均发言token消耗：{round(sum(debate_token_count) / len(debate_token_count), 3)}\n")
        f.write(f"平均判决token消耗：{round(sum(judge_token_count) / len(judge_token_count), 3)}\n")

# All output tokens
def cal_speech_token_thinkg(output_path):
    with open(f"{output_path}/court_record_tokens.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(f"{output_path}/record_token.json", "r", encoding="utf-8") as f:
        data_judge = json.load(f)
    
    assert len(data) == len(data_judge)
    avg_token_s = []
    for idx, tokens in data.items():
        avg_token_s.append(sum(tokens) / len(tokens))
    print(f"发言平均token消耗-thinking：{round(sum(avg_token_s) / len(avg_token_s), 3)}")
    
    token_j = 0
    for idx, token in data_judge.items():
        token_j += token
    print(f"平均判决token消耗-thinking：{round(token_j / len(data_judge), 3)}")
    
    with open(f"{output_path}/token_thinking.txt", "w", encoding="utf-8") as f:
        f.write(f"发言平均token消耗-thinking：{round(sum(avg_token_s) / len(avg_token_s), 3)}")
        f.write(f"平均判决token消耗-thinking：{round(token_j / len(data_judge), 3)}")

# Convert judge result values
def cal_judge_num(res_file_path, save_path):
    with open(res_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for single_res in tqdm(data.keys()):
        judge_res = data[single_res]['审判结果']
        text = f"""请你将下列数据中涉及数值部分的内容整理为数字格式
        刑期整理为以月为单位的数字，罚金以元为单元的数字，当刑期中存在缓刑时，单独计算。

        将整理结果以JSON格式输出，不要输出任何其他无关的内容！
        
        参考示例：
        原内容：
        刑期：判处拘役二个月
        罚金：处罚金人民币四千元
        参考输出：
        {{
        "sentence": 2,
        "probation": 0,
        "fine": 4000
        }}

        参考示例：
        原内容：
        刑期：判处有期徒刑六个月，缓刑一年
        罚金：处罚金人民币三万元
        参考输出：
        {{
        "sentence": 6,
        "probation": 12,
        "fine": 30000
        }}

        参考示例：
        原内容：
        刑期：判处有期徒刑一年三个月
        罚金：处罚金人民币三万六千元
        参考输出：
        {{
        "sentence": 15,
        "probation": 0,
        "fine": 36000
        }}

        原内容：
        刑期：{judge_res['刑期']}
        罚金：{judge_res['罚金'] if '罚金' in judge_res else '无'}
        输出：
        {{
        "sentence": ...,
        "probation": ...,
        "fine": ...        
        }}
        """

        res = llm_api("qwen3:30b-a3b", prompt=text)
        res = extract_json_from_text(res)

        data[single_res]['审判结果']['sentence'] = res['sentence']
        data[single_res]['审判结果']['probation'] = res['probation']
        data[single_res]['审判结果']['fine'] = res['fine']

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# cal_judge_num()

# Calculate judgment metrics
def cal_judge_metric(output_path):
    # with open(f"{output_path}/judge_result_num.json", "r", encoding="utf-8") as f:
    with open(output_path, "r", encoding="utf-8") as f:
        judge_res = json.load(f)
    with open("./data/processed_cases.json", "r", encoding="utf-8") as f:
        gold_judge = json.load(f)
    with open("./data/criminal_cases.json", "r", encoding='utf-8') as f:
        criminal_cases = json.load(f)

    fine_bound = [0, 1001, 5001, 10001, 50001, 100001, 500001, 1000001]
    sentence_bound = [0, 7, 13, 37, 61, 121]
    probation_bound = [0, 13, 25, 61]

    sentence = []
    fine = []
    criminal_name = 0
    for case_id, single_res in judge_res.items():
        case_id = [x['案件名'] + x['原告（公诉）'] for x in criminal_cases if x['案号'] == case_id][0]
        cur_ref = [x for x in gold_judge if x['case_name'] + x['plaintiff'] == case_id][0]

        # Calculate charge
        pre_name_index = cur_ref["charge"].find('犯')
        pred_name = cur_ref["charge"][pre_name_index:]

        gold_name_index = single_res["审判结果"]["罪名"].find('犯')
        gold_name = single_res["审判结果"]["罪名"][gold_name_index:]

        if pred_name == gold_name:
            criminal_name += 1

        # Calculate sentence
        if cur_ref['probation'] == 0:
            if single_res['审判结果']['probation'] == 0:
                if in_same_range(cur_ref['sentence_num'], single_res['审判结果']['sentence'], sentence_bound):
                    sentence.append(1)
                else:
                    sentence.append(0)
            else:
                sentence.append(0)
        else:
            if in_same_range(cur_ref['sentence_num'], single_res['审判结果']['sentence'], sentence_bound) and in_same_range(cur_ref['probation'], single_res['审判结果']['probation'], probation_bound):
                sentence.append(1)
            else:
                sentence.append(0)

        # Calculate fine
        if cur_ref['fine_num'] == 0:
            if single_res['审判结果']['fine'] == 0:
                fine.append(1)
            else:
                fine.append(0)
        else:
            if in_same_range(cur_ref['fine_num'], single_res['审判结果']['fine'], fine_bound):
                fine.append(1)
            else:
                fine.append(0)

    print(f"罪名准确率：{round(criminal_name / len(judge_res), 4)}")
    print(f"刑期准确率：{round(sum(sentence) / len(sentence), 4)}")
    print(f"罚金准确率：{round(sum(fine) / len(fine), 4)}")

    # with open(f"{output_path}/criminal_evaluation_result.txt", "w", encoding="utf-8") as f:
    #     f.write(f'Charge accuracy: {round(criminal_name / len(judge_res), 4)}\n')
    #     f.write(f'Sentence accuracy: {round(sum(sentence) / len(sentence), 4)}\n')
    #     f.write(f'Fine accuracy: {round(sum(fine) / len(fine), 4)}\n')


if __name__ == "__main__":
    # cal_judge_metric()
    # cal_judge_num(
    #     "results/4-rounds-qwen3-30b/retest-2/judge_result.json",
    #     "results/4-rounds-qwen3-30b/retest-2/judge_result_num.json",
    # )
    cal_judge_metric(
        "results/4-rounds-qwen3-30b/retest-2/judge_result_num.json"
    )
    # statistics_total()
    # cal_speech_token_thinkg()
    # res = []
    # with open("data/processed_cases.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # for item in data:
    #     res.append(item['raw_data'])

    # with open("data/criminal_cases.json", "w", encoding="utf-8") as f:
    #     json.dump(res, f, ensure_ascii=False, indent=4)
