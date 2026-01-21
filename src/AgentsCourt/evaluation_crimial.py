import json
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
import bisect 

def get_range(value, bounds):
    """Return the range (lower, upper) where value is located"""
    idx = bisect.bisect_right(bounds, value)
    lower = bounds[idx-1]
    upper = bounds[idx] if idx < len(bounds) else float('inf')
    return (lower, upper)

# Compare if two values are in the same range
def in_same_range(val1, val2, bounds):
    return get_range(val1, bounds) == get_range(val2, bounds)


# Gold result of the dataset
gold_judge_path = './data/criminal_cases.json'
with open(gold_judge_path, 'r', encoding='utf-8') as file:
    gold_judge = json.load(file)

# with open("./data/criminal_cases.json", "r", encoding='utf-8') as f:
#     criminal_cases = json.load(f)

# Output result
model_judge_path = "results/4-rounds-llama3-8b/20251030_145627/judge_result_llama3-8b-4-rounds-judge.json"
with open(model_judge_path, 'r', encoding='utf-8') as file:
    model_judge = json.load(file)

total_num = total_name = total_sentence = total_fine = 0

for gold_result in gold_judge:
    number = gold_result['案号']
    # print(number)
    if gold_result["类别"] == '刑事':
        total_num += 1
        pred_result = model_judge[number]
        gold_name = gold_result.get('刑事_罪名')
        gold_name_index = gold_name.find('犯')
        gold_name = gold_name[gold_name_index:]

        gold_sentence = gold_result.get('刑事_刑期')
        gold_fine = gold_result.get('刑事_罚金')
        if "处" in gold_fine:
            gold_fine = gold_fine.replace("处", "")

        pred_name = pred_result["审判结果"].get('罪名')
        pre_name_index = pred_name.find('犯')
        pred_name = pred_name[pre_name_index:]
        pred_sentence = pred_result["审判结果"].get('刑期')
        pred_fine = pred_result["审判结果"].get('罚金')

        if gold_name == pred_name:
            total_name += 1
        else:
            print(f"案号：{number}，标准答案：{gold_name}, 模型答案：{pred_name}")
        if gold_sentence == pred_sentence:
            total_sentence += 1
        if gold_fine == pred_fine:
            total_fine += 1
            print(f"案号：{number}，标准答案：{gold_fine}, 模型答案：{pred_fine}")

total_name_p = total_name/total_num
total_sentence_p = total_sentence/total_num
total_fine_p = total_fine/total_num


print('total_name', total_name_p)
print('total_sentence', total_sentence_p)
print('total_fine', total_fine_p)
