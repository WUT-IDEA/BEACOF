# Automatically process all processes
import json
import os
from role.main_court import auto_run
from judge_analysis_record import auto_judge
from extract_laws_in_judge import auto_extract_judge_result
from evaluation_laws import auto_evaluate_laws
from test import cal_judge_num, cal_judge_metric, cal_speech_token_thinkg, statistics_total
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_name = config["model_name"]
# Error cases - additional run
test_list = []
# Debate rounds
round_num = 8
# Test task name
task = f"{model_name}-temperature{config['temperature']}-{round_num}rounds"
# task = f"4-rounds-temperature0/retest-1"

# Save directory
output_path = f"results/{task}"
# Create directory
os.makedirs(output_path, exist_ok=True)
print(f"目录创建成功:{output_path}")

def get_len(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(len(data))

def main():
    # Debate
    auto_run(output_path, test_list=test_list, round_num=round_num)

    # court_record_path = 'results/4-rounds-temperature0/court_records.json'
    # Judge judgment
    auto_judge(output_path, test_list=test_list)

    # Extract judgment results
    auto_extract_judge_result(output_path, test_list=test_list)

    # Evaluation results - laws
    auto_evaluate_laws(output_path)

    # Judgment results - digitization
    cal_judge_num(output_path)

    # Evaluation results - judgment results
    cal_judge_metric(output_path)

    # Calculate Token
    cal_speech_token_thinkg(output_path)

    # Calculate Token-nothink
    statistics_total(output_path)

if __name__ == "__main__":
    main()