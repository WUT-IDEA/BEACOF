import json
import sys

import asyncio
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.types import ModelPlatformType
from interaction import determine_specialties
from ollama import Ollama
from tqdm import tqdm
from utils import extract_json_from_text, remove_think_tags
from datetime import datetime
import os

round_limit = 4
model_name = "gemma3:12b"
temperature = 0
max_tokens = 4096
util_llm: Ollama = Ollama(model_name=model_name)

real_chat_turns = 8


def load_data(scenario, retry_list):
    data_path = "/data/"
    if scenario == "court":
        data = []
        with open(data_path + "criminal_cases.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(retry_list) == 0:
            return data
        return [case for case in data if case["case_id"] in retry_list]
    elif scenario == "medqa":
        data = []
        with open(data_path + "medqa_mc_test.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(retry_list) == 0:
            return data
        return [q for q in data if q["qid"] in retry_list]
    elif scenario == "persona":
        data = []
        with open(
            data_path + "Synthetic_persona_chat_test.json", "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        if len(retry_list) == 0:
            return data
        return [p for p in data if p["id"] in retry_list]
    else:
        raise ValueError("Unknown scenario")


def run_court_debate(case_data):
    # case_data = case_data[:1]
    result_dir = f"results/CAMEL/court_debate/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, "interaction_result.jsonl")

    error_list = []

    for case in tqdm(case_data):
        print(f"Simulating case ID: {case['case_id']}")

        try:
            result = {
                "id": case["case_id"],
                "interaction_history": [],
                "token": {"interaction_token": []},
            }
            case_info = f"案件名: {case.get('case_name', '')}\n原告：{case.get('plaintiff', '')}\n被告：{case.get('defendant', '')}\n被告相关信息：{case.get('defendant_info','')}\n原告诉称：{case.get('plaintiff_claim','')}\n被告辩称：{case.get('defendant_claim','')}\n案件基本事实: {case.get('facts', '')}\n核心争议点: {case.get('key_issues', '')}"

            model = ModelFactory.create(
                model_platform=ModelPlatformType.OLLAMA,
                model_type=model_name,
                model_config_dict={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            plaintiff_agent = ChatAgent(
                system_message=f"现在正在进行一场法庭庭审辩论，你是本案的原告。\n案件基本信息如下：\n {case_info}\n请你作为原告进行发言。",
                model=model,
            )

            defendant_agent = ChatAgent(
                system_message=f"现在正在进行一场法庭庭审辩论，你是本案的被告。\n案件基本信息如下：\n{case_info}\n请你作为被告进行发言。",
                model=model,
            )

            print("Starting court debate...")

            for turn in range(round_limit):
                print(f"--- Turn {turn + 1} ---")
                if turn == 0:
                    msg = "请你作为原告进行发言。"

                plaintiff_response = plaintiff_agent.step(msg)
                # if turn == 0:
                #     print(
                #         f"Message structure:\n{type(plaintiff_response.info)} {plaintiff_response.info}\n\n"
                #     )
                print(f"原告\n: {plaintiff_response.msg.content}\n\n")
                result["interaction_history"].append(
                    {
                        "role": "原告",
                        "content": plaintiff_response.msg.content,
                        "round": turn + 1,
                    }
                )
                result["token"]["interaction_token"].append(
                    {
                        "role": "原告",
                        "token": plaintiff_response.info["usage"]["completion_tokens"],
                    }
                )

                defendant_response = defendant_agent.step(plaintiff_response.msg)
                print(f"被告\n: {defendant_response.msg.content}\n\n")
                result["interaction_history"].append(
                    {
                        "role": "被告",
                        "content": defendant_response.msg.content,
                        "round": turn + 1,
                    }
                )
                result["token"]["interaction_token"].append(
                    {
                        "role": "被告",
                        "token": defendant_response.info["usage"]["completion_tokens"],
                    }
                )

                # print(f"memory: {plaintiff_agent.memory.get_context()}\n\n{defendant_agent.memory.get_context()}\n\n")

                msg = defendant_response.msg.content

                if (
                    "CAMEL_TASK_DONE" in plaintiff_response.msg.content
                    or "CAMEL_TASK_DONE" in defendant_response.msg.content
                ):
                    break
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error in court debate for id {case['case_id']}: {e}")
            error_list.append(case["case_id"])
    print(f"Error cases: {error_list}")


async def run_medqa(question_data):
    # question_data = question_data[:1]
    result_dir = (
        f"results/CAMEL/medqa/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, "interaction_result.jsonl")

    error_list = []

    for q_idx, question in enumerate(tqdm(question_data)):
        print(f"Simulating case ID: {question['qid']}", flush=True)
        try:
            result = {
                "id": question.get("qid", ""),
                "specialty": [],
                "interaction_history": [],
                "token": {"interaction_token": []},
            }
            options_str = "\n".join(
                [f"{k}: {v}" for k, v in question["options"].items()]
            )

            # generate two different doctor profiles
            doctor1_name = "Dr. Elaine Chen"
            doctor2_name = "Dr. Marcus Li"
            specialty1, specialty2 = await determine_specialties(
                question["question"], util_llm
            )
            result["specialty"] = [specialty1, specialty2]

            print(
                f"======Determined specialties======\n {specialty1}, {specialty2}\n",
                flush=True,
            )

            question_info = f"Question: {question['question']}\nOptions:\n{options_str}"

            model = ModelFactory.create(
                model_platform=ModelPlatformType.OLLAMA,
                model_type=model_name,
                model_config_dict={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            doctor1_agent = ChatAgent(
                system_message=f"""You are {doctor1_name}, a {specialty1} doctor. Discuss the medical question and choose the correct answer in a debate format.

Your response MUST end with a JSON object like: {{"answer": "<option>", "reason": "<your reasoning>"}}. When you need to use quotation marks inside the content of the reason field, use single quotes (') instead of double quotes (").

Output examples:
{{"answer": "A", "reason": "After considering the patient's symptoms and medical history, option A is the most appropriate because..."}}
""",
                model=model,
            )

            doctor2_agent = ChatAgent(
                system_message=f"""You are {doctor2_name}, a {specialty2} doctor. Discuss the medical question and choose the correct answer in a debate format.

Your response MUST end with a JSON object like: {{"answer": "<option>", "reason": "<your reasoning>"}}. When you need to use quotation marks inside the content of the reason field, use single quotes (') instead of double quotes (").

Output examples:
{{"answer": "A", "reason": "After considering the patient's symptoms and medical history, option A is the most appropriate because..."}}
""",
                model=model,
            )

            print("Starting medical QA discussion...", flush=True)

            for turn in range(round_limit):
                print(f"\n--- Turn {turn + 1} ---\n", flush=True)
                if turn == 0:
                    msg = f'{question_info}\n\nLet\'s discuss the medical question and choose the correct answer. Your response MUST end with a JSON object like: {{"answer": "<option>", "reason": "<your reasoning>"}}'

                doctor1_response = doctor1_agent.step(msg)
                print(
                    f"{doctor1_name}:\n {doctor1_response.msg.content}\n\n", flush=True
                )
                res1 = extract_json_from_text(doctor1_response.msg.content)

                result["interaction_history"].append(
                    {
                        "round": turn + 1,
                        "role": doctor1_name,
                        "answer": res1["answer"],
                        "content": res1["reason"],
                    }
                )
                result["token"]["interaction_token"].append(
                    {
                        "role": doctor1_name,
                        "token": doctor1_response.info["usage"]["completion_tokens"],
                    }
                )

                doctor2_response = doctor2_agent.step(doctor1_response.msg)
                print(
                    f"{doctor2_name}:\n {doctor2_response.msg.content}\n\n", flush=True
                )
                res2 = extract_json_from_text(doctor2_response.msg.content)

                result["interaction_history"].append(
                    {
                        "round": turn + 1,
                        "role": doctor2_name,
                        "answer": res2["answer"],
                        "content": res2["reason"],
                    }
                )
                result["token"]["interaction_token"].append(
                    {
                        "role": doctor2_name,
                        "token": doctor2_response.info["usage"]["completion_tokens"],
                    }
                )
                msg = f'{doctor2_response.msg.content} Your response MUST end with a JSON object like: {{"answer": "<option>", "reason": "<your reasoning>"}}'

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            print(
                f"Error in medical QA discussion for id {question.get('qid', '')}: {e}",
                flush=True,
            )
            error_list.append(question.get("qid", ""))
    print(f"Error cases: {error_list}")


def run_persona(persona_data):
    # persona_data = persona_data[:1]
    result_dir = (
        f"results/CAMEL/persona/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, "interaction_result.jsonl")

    error_list = []

    for p_idx, persona in enumerate(tqdm(persona_data)):
        print(f"Simulating case ID: {persona['id']}")
        try:
            result = {
                "id": persona.get("id", ""),
                "interaction_history": [],
                "token": {"interaction_token": []},
            }
            person1_name = "Alex"
            person2_name = "Frank"
            user1_persona = "\n".join(persona["user_1_persona"])
            user2_persona = "\n".join(persona["user_2_persona"])
            dialogue_history = "\n".join(
                [
                    f"{person1_name}: {u}" if i % 2 == 0 else f"{person2_name}: {u}"
                    for i, u in enumerate(persona["utterances"][:real_chat_turns])
                ]
            )

            persona_info = f"{person1_name}'s persona:\n{user1_persona}\n\n{person2_name}'s persona:\n{user2_persona}\n\ndialogue history:\n{dialogue_history}"

            model = ModelFactory.create(
                model_platform=ModelPlatformType.OLLAMA,
                model_type=model_name,
                model_config_dict={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            person1_agent = ChatAgent(
                system_message=f"You are {person1_name}. {user1_persona}. Continue the conversation naturally.",
                model=model,
            )

            person2_agent = ChatAgent(
                system_message=f"You are {person2_name}. {user2_persona}. Continue the conversation naturally.",
                model=model,
            )

            print("Starting persona chat...")
            for turn in range(round_limit):
                print(f"--- Turn {turn + 1} ---")
                if turn == 0:
                    msg = "Let's continue our conversation."
                person1_response = person1_agent.step(msg)
                print(f"{person1_name}: \n{person1_response.msg.content}\n\n")
                result["interaction_history"].append(
                    {
                        "role": person1_name,
                        "content": person1_response.msg.content,
                        "round": turn + 1,
                    }
                )
                result["token"]["interaction_token"].append(
                    {
                        "role": person1_name,
                        "token": person1_response.info["usage"]["completion_tokens"],
                    }
                )

                person2_response = person2_agent.step(person1_response.msg)
                print(f"{person2_name}:\n {person2_response.msg.content}\n\n")
                result["interaction_history"].append(
                    {
                        "role": person2_name,
                        "content": person2_response.msg.content,
                        "round": turn + 1,
                    }
                )
                result["token"]["interaction_token"].append(
                    {
                        "role": person2_name,
                        "token": person2_response.info["usage"]["completion_tokens"],
                    }
                )

                msg = person2_response.msg.content

                if (
                    "CAMEL_TASK_DONE" in person1_response.msg.content
                    or "CAMEL_TASK_DONE" in person2_response.msg.content
                ):
                    break

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error in persona chat for id {persona.get('id', '')}: {e}")
            error_list.append(persona.get("id", ""))
    print(f"Error cases: {error_list}")


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "court"

    retry_list: list = [62]

    data = load_data(scenario, retry_list)

    if scenario == "court":
        run_court_debate(data)
    elif scenario == "medqa":
        asyncio.run(run_medqa(data))
    elif scenario == "persona":
        run_persona(data)
