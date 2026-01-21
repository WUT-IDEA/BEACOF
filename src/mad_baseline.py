import json
import sys
import asyncio
from ollama import Ollama
from tqdm import tqdm
from utils import remove_think_tags
from datetime import datetime
import os
from prompts.medical_prompts import get_mad_prompts
from prompts.persona_prompts import get_mad_prompts_persona

round_limit = 4
model_name = "gemma3:12b"
temperature = 0
max_tokens = 4096
real_chat_turns = 8
candidates_per_turn = 3


def load_data(scenario):
    data_path = "data/"
    if scenario == "medqa":
        with open(data_path + "medqa_mc_test.json", "r", encoding="utf-8") as f:
            return json.load(f)
    elif scenario == "persona":
        with open(
            data_path + "Synthetic_persona_chat_test.json", "r", encoding="utf-8"
        ) as f:
            return json.load(f)
    else:
        raise ValueError("Unknown scenario")


class DebatePlayer:
    def __init__(self, model_name, name, temperature, openai_api_key, sleep_time):
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.sleep_time = sleep_time
        self.llm = Ollama(model_name)
        self.memory_lst = []
        self.meta_prompt = ""

    def set_meta_prompt(self, prompt):
        self.meta_prompt = prompt

    def add_event(self, event):
        self.memory_lst.append({"role": "user", "content": event})

    def add_memory(self, memory):
        self.memory_lst.append({"role": "assistant", "content": memory})

    async def ask(self):
        messages = [{"role": "system", "content": self.meta_prompt}] + self.memory_lst
        response, token = await self.llm._chat(messages)
        return response, token

    async def ask_with_event(self, event: str):
        """Ask the LLM with an extra user event without mutating internal memory.
        Returns (response, token).
        """
        messages = (
            [{"role": "system", "content": self.meta_prompt}]
            + self.memory_lst
            + [{"role": "user", "content": event}]
        )
        response, token = await self.llm._chat(messages)
        return response, token


class MultiAgentSimulation:
    def __init__(
        self,
        model_name=model_name,
        temperature=0,
        num_players=3,
        openai_api_key=None,
        config=None,
        max_round=3,
        sleep_time=0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.openai_api_key = openai_api_key
        self.config = config
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.interaction_history = []
        self.interaction_tokens = []
        self.init_prompt()
        self.creat_agents()

    def init_prompt(self):
        raise NotImplementedError

    def creat_agents(self):
        raise NotImplementedError

    async def init_agents(self):
        raise NotImplementedError

    async def run(self):
        raise NotImplementedError

    def print_answer(self):
        raise NotImplementedError


class DebateSimulation(MultiAgentSimulation):
    def init_prompt(self):
        self.config["debate_topic"] = self.config.get("question", "")
        keys = [
            "player_meta_prompt",
            "moderator_meta_prompt",
            "affirmative_prompt",
            "judge_prompt_last2",
        ]
        for key in keys:
            if key in self.config:
                self.config[key] = self.config[key].replace(
                    "##debate_topic##", self.config["debate_topic"]
                )

    def creat_agents(self):
        NAME_LIST = ["Primary Doctor", "Secondary Doctor", "Judge"]
        self.players = [
            DebatePlayer(
                self.model_name,
                name,
                self.temperature,
                self.openai_api_key,
                self.sleep_time,
            )
            for name in NAME_LIST
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    async def init_agents(self):
        self.affirmative.set_meta_prompt(self.config["player_meta_prompt"])
        self.negative.set_meta_prompt(self.config["player_meta_prompt"])
        self.moderator.set_meta_prompt(self.config["moderator_meta_prompt"])

        print("===== Debate Round-1 =====\n")
        aff_prompt = self.config["affirmative_prompt"].format(
            question=self.config["question"], options=self.config["options"]
        )
        self.affirmative.add_event(aff_prompt)
        self.aff_ans, aff_token = await self.affirmative.ask()
        print(f"Affirmative Response: {self.aff_ans}\n")
        self.affirmative.add_memory(self.aff_ans)
        self.config["base_answer"] = self.aff_ans
        self.interaction_history.append(
            {"role": self.affirmative.name, "content": self.aff_ans, "round": 1}
        )
        self.interaction_tokens.append(
            {
                "role": self.affirmative.name,
                "token": (
                    aff_token["eval_count"]
                    if isinstance(aff_token, dict)
                    else aff_token
                ),
            }
        )

        neg_prompt = self.config["negative_prompt"].format(
            aff_ans=self.aff_ans, options=self.config["options"]
        )
        self.negative.add_event(neg_prompt)
        self.neg_ans, neg_token = await self.negative.ask()
        print(f"Negative Response: {self.neg_ans}\n")
        self.negative.add_memory(self.neg_ans)
        self.interaction_history.append(
            {"role": self.negative.name, "content": self.neg_ans, "round": 1}
        )
        self.interaction_tokens.append(
            {
                "role": self.negative.name,
                "token": (
                    neg_token["eval_count"]
                    if isinstance(neg_token, dict)
                    else neg_token
                ),
            }
        )

        mod_prompt = self.config["moderator_prompt"].format(
            aff_ans=self.aff_ans, neg_ans=self.neg_ans, round="first"
        )
        self.moderator.add_event(mod_prompt)
        self.mod_ans, mod_token = await self.moderator.ask()
        print(f"Moderator Result: {self.mod_ans}\n")
        self.moderator.add_memory(self.mod_ans)
        self.interaction_history.append(
            {"role": self.moderator.name, "content": self.mod_ans, "round": 1}
        )
        self.interaction_tokens.append(
            {
                "role": self.moderator.name,
                "token": (
                    mod_token["eval_count"]
                    if isinstance(mod_token, dict)
                    else mod_token
                ),
            }
        )
        try:
            self.mod_ans = json.loads(self.mod_ans)
        except json.JSONDecodeError:
            self.mod_ans = {}

    def round_dct(self, num):
        dct = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
            7: "seventh",
            8: "eighth",
            9: "ninth",
            10: "tenth",
        }
        return dct[num]

    def print_answer(self):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config["debate_topic"])
        print("\n----- Base Answer -----")
        print(self.config["base_answer"])
        print("\n----- Debate Answer -----")
        print(self.config.get("answer", ""))
        print("\n----- Debate Reason -----")
        print(self.config.get("Reason", ""))

    async def run(self):
        # Already debated one round during init
        for round in range(self.max_round - 1):
            print(f"===== Debate Round-{round+2} =====\n")
            aff_debate = self.config["debate_prompt"].format(oppo_ans=self.neg_ans)
            self.affirmative.add_event(aff_debate)
            self.aff_ans, aff_token = await self.affirmative.ask()
            print(f"Affirmative Response: {self.aff_ans}\n")
            self.affirmative.add_memory(self.aff_ans)
            self.interaction_history.append(
                {
                    "role": self.affirmative.name,
                    "content": self.aff_ans,
                    "round": round + 2,
                }
            )
            self.interaction_tokens.append(
                {
                    "role": self.affirmative.name,
                    "token": (
                        aff_token["eval_count"]
                        if isinstance(aff_token, dict)
                        else aff_token
                    ),
                }
            )

            neg_debate = self.config["debate_prompt"].format(oppo_ans=self.aff_ans)
            self.negative.add_event(neg_debate)
            self.neg_ans, neg_token = await self.negative.ask()
            print(f"Negative Response: {self.neg_ans}\n")
            self.negative.add_memory(self.neg_ans)
            self.interaction_history.append(
                {
                    "role": self.negative.name,
                    "content": self.neg_ans,
                    "round": round + 2,
                }
            )
            self.interaction_tokens.append(
                {
                    "role": self.negative.name,
                    "token": (
                        neg_token["eval_count"]
                        if isinstance(neg_token, dict)
                        else neg_token
                    ),
                }
            )

            mod_debate = self.config["moderator_prompt"].format(
                aff_ans=self.aff_ans,
                neg_ans=self.neg_ans,
                round=self.round_dct(round + 2),
            )
            self.moderator.add_event(mod_debate)
            self.mod_ans, mod_token = await self.moderator.ask()
            print(f"Moderator Result: {self.mod_ans}\n")
            self.moderator.add_memory(self.mod_ans)
            self.interaction_history.append(
                {
                    "role": self.moderator.name,
                    "content": self.mod_ans,
                    "round": round + 2,
                }
            )
            self.interaction_tokens.append(
                {
                    "role": self.moderator.name,
                    "token": (
                        mod_token["eval_count"]
                        if isinstance(mod_token, dict)
                        else mod_token
                    ),
                }
            )
            try:
                self.mod_ans = json.loads(self.mod_ans)
            except json.JSONDecodeError:
                self.mod_ans = {}

        if self.mod_ans.get("answer"):
            self.config.update(self.mod_ans)
            self.config["success"] = True
        else:
            self.config["success"] = False
        self.print_answer()


class DialogueSimulation(MultiAgentSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interaction_history = []
        self.interaction_tokens = []
        self.judge_feedback = []
        self.selection_logs = []
        # candidates per turn for MAD-style internal debate
        self.candidates_per_turn = self.config.get("candidates_per_turn", 3)

    def init_prompt(self):
        # For persona chat, no debate_topic replacement needed
        pass

    def creat_agents(self):
        NAME_LIST = ["Character A", "Character B", "Judge"]
        self.players = [
            DebatePlayer(
                self.model_name,
                name,
                self.temperature,
                self.openai_api_key,
                self.sleep_time,
            )
            for name in NAME_LIST
        ]
        self.character_a = self.players[0]
        self.character_b = self.players[1]
        self.judge = self.players[2]

    async def init_agents(self):
        self.character_a.set_meta_prompt(self.config["player_meta_prompt"])
        self.character_b.set_meta_prompt(self.config["player_meta_prompt"])
        self.judge.set_meta_prompt(self.config["moderator_meta_prompt"])

        print("===== Dialogue Round-1 =====\n")
        # Character A starts: generate candidates, select best via Judge, then commit
        aff_event = self.config["affirmative_prompt"].format(
            history="\n".join(self.config["history"]),
            personas=str(self.config["personas"]),
        )
        a_candidates, a_tokens = await self.generate_candidates(
            self.character_a, aff_event
        )
        a_best, a_best_idx, a_reason = await self.select_best(
            speaker=self.character_a.name, candidates=a_candidates, context={"round": 1}
        )
        # Commit chosen event and reply to memory
        self.character_a.add_event(aff_event)
        self.character_a.add_memory(a_best)
        self.a_ans = a_best
        self.interaction_history.append(
            {"role": self.character_a.name, "content": a_best, "round": 1}
        )
        # token log: sum tokens for candidates and add one entry for chosen
        self.interaction_tokens.append(
            {
                "role": self.character_a.name,
                "token": sum(
                    (
                        (t.get("eval_count") if isinstance(t, dict) else t)
                        for t in a_tokens
                    )
                ),
            }
        )
        self.selection_logs.append(
            {
                "round": 1,
                "speaker": self.character_a.name,
                "candidates": a_candidates,
                "selected_index": a_best_idx,
                "reason": a_reason,
            }
        )

        # Character B responds: candidates -> select -> commit
        b_event = self.config["negative_prompt"].format(aff_ans=self.a_ans)
        b_candidates, b_tokens = await self.generate_candidates(
            self.character_b, b_event
        )
        b_best, b_best_idx, b_reason = await self.select_best(
            speaker=self.character_b.name, candidates=b_candidates, context={"round": 1}
        )
        self.character_b.add_event(b_event)
        self.character_b.add_memory(b_best)
        self.b_ans = b_best
        self.interaction_history.append(
            {"role": self.character_b.name, "content": b_best, "round": 1}
        )
        self.interaction_tokens.append(
            {
                "role": self.character_b.name,
                "token": sum(
                    (
                        (t.get("eval_count") if isinstance(t, dict) else t)
                        for t in b_tokens
                    )
                ),
            }
        )
        self.selection_logs.append(
            {
                "round": 1,
                "speaker": self.character_b.name,
                "candidates": b_candidates,
                "selected_index": b_best_idx,
                "reason": b_reason,
            }
        )
        # Round-1 selection uses judge internally; no extra external evaluation to keep flow natural

    def print_answer(self):
        print("\n\n===== Dialogue Done! =====")
        print("\n----- Initial Dialogue History -----")
        print(self.config["history"])
        print("\n----- Full Dialogue (Selected Replies) -----")
        for entry in self.interaction_history:
            print(f"{entry['role']} (Round {entry['round']}): {entry['content']}")
        print("\n----- Selection Logs (MAD-internal debate) -----")
        for s in self.selection_logs:
            print(
                f"Round {s['round']} | {s['speaker']} -> idx {s['selected_index']}: {s['reason']}"
            )

    async def run(self):
        # init already completed round 1
        for round in range(self.max_round - 1):
            print(f"===== Dialogue Round-{round+2} =====\n")
            # A's turn: candidates -> select -> commit
            a_event = self.config["debate_prompt"].format(oppo_ans=self.b_ans)
            a_candidates, a_tokens = await self.generate_candidates(
                self.character_a, a_event
            )
            a_best, a_best_idx, a_reason = await self.select_best(
                speaker=self.character_a.name,
                candidates=a_candidates,
                context={"round": round + 2},
            )
            self.character_a.add_event(a_event)
            self.character_a.add_memory(a_best)
            self.a_ans = a_best
            self.interaction_history.append(
                {"role": self.character_a.name, "content": a_best, "round": round + 2}
            )
            self.interaction_tokens.append(
                {
                    "role": self.character_a.name,
                    "token": sum(
                        (
                            (t.get("eval_count") if isinstance(t, dict) else t)
                            for t in a_tokens
                        )
                    ),
                }
            )
            self.selection_logs.append(
                {
                    "round": round + 2,
                    "speaker": self.character_a.name,
                    "candidates": a_candidates,
                    "selected_index": a_best_idx,
                    "reason": a_reason,
                }
            )

            # B's turn: candidates -> select -> commit
            b_event = self.config["debate_prompt"].format(oppo_ans=self.a_ans)
            b_candidates, b_tokens = await self.generate_candidates(
                self.character_b, b_event
            )
            b_best, b_best_idx, b_reason = await self.select_best(
                speaker=self.character_b.name,
                candidates=b_candidates,
                context={"round": round + 2},
            )
            self.character_b.add_event(b_event)
            self.character_b.add_memory(b_best)
            self.b_ans = b_best
            self.interaction_history.append(
                {"role": self.character_b.name, "content": b_best, "round": round + 2}
            )
            self.interaction_tokens.append(
                {
                    "role": self.character_b.name,
                    "token": sum(
                        (
                            (t.get("eval_count") if isinstance(t, dict) else t)
                            for t in b_tokens
                        )
                    ),
                }
            )
            self.selection_logs.append(
                {
                    "round": round + 2,
                    "speaker": self.character_b.name,
                    "candidates": b_candidates,
                    "selected_index": b_best_idx,
                    "reason": b_reason,
                }
            )

        self.config["success"] = True
        self.print_answer()

    async def generate_candidates(self, player: DebatePlayer, event: str):
        """Generate multiple candidate replies without mutating player's memory.
        Returns (candidates: List[str], tokens: List[Any])
        """
        candidates = []
        tokens = []
        for _ in range(self.candidates_per_turn):
            resp, tok = await player.ask_with_event(event)
            candidates.append(resp)
            tokens.append(tok)
        return candidates, tokens

    async def select_best(self, speaker: str, candidates, context=None):
        """Use Judge to select the best candidate via a MAD-style critique.
        Returns (best_text, best_index, reason)
        """
        if context is None:
            context = {}
        # Build selection prompt
        criteria = (
            "personality_consistency, communication_skill, emotional_intelligence, "
            "engagement_level, relationship_building"
        )
        cands_block = "\n".join([f"{i}. {c}" for i, c in enumerate(candidates)])
        sel_event = (
            f"You are the Judge. Evaluate the following candidate replies for {speaker}.\n"
            f"Criteria: {criteria}.\n"
            "Select the best one that continues the dialogue naturally.\n"
            "Return ONLY a valid JSON object with keys: 'best_index' (integer, 0-based) and 'reason' (string).\n\n"
            f"Candidates:\n{cands_block}\n"
        )
        print(f"DEBUG: sel_event = {sel_event}")
        # Ask judge without mutating its memory
        sel_ans, _ = await self.judge.ask_with_event(sel_event)
        print(f"DEBUG: sel_ans = {sel_ans}")
        # Ask judge without mutating its memory
        sel_ans, _ = await self.judge.ask_with_event(sel_event)
        try:
            parsed = json.loads(sel_ans)
            best_idx = int(parsed.get("best_index", 0))
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Judge selection failed: {e}, raw output: {sel_ans}")
            # Fallback: choose the first non-empty, else 0
            best_idx = next((i for i, c in enumerate(candidates) if c.strip()), 0)
            reason = "fallback_selection"
        best_idx = max(0, min(best_idx, len(candidates) - 1))
        return candidates[best_idx], best_idx, reason

    def round_dct(self, num):
        dct = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
            7: "seventh",
            8: "eighth",
            9: "ninth",
            10: "tenth",
        }
        return dct[num]


async def run_medical_qa(question_data):
    result_dir = f"results/MAD/medical_qa/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, "interaction_result.jsonl")

    error_list = []

    mad_prompts = get_mad_prompts()

    for case in tqdm(question_data):
        try:
            config = {
                "question": case["question"],
                "options": case["options"],
                "answer": case["answer"],
                **mad_prompts,
            }

            debate = DebateSimulation(config=config, max_round=round_limit)
            await debate.init_agents()
            await debate.run()

            result = {
                "id": case.get("qid", ""),
                "interaction_history": debate.interaction_history,
                "token": {"interaction_token": debate.interaction_tokens},
            }

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            error_list.append(case["qid"])
            print(f"Error in case: {e}")

    print(f"Results saved to {file_path}")
    print(f"Error Cases: {error_list}")


async def run_persona_chat(persona_data):
    result_dir = f"results/MAD/persona_chat/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, "interaction_result.jsonl")

    error_list = []

    mad_prompts = get_mad_prompts_persona()

    for case in tqdm(persona_data):
        try:
            config = {
                "history": case["utterances"][
                    :real_chat_turns
                ],  # Assuming utterances is the history
                "personas": {
                    "user_1": case["user_1_persona"],
                    "user_2": case["user_2_persona"],
                },
                "candidates_per_turn": candidates_per_turn,
                **mad_prompts,
            }

            dialogue = DialogueSimulation(config=config, max_round=round_limit)
            await run_dialogue(dialogue)

            result = {
                "id": case.get("id", ""),
                "interaction_history": dialogue.interaction_history,
                "token": {"interaction_token": dialogue.interaction_tokens},
                "judge_feedback": dialogue.judge_feedback,
                "selection_logs": dialogue.selection_logs,
            }

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            error_list.append(case["id"])
            print(f"Error in case: {e}")

    print(f"Results saved to {file_path}")
    print(f"Error Cases: {error_list}")


async def run_dialogue(dialogue):
    await dialogue.init_agents()
    await dialogue.run()


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "medqa"

    data = load_data(scenario)

    retry_list = []

    if scenario == "medqa":
        if len(retry_list) > 0:
            data = [i for i in data if i["qid"] in retry_list]
        asyncio.run(run_medical_qa(data))

    elif scenario == "persona":
        if len(retry_list) > 0:
            data = [i for i in data if i["id"] in retry_list]
        asyncio.run(run_persona_chat(data))
