from typing import Any, TYPE_CHECKING
from belief import IBelief
from base import ScenarioType

if TYPE_CHECKING:
    from role import LLMParticipant


def speech_prompt(
    scenario: ScenarioType,
    role: str,
    belief: dict[str, IBelief],
    profile: str,
    context: dict[str, Any],
    forced_strategy: str = None,
) -> str:
    """
    prompt of generating interaction content for Court Debate
    """
    if scenario == ScenarioType.COURT_DEBATE:

        payoff: str = ""
        for name, payoff_dict in context["payoff"].items():
            if name == role:
                payoff += f"你本回合策略收益估计：\n"
                for s, p in payoff_dict.items():
                    payoff += f"- {s}: {p}\n"

        action_predict: str = ""
        for name, probs in context["action_predict"].items():
            if name != role:
                action_predict += f"\n{name}选择各策略的概率：\n"
                action_predict += "\n".join([f"- {s}: {p}" for s, p in probs.items()])

        belief_info: str = ""
        for name, b in belief.items():
            if name != role:
                belief_info += f"\n关于{name}的信念：\n"
                belief_info += "\n".join([f"- {k}: {v}" for k, v in b.belief.items()])

        strategy_instruction = (
            f'分析当前所有已知信息，并严格采用"{forced_strategy}"策略'
            if forced_strategy
            else "分析当前所有已知信息，选择你的策略"
        )

        return f"""你是{role}，{profile}。现在轮到你发言了。请基于所有可用信息，以符合你角色的方式提供你的分析！

# Background Information
{context['info']}

# Known Information
- 当前回合：{context['current_round']}，总回合数：{context['max_rounds']}
- 你对他方律师的信念和置信度水平（信念是你对他方律师的可信度和论点在各个维度的主观评估，数值范围为0-1；置信度表示你对自己判断的确信程度，数值范围为0-1）：
{belief_info}
{payoff}
- 他方律师在本回合策略选择的概率估计：
{action_predict}

## Output
{strategy_instruction}，并提供你的陈述，在陈述内容中不要换行，以整段话的形式表达。请严格按照以下JSON格式输出：
{{
    "strategy": "{forced_strategy if forced_strategy else "选择的策略"}",
    "content": "陈述内容"
}}

陈述："""


def get_payoff_prompt(
    scenario: ScenarioType,
    participants: list["LLMParticipant"],
    context: dict[str, Any],
) -> str:
    if scenario == ScenarioType.COURT_DEBATE:
        if len(context["history"]) == 0:
            history = "无"
        else:
            history = "\n".join(
                [
                    f"{h.role}选择{h.strategy}策略进行发言：{h.content}"
                    for h in context["history"]
                ]
            )
        return f"""现在正在进行一场{scenario.value}，请你根据当前已知的所有信息，为每个参与者设计一个符合各自立场的收益估计，表示各参与者在发言时选择各策略对应的收益。

# Strategy Definition
## Cooperation
指参与者选择有利于集体或对方利益的行动，即使这可能在短期内牺牲个人的直接利益，目标是实现双赢或集体最优的结果。例如积极倾听、提供真实信息、寻求共识、避免攻击性言辞等。

## Competition
指参与者选择旨在最大化自身相对优势或收益的行动，通常以牺牲对方利益为代价，目标是追求胜利而非共识。例如试图说服、压倒对方、揭露对方弱点、争夺话语权。

## Coopetition
指参与者在某些方面合作以实现共同目标或创造更大价值，同时在其他方面竞争以争夺由此产生的利益。例如交换信息的同时争夺价值分配。

# Background Information
{context['info']}

# Known Information
- 参与者：{'，'.join([p.role for p in participants])}
- 当前交互回合：{context['current_round']}，总回合数：{context['max_rounds']}
- 交互历史：{history}

# Output
分析当前所有已知信息，为每个参与者计算当前收益估计（收益值为0-10的浮点数，保留2位小数），并以如下JSON格式输出（请严格按照下列格式输出）：
在输出的json中不要包含解释，说明。

{{
    "原告": {{
        "合作": 收益值,
        "竞争": 收益值,
        "合竞": 收益值,
    }},
    "被告": {{
        "合作": 收益值,
        "竞争": 收益值,
        "合竞": 收益值,
    }},
}}

# Output example:
{{
    "原告": {{
        "合作": 6.50,
        "竞争": 8.20,
        "合竞": 7.10,
    }},
    "被告": {{
        "合作": 5.30,
        "竞争": 9.00,
        "合竞": 6.80,
    }}
}}

各参与者收益：
"""
    else:
        return ""


def get_action_predict_prompt(
    scenario: ScenarioType,
    participants: list["LLMParticipant"],
    context: dict[str, Any],
) -> str:
    if scenario == ScenarioType.COURT_DEBATE:
        if len(context["history"]) == 0:
            history = "无"
        else:
            history = "\n".join(
                [
                    f"{h.role}选择{h.strategy}策略进行发言：{h.content}"
                    for h in context["history"]
                ]
            )
        payoff = ""
        for name, payoff_n in context["payoff"].items():
            payoff += f"{name}在本轮的预估策略收益：\n"
            payoff += "\n".join([f"- {s}: {p}" for s, p in payoff_n.items()])
        return f"""现在正在进行一场{scenario.value}，请你根据当前已知的所有信息，预测每个参与者在当前回合选择各策略的概率。

# Strategy Definition
## Cooperation
指参与者选择有利于集体或对方利益的行动，即使这可能在短期内牺牲个人的直接利益，目标是实现双赢或集体最优的结果。例如积极倾听、提供真实信息、寻求共识、避免攻击性言辞等。

## Competition
指参与者选择旨在最大化自身相对优势或收益的行动，通常以牺牲对方利益为代价，目标是追求胜利而非共识。例如试图说服、压倒对方、揭露对方弱点、争夺话语权。

## Coopetition
指参与者在某些方面合作以实现共同目标或创造更大价值，同时在其他方面竞争以争夺由此产生的利益。例如交换信息的同时争夺价值分配。

# Background Information
{context['info']}

# Known Information
- 参与者：{'，'.join([p.role for p in participants])}
- 当前交互回合：{context['current_round']}，总回合数：{context['max_rounds']}
- 下一轮交互中每位参与者策略的预估收益：
{payoff}
- 交互历史：{history}

# Output
分析当前所有已知信息，为每个参与者预测当前回合选择各策略的概率（概率值为0-1的浮点数，保留2位小数，各策略概率之和为1），并以如下JSON格式输出：

{{
    "原告": {{
        "合作": 概率值,
        "竞争": 概率值,
        "合竞": 概率值,
    }},
    "被告": {{
        "合作": 概率值,
        "竞争": 概率值,
        "合竞": 概率值,
    }},
}}

各参与者策略选择概率：
"""
    else:
        return ""


def get_evaluate_speech_prompt(
    scenario: ScenarioType, role: str, content: str, context: dict[str, Any]
) -> str:
    if scenario == ScenarioType.COURT_DEBATE:
        return f"""你是一各专业的{scenario.value}分析专家。请分析一下发言内容，从多个维度评估发言者的表现。

## Background Information
{context['info']}

## Speech Information
- 发言者角色: {role}
- 发言内容: {content}
- 当前回合: {context['current_round']}，总回合数：{context['max_rounds']}

## Analysis Dimensions
请从以下维度分析这次发言，并给出0-1之间的数值评估：

1. evidence_strength(证据强度): 发言中证据的说服力和可靠性
2. legal_position (法律地位): 发言显示的法律论证优势
3. credibility (可信度): 发言者的整体可信度和专业性
4. strategic_competence (策略能力): 发言显示的策略思维和辩论技巧
5. winning_prob (胜诉概率): 基于此次发言对胜诉概率的影响

## Confidence Assessment
同时请评估你对每个维度分析的置信度(0-1)。

## Output Format
请严格按照以下JSON格式输出，不要在json中输出解释，说明等注释内容：
{{
    "belief": {{
        "evidence_strength": 评分值,
        "legal_position": 评分值,
        "credibility": 评分值,
        "strategic_competence": 评分值,
        "winning_prob": 评分值
    }},
    "confidence": {{
        "evidence_strength": 置信度,
        "legal_position": 置信度,
        "credibility": 置信度,
        "strategic_competence": 置信度,
        "winning_prob": 置信度
    }}
}}

评估结果：
"""
    else:
        return ""
