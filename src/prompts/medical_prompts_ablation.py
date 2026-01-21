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
    prompt of generating interaction content for Medical Q&A
    """
    if scenario == ScenarioType.MEDICAL_QA:

        payoff: str = ""
        for name, payoff_dict in context["payoff"].items():
            if name == role:
                payoff += f"Your estimated strategy payoffs for this round:\n"
                for s, p in payoff_dict.items():
                    payoff += f"- {s}: {p}\n"

        action_predict: str = ""
        for name, probs in context["action_predict"].items():
            if name != role:
                action_predict += (
                    f"\n{name}'s probabilities of choosing each strategy:\n"
                )
                action_predict += "\n".join([f"- {s}: {p}" for s, p in probs.items()])

        belief_info: str = ""
        for name, b in belief.items():
            if name != role:
                belief_info += f"\nBeliefs about {name}:\n"
                belief_info += "\n".join([f"- {k}: {v}" for k, v in b.belief.items()])

        strategy_instruction = (
            f'Analyze all currently known information and strictly adopt "{forced_strategy}" strategy'
            if forced_strategy
            else "Analyze all currently known information, choose your strategy"
        )

        return f"""You are {role}, {profile}. It is now your turn to speak. Please provide your analysis based on all available information in a way that aligns with your role!

# Background Information
{context['info']}

# Known Information
- Current round: {context['current_round']}, Total rounds: {context['max_rounds']}
- Your beliefs and confidence levels regarding other doctors (beliefs are your subjective assessments of other doctors' capabilities and status across various dimensions, with values ranging from 0-1; confidence represents your degree of certainty in your own judgment, with values ranging from 0-1):
{belief_info}
{payoff}
- Probability estimates of other doctors' strategic choices in this round:
{action_predict}

## Output
{strategy_instruction}, and provide your statement. Please output strictly in the following JSON format. In the content, "" cannot be used; when quotation is needed, '' should be used to indicate the quotation.:
{{
    "strategy": "{forced_strategy if forced_strategy else "chosen strategy"}",
    "answer": "chosen answer",
    "content": "statement content"
}}

Statement:"""


def get_payoff_prompt(
    scenario: ScenarioType,
    participants: list["LLMParticipant"],
    context: dict[str, Any],
) -> str:
    if scenario == ScenarioType.MEDICAL_QA:
        if len(context["history"]) == 0:
            history = "None"
        else:
            history = "\n".join(
                [
                    f"{h.role} chose {h.strategy} strategy to speak: {h.content}"
                    for h in context["history"]
                ]
            )
        return f"""A {scenario.value} is currently underway. Please, based on all the information currently available, design a payoff estimate for each participant that aligns with their respective roles, representing the payoffs each participant would receive when choosing different strategies during their statements.

# Strategy Definitions
## Cooperation
Refers to the actions taken by participants that are beneficial to the collective or the other party's interests, even if this may require sacrificing their own immediate personal interests in the short term. The goal is to achieve a win-win outcome or the collective optimal result. Examples include actively listening, providing accurate information, seeking consensus, and avoiding confrontational remarks.

## Competition
Refers to the actions taken by participants that aim to maximize their own relative advantages or payoffs, usually at the expense of the other party's interests. The goal is to pursue individual success rather than consensus. Examples include attempting to persuade or overwhelm the other party, highlighting their weaknesses, and competing for the right to speak.

## Coopetition
Refers to a situation where participants cooperate in certain aspects to achieve common goals or create greater value, while competing in other aspects to secure the payoffs generated from such cooperation. Examples include exchanging information while competing for decision-making authority.

# Background Information
{context['info']}

# Known Information
- Participants: {', '.join([p.role for p in participants])}
- Current interaction round: {context['current_round']}, Total rounds: {context['max_rounds']}
- Interaction history: {history}

# Output
Analyze all the currently known information, calculate the current payoff estimate (with payoff values being float ranging from 0 to 10, with two decimal places retained) for each participant, and output it in the following JSON format:

{{
    "Participant A's name": {{
        "Cooperation": payoff_value,
        "Competition": payoff_value,
        "Coopetition": payoff_value
    }},
    "Participant B's name": {{
        "Cooperation": payoff_value,
        "Competition": payoff_value,
        "Coopetition": payoff_value
    }}
}}

Payoffs:"""


def get_action_predict_prompt(
    scenario: ScenarioType,
    participants: list["LLMParticipant"],
    context: dict[str, Any],
) -> str:
    if scenario == ScenarioType.MEDICAL_QA:
        if len(context["history"]) == 0:
            history = "None"
        else:
            history = "\n".join(
                [
                    f"{h.role} chose {h.strategy} strategy to speak: {h.content}"
                    for h in context["history"]
                ]
            )

        payoff = ""
        for name, payoff_n in context["payoff"].items():
            payoff += f"{name}'s estimated strategy payoffs for this round:\n"
            payoff += "\n".join([f"- {s}: {p}" for s, p in payoff_n.items()])

        return f"""A {scenario.value} is currently taking place. Based on all currently known information, please predict the possible strategy choices each participant may make in the next round!

# Background Information
{context['info']}

# Known Information
- Participants' Names: {', '.join([p.role for p in participants])}
- Next interaction round: {context['current_round']}, Total rounds: {context['max_rounds']}
- Estimated payoffs for each participant's strategies in the next interaction round:
{payoff}
- Interaction history:
{history}

# Output
Analyze all currently known information and predict the probability of each participant choosing each strategy in the next round (probabilities are floating-point numbers ranging from 0 to 1, rounded to two decimal places, and the sum of probability values for each participant's strategies must equal 1!). Output in the following JSON format:

{{
    "Participant A's name": {{
        "Cooperation": probability_value,
        "Competition": probability_value,
        "Coopetition": probability_value
    }},
    "Participant B's name": {{
        "Cooperation": probability_value,
        "Competition": probability_value,
        "Coopetition": probability_value
    }}
}}

# Output Examples
{{
    "Alex": {{
        "Cooperation": 0.60,
        "Competition": 0.25,
        "Coopetition": 0.15 
    }},
    "Frank": {{
        "Cooperation": 0.30,
        "Competition": 0.50,
        "Coopetition": 0.20
    }}
}}

Strategy probabilities for each participant:"""


def get_evaluate_speech_prompt(
    scenario: ScenarioType, role: str, content: str, context: dict[str, Any]
) -> str:
    if scenario == ScenarioType.MEDICAL_QA:
        return f"""You are a professional {scenario.value} analysis expert. Please analyze the following statement content and evaluate the speaker's performance across multiple dimensions.

## Background Information
{context['info']}

## Statement Information
- Speaker Role: {role}
- Statement Content: {content}
- Current Round: {context['current_round']}, Total Rounds: {context['max_rounds']}

## Analysis Dimensions
Please analyze this statement from the following dimensions and provide numerical assessments on a scale of 0-1:

1. evidence_strength (Evidence Strength): The persuasiveness and reliability of evidence presented in the statement
2. medical_knowledge (Medical Knowledge): The depth and accuracy of medical knowledge demonstrated
3. credibility (Credibility): The overall credibility and professionalism of the speaker
4. diagnostic_accuracy (Diagnostic Accuracy): The accuracy of diagnostic reasoning and conclusions
5. consensus_prob (Consensus Probability): The likelihood of reaching consensus based on this statement

## Confidence Assessment
Additionally, please assess your confidence level (0-1) for each dimensional analysis.

## Output Format
Please output strictly according to the following JSON format:

{{
    "belief": {{
        "evidence_strength": score_value,
        "medical_knowledge": score_value,
        "credibility": score_value,
        "diagnostic_accuracy": score_value,
        "consensus_prob": score_value
    }},
    "confidence": {{
        "evidence_strength": confidence_level,
        "medical_knowledge": confidence_level,
        "credibility": confidence_level,
        "diagnostic_accuracy": confidence_level,
        "consensus_prob": confidence_level
    }}
}}
Assessment Results:"""


def get_mad_prompts():
    """MAD prompts for medical QA debate"""
    return {
        "player_meta_prompt": """You are a medical expert participating in a debate to determine the correct answer to a medical question.
Provide accurate, evidence-based responses. Consider the question, options, and previous arguments.""",
        "moderator_meta_prompt": """You are a judge evaluating medical debate arguments. Assess the quality of answers and decide the best one.
You must always output a valid JSON with 'answer' (A, B, C, D, or E) and 'Reason', even if uncertain—choose the most likely option.""",
        "affirmative_prompt": """Question: {question}
Options: {options}
As the primary doctor (affirmative side), provide your initial answer and reasoning.""",
        "negative_prompt": """You are the negative side in a medical diagnosis debate. The affirmative proposed: {aff_ans}. Propose an alternative diagnosis from the options ({options}) and argue why it's more likely than the affirmative's proposal.""",
        "debate_prompt": """Opponent's answer: {oppo_ans}
Respond to this argument by providing your current answer (A, B, C, D, or E) and detailed reasoning, defending or refining your position based on the discussion so far.""",
        "moderator_prompt": """You are a judge evaluating the debate. Affirmative: {aff_ans} Negative: {neg_ans} Based on the discussion in this round, decide the most likely diagnosis (A, B, C, D, or E). You must always provide an 'answer'—choose the best option even if it's a tie or uncertain. Output JSON: {{"answer": "A", "Reason": "reason"}}.""",
    }
