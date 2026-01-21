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
    prompt of generating interaction content for Persona Chat
    """
    if scenario == ScenarioType.PERSONA_CHAT:

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

        return f"""You are {role}, {profile}. It is now your turn to speak. Please continue the conversation based on all available information in a way that aligns with your persona!

# Background Information
{context['info']}

# Known Information
- Current round: {context['current_round']}, Total rounds: {context['max_rounds']}
- Your beliefs and confidence levels regarding the other persona (beliefs are your subjective assessments of the other persona's personality and communication style across various dimensions, with values ranging from 0-1; confidence represents your degree of certainty in your own judgment, with values ranging from 0-1):
{belief_info}
{payoff}
- Probability estimates of the other persona's strategic choices in this round:
{action_predict}

## Output
{strategy_instruction}, and provide your response. Please output strictly in the following JSON format:
{{
    "strategy": "{forced_strategy if forced_strategy else "chosen strategy"}",
    "content": "response content"
}}

## Output Examples
{{
    "strategy": "Cooperation",
    "content": "I really appreciate your perspective on this matter. It's important that we find common ground and work together towards a solution that benefits us both."
}}

Response:"""


def get_payoff_prompt(
    scenario: ScenarioType,
    participants: list["LLMParticipant"],
    context: dict[str, Any],
) -> str:
    if scenario == ScenarioType.PERSONA_CHAT:
        if len(context["history"]) == 0:
            history = "None"
        else:
            history = "\n".join(
                [
                    f"{h.role} chose {h.strategy} strategy to speak: {h.content}"
                    for h in context["history"]
                ]
            )
        return f"""A {scenario.value} is currently underway. Please, based on all the information currently available, design a payoff estimate for each participant that aligns with their respective personas, representing the payoffs each participant would receive when choosing different strategies during their responses.

# Strategy Definitions
## Cooperation
Refers to the actions taken by participants that are beneficial to the collective or the other party's interests, even if this may require sacrificing their own immediate personal interests in the short term. The goal is to achieve a win-win outcome or the collective optimal result. Examples include actively listening, providing supportive information, seeking mutual understanding, and avoiding confrontational remarks.

## Competition
Refers to the actions taken by participants that aim to maximize their own relative advantages or payoffs, usually at the expense of the other party's interests. The goal is to pursue individual success rather than consensus. Examples include attempting to persuade or overwhelm the other party, highlighting their weaknesses, and competing for conversational dominance.

## Coopetition
Refers to a situation where participants cooperate in certain aspects to achieve common goals or create greater value, while competing in other aspects to secure the payoffs generated from such cooperation. Examples include sharing personal experiences while competing for emotional validation.

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

# Output Examples:
{{
    "Frank": {{
        "Cooperation": 8.20,
        "Competition": 1.50,
        "Coopetition": 5.00
    }},
    "Alex": {{
        "Cooperation": 8.50,
        "Competition": 1.20,
        "Coopetition": 4.80
    }}
}}

Payoffs:"""


def get_action_predict_prompt(
    scenario: ScenarioType,
    participants: list["LLMParticipant"],
    context: dict[str, Any],
) -> str:
    if scenario == ScenarioType.PERSONA_CHAT:
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

# Output Examples:
{{
    "Frank": {{
        "Cooperation": 0.70,
        "Competition": 0.15,
        "Coopetition": 0.15
    }},
    "Alex": {{
        "Cooperation": 0.60,
        "Competition": 0.20,
        "Coopetition": 0.20
    }}
}}

Strategy probabilities for each participant:"""


def get_evaluate_speech_prompt(
    scenario: ScenarioType, role: str, content: str, context: dict[str, Any]
) -> str:
    if scenario == ScenarioType.PERSONA_CHAT:
        return f"""You are a professional {scenario.value} analysis expert. Please analyze the following response content and evaluate the speaker's performance across multiple dimensions.

## Background Information
{context['info']}

## Response Information
- Speaker Role: {role}
- Response Content: {content}
- Current Round: {context['current_round']}, Total Rounds: {context['max_rounds']}

## Analysis Dimensions
Please analyze this response from the following dimensions and provide numerical assessments on a scale of 0-1:

1. personality_consistency (Personality Consistency): How well the response aligns with the speaker's defined persona
2. communication_skill (Communication Skill): The effectiveness and clarity of the communication
3. emotional_intelligence (Emotional Intelligence): The awareness and management of emotions in the interaction
4. engagement_level (Engagement Level): The level of active participation and interest shown
5. relationship_building (Relationship Building): The ability to foster positive interpersonal connections

## Confidence Assessment
Additionally, please assess your confidence level (0-1) for each dimensional analysis.

## Output Format
Please output strictly according to the following JSON format:

{{
    "belief": {{
        "personality_consistency": score_value,
        "communication_skill": score_value,
        "emotional_intelligence": score_value,
        "engagement_level": score_value,
        "relationship_building": score_value
    }},
    "confidence": {{
        "personality_consistency": confidence_level,
        "communication_skill": confidence_level,
        "emotional_intelligence": confidence_level,
        "engagement_level": confidence_level,
        "relationship_building": confidence_level
    }}
}}
Assessment Results:"""


def get_mad_prompts_persona():
    """MAD prompts for persona chat debate"""
    return {
        "player_meta_prompt": """You are a persona participating in a debate to continue or refine a conversation.
Provide natural, persona-aligned responses. Consider the conversation history and personas.""",
        "moderator_meta_prompt": """You are a judge evaluating persona-based conversation debates. Assess the quality of responses and decide the best continuation.
You must always output a valid JSON with 'debate_answer' (the best continuation text) and 'Reason', even if uncertain—provide a reasonable continuation.""",
        "affirmative_prompt": """Conversation history: {history}
Personas: {personas}
As the first persona (affirmative side), provide your initial response to continue the conversation.""",
        "negative_prompt": """The first persona said: {aff_ans}
As the second persona (negative side), provide a counter-response or alternative continuation.""",
        "debate_prompt": """Opponent's response: {oppo_ans}
Respond to this, defending or refining your continuation.""",
        "moderator_prompt": """Affirmative: {aff_ans}
Negative: {neg_ans}
Round: {round}
Evaluate the responses and decide the best continuation. You must always provide a 'debate_answer'—choose or create the best response text even if it's a compromise. Output JSON: {{"debate_answer": "continuation text", "Reason": "reason"}}.""",
        "judge_prompt_last1": """Affirmative: {aff_ans}
Negative: {neg_ans}
Extract possible continuation candidates from the debate.""",
        "judge_prompt_last2": """Based on the debate, select the best continuation from the candidates. Output JSON: {{"debate_answer": "continuation", "Reason": "reason"}}.""",
    }
