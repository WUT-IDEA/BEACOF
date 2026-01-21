from base import ScenarioType


def get_system_prompt(scenario: ScenarioType) -> str:
    if scenario == ScenarioType.MEDICAL_QA:
        return f"""A {scenario.value} is currently underway. Please carefully read all information and act in accordance with your role.

# Strategy Definitions
## Cooperation
Participants choose actions that benefit the collective or the other party, even if it means sacrificing immediate personal interests in the short term. The goal is to achieve win-win outcomes or optimal collective results. Examples include actively listening, providing accurate information, seeking consensus, and avoiding aggressive language.

## Competition
Participants choose actions aimed at maximizing their own relative advantages or payoffs, usually at the expense of the other party's interests. The goal is to pursue victory rather than consensus. Examples include attempting to persuade or overwhelm others, exposing weaknesses, and competing for speaking rights.

## Coopetition
Participants cooperate in certain aspects to achieve common goals or create greater value, while competing in other aspects to secure the payoffs generated. Examples include exchanging information while competing for value distribution.

# Belief Explanation
## What are Beliefs?
Beliefs are your subjective assessments of other doctors' capabilities and status across various dimensions, with values ranging from 0-1:
- **0.0-0.3**: You believe other doctors perform poorly or have weak capabilities in this aspect
- **0.4-0.6**: You believe other doctors perform averagely or at a moderate level in this aspect
- **0.7-1.0**: You believe other doctors perform excellently or have strong capabilities in this aspect

## What is Confidence?
Confidence represents your degree of certainty in your own judgment, with values ranging from 0-1:
- **0.0-0.3**: You are very uncertain about your judgment and may change your mind easily
- **0.4-0.6**: You have some confidence in your judgment but still have doubts
- **0.7-1.0**: You are very confident in your judgment and unlikely to change easily

## Belief Dimension Explanations
- **evidence_strength**: The strength and persuasiveness of other doctors' evidence
- **diagnostic_confidence**: The degree of advantage other doctors have in diagnostic positions
- **credibility**: The overall credibility and professionalism of other doctors
- **expertise_competence**: The professional thinking and reasoning skills of other doctors
- **correctness_prob**: The overall probability that other doctors' answers are correct
"""

    elif scenario == ScenarioType.COURT_DEBATE:
        return f"""A {scenario.value} is currently underway. Please carefully read all information and act according to your role.

# Strategy Definition
## Cooperation
Participants choose actions that benefit the collective or the other party, even if it means sacrificing immediate personal interests in the short term. The goal is to achieve win-win outcomes or optimal collective results. Examples include actively listening, providing accurate information, seeking consensus, and avoiding aggressive language.

## Competition
Participants choose actions aimed at maximizing their own relative advantages or payoffs, usually at the expense of the other party's interests. The goal is to pursue victory rather than consensus. Examples include attempting to persuade or overwhelm others, exposing weaknesses, and competing for speaking rights.

## Coopetition
Participants cooperate in certain aspects to achieve common goals or create greater value, while competing in other aspects to secure the payoffs generated. Examples include exchanging information while competing for value distribution.

# Belief Explanation
## What is Belief?
Beliefs are your subjective assessments of the opposing lawyer's capabilities and status across various dimensions, with values ranging from 0-1:
- **0.0-0.3**: You believe the opposing lawyer performs poorly or has weak capabilities in this aspect
- **0.4-0.6**: You believe the opposing lawyer performs averagely or at a moderate level in this aspect
- **0.7-1.0**: You believe the opposing lawyer performs excellently or has strong capabilities in this aspect

## What is Confidence?
Confidence represents your degree of certainty in your own judgment, with values ranging from 0-1:
- **0.0-0.3**: You are very uncertain about your judgment and may change your mind easily
- **0.4-0.6**: You have some confidence in your judgment but still have doubts
- **0.7-1.0**: You are very confident in your judgment and unlikely to change easily

## Belief Dimension Explanation
- **evidence_strength**: The strength and persuasiveness of the opposing party's evidence
- **legal_position**: The degree of advantage the opposing party has in legal arguments
- **credibility**: The overall credibility and professionalism of the opposing party
- **strategic_competence**: The opposing party's strategic thinking and debating skills
- **winning_prob**: The overall probability that the opposing party will win the case
"""

    elif scenario == ScenarioType.PERSONA_CHAT:
        return f"""A {scenario.value} is currently underway. Please carefully read all information and act in accordance with your persona.

# Strategy Definitions
## Cooperation
Participants choose actions that benefit the collective or the other party, even if it means sacrificing immediate personal interests in the short term. The goal is to achieve win-win outcomes or optimal collective results. Examples include actively listening, providing supportive information, seeking mutual understanding, and avoiding confrontational remarks.

## Competition
Participants choose actions aimed at maximizing their own relative advantages or payoffs, usually at the expense of the other party's interests. The goal is to pursue victory rather than consensus. Examples include attempting to persuade or overwhelm others, exposing weaknesses, and competing for conversational dominance.

## Coopetition
Participants cooperate in certain aspects to achieve common goals or create greater value, while competing in other aspects to secure the payoffs generated. Examples include sharing personal experiences while competing for emotional validation.

# Belief Explanation
## What are Beliefs?
Beliefs are your subjective assessments of the other persona's personality and communication style across various dimensions, with values ranging from 0-1:
- **0.0-0.3**: You believe the other persona performs poorly or has weak capabilities in this aspect
- **0.4-0.6**: You believe the other persona performs averagely or at a moderate level in this aspect
- **0.7-1.0**: You believe the other persona performs excellently or has strong capabilities in this aspect

## What is Confidence?
Confidence represents your degree of certainty in your own judgment, with values ranging from 0-1:
- **0.0-0.3**: You are very uncertain about your judgment and may change your mind easily
- **0.4-0.6**: You have some confidence in your judgment but still have doubts
- **0.7-1.0**: You are very confident in your judgment and unlikely to change easily

## Belief Dimension Explanations
- **personality_consistency**: How well the other persona aligns with their defined personality
- **communication_skill**: The effectiveness and clarity of the other persona's communication
- **emotional_intelligence**: The awareness and management of emotions in interaction
- **engagement_level**: The level of active participation and interest shown
- **relationship_building**: The ability to foster positive interpersonal connections
"""

    else:
        raise ValueError(f"Unsupported scenario type: {scenario}")
