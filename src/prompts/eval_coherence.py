def evaluate_coherence_prompt(dialogue_history, utterance):
    return f"""
# Role: AI Dialogue Evaluator

## Task:
You are an expert in analyzing conversations. Your task is to evaluate a single utterance from a dialogue based on the provided conversation history. Please assess the utterance on three key aspects: Coherence, Relevance, and Fluency. Provide a score from 0 to 5 for each aspect, along with a brief justification for your scores.

## Conversation Context:
You will be given the dialogue history leading up to the utterance you need to evaluate, and the specific utterance itself.

### Dialogue History:
{dialogue_history}

### Utterance to Evaluate:
{utterance}

## Evaluation Criteria & Scoring Guide (0-5 points):

### 1. Coherence:
This assesses how logically the utterance connects with the preceding conversation.
- **5 (Excellent):** The utterance is a perfectly logical continuation of the conversation. It follows the flow of the discussion seamlessly.
- **4 (Good):** The utterance is logical and fits well into the conversation, though there might be a very minor logical leap.
- **3 (Acceptable):** The utterance is somewhat logical but might require some inference to connect it to the previous turns. The connection is not immediately obvious.
- **2 (Weak):** The utterance has a weak logical connection to the conversation. It may be a significant deviation or a non-sequitur.
- **1 (Poor):** The utterance is illogical and disrupts the flow of the conversation.
- **0 (No Coherence):** The utterance is completely random and has no logical connection to the dialogue history.

### 2. Relevance:
This assesses how relevant the utterance is to the overall topic of the conversation.
- **5 (Excellent):** The utterance is highly relevant and directly addresses the core topic of the discussion.
- **4 (Good):** The utterance is relevant to the topic, though it might touch upon a slightly tangential point.
- **3 (Acceptable):** The utterance is broadly related to the topic but doesn't directly contribute to the main conversation thread.
- **2 (Weak):** The utterance is only slightly relevant and feels out of place.
- **1 (Poor):** The utterance is largely irrelevant to the conversation topic.
- **0 (Irrelevant):** The utterance is completely off-topic.

### 3. Fluency:
This assesses the grammatical correctness, clarity, and naturalness of the utterance.
- **5 (Excellent):** The utterance is grammatically perfect, clear, and sounds like natural human speech.
- **4 (Good):** The utterance has minor grammatical errors or awkward phrasing but is still easy to understand.
- **3 (Acceptable):** The utterance has noticeable grammatical errors or is poorly phrased, but the meaning is still comprehensible.
- **2 (Weak):** The utterance is difficult to understand due to significant grammatical errors or confusing structure.
- **1 (Poor):** The utterance is nearly incomprehensible.
- **0 (Incomprehensible):** The utterance is nonsensical or ungrammatical to the point that it cannot be understood.

## Your Evaluation:
Please provide your evaluation in the following JSON format. Do not include any other text outside of the JSON block.

```json
{{
  "coherence": {{
    "score": <score_0_to_5>,
    "justification": "<brief_justification>"
  }},
  "relevance": {{
    "score": <score_0_to_5>,
    "justification": "<brief_justification>"
  }},
  "fluency": {{
    "score": <score_0_to_5>,
    "justification": "<brief_justification>"
  }}
}}
```

## Few-shot Examples:
```json
{{
  'coherence': {{
    'score': 4,
    'justification': "Frank's response maintains coherence by addressing both the videos of Alex's dogs and the moving process which Alex asked about. The slight logical leap is the introduction of dancing at an ultimate frisbee tournament, which is a novel combination not previously discussed but does tie Alex's interest in dancing and Frank's in ultimate frisbee."
  }},
  'relevance': {{
    'score': 4,
    'justification': "The response is relevant as it addresses the topics directly mentioned by Alex regarding the dogs' videos and the inquiry about the new house. The mention of dancing at an ultimate frisbee tournament introduces a slight tangent, but it creatively connects to the interests of both individuals, thus still holding relevance."
  }},
  'fluency': {{
    'score': 5,
    'justification': 'The utterance is grammatically correct, clear, and natural. It maintains a conversational tone that fits well with the casual exchange between Frank and Alex, making it sound like a natural human dialogue.'
  }}
}}
```
"""
