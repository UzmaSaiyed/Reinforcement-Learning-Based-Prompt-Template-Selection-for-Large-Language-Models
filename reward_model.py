"""
reward_model.py
This is our "grading system" - it asks an LLM to judge how good a
response was, and turns that judgment into a single reward number.

Two parts:
  1. build_judge_prompt() - writes out the grading instructions
  2. parse_judge_response() - reads the judge's reply and extracts scores
"""

import json
import re


def build_judge_prompt(query: str, response: str) -> str:
    """
    Creates the prompt we send to the LLM judge.
    We ask for a JSON reply so it's easy to parse reliably.

    NOTE: We found in testing that a loosely-worded judge prompt made the
    LLM judge too generous (it gave 5/5 in cases where a human grader gave
    3/5). This version is stricter: it defines exactly what each score
    number means, and explicitly tells the judge that most answers should
    land in the middle, not at the top.
    """
    return f"""You are a STRICT grader evaluating an AI assistant's answer.
Most answers are NOT perfect - reserve 5/5 only for truly excellent answers
with no flaws. Do not give high scores just because the answer is on-topic.

Original question: {query}

AI's answer: {response}

Grade the answer on 3 things, using this exact scale for each:
1 = Poor: wrong, irrelevant, or barely addresses the question
2 = Below average: partially relevant but has significant gaps or errors
3 = Average: answers the question adequately but is generic, incomplete, or could be clearer
4 = Good: answers well with only minor room for improvement
5 = Excellent: thorough, precise, and clear - nothing meaningful to improve

Grade on:
- relevance: does it actually answer the question asked?
- clarity: is it easy to understand?
- completeness: does it cover what's needed, not too little or too much?

Be critical. If an answer is short, vague, or generic, it should NOT score
above 3, even if it is technically correct.

Reply with ONLY a JSON object in this exact format, nothing else:
{{"relevance": <number 1-5>, "clarity": <number 1-5>, "completeness": <number 1-5>}}
"""


def parse_judge_response(judge_text: str) -> dict:
    """
    Takes the judge LLM's raw text reply and extracts the 3 scores.
    LLMs sometimes add extra text around the JSON (like "Here's my
    grading: {...}") even when told not to, so we search for the
    JSON object inside the text rather than assuming the whole
    reply is clean JSON.

    Returns a dict like: {"relevance": 4, "clarity": 5, "completeness": 3, "total": 4.0}
    If parsing fails, returns None so the caller can handle/skip it.
    """
    # Find the first { ... } block in the text
    match = re.search(r"\{.*\}", judge_text, re.DOTALL)
    if not match:
        return None

    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    required_keys = ["relevance", "clarity", "completeness"]
    if not all(k in scores for k in required_keys):
        return None

    # Make sure values are actually numbers in the 1-5 range
    for k in required_keys:
        val = scores[k]
        if not isinstance(val, (int, float)):
            return None
        if val < 1 or val > 5:
            return None

    scores["total"] = round(
        (scores["relevance"] + scores["clarity"] + scores["completeness"]) / 3, 2
    )
    return scores


def reward_from_scores(scores: dict) -> float:
    """
    Converts our 1-5 score dict into a single reward number for the RL agent.
    We normalize to a 0-1 range, which works better with Q-learning updates.
    """
    if scores is None:
        return 0.0  # if judging failed, treat as worst-case reward
    # scores["total"] is on a 1-5 scale -> convert to 0-1 scale
    return round((scores["total"] - 1) / 4, 3)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Clean JSON response (ideal case)")
    fake_reply = '{"relevance": 5, "clarity": 4, "completeness": 4}'
    scores = parse_judge_response(fake_reply)
    print(f"  Input: {fake_reply}")
    print(f"  Parsed: {scores}")
    print(f"  Reward: {reward_from_scores(scores)}")

    print("\n" + "=" * 60)
    print("TEST 2: Judge adds extra text around JSON (realistic case)")
    fake_reply = 'Sure, here is my grading:\n{"relevance": 3, "clarity": 3, "completeness": 2}\nHope this helps!'
    scores = parse_judge_response(fake_reply)
    print(f"  Input: {fake_reply}")
    print(f"  Parsed: {scores}")
    print(f"  Reward: {reward_from_scores(scores)}")

    print("\n" + "=" * 60)
    print("TEST 3: Broken/garbage response (should fail gracefully)")
    fake_reply = "I think the answer was pretty good overall, maybe an 8/10."
    scores = parse_judge_response(fake_reply)
    print(f"  Input: {fake_reply}")
    print(f"  Parsed: {scores}")
    print(f"  Reward: {reward_from_scores(scores)}")

    print("\n" + "=" * 60)
    print("TEST 4: Judge gives an out-of-range score (should fail gracefully)")
    fake_reply = '{"relevance": 9, "clarity": 4, "completeness": 3}'
    scores = parse_judge_response(fake_reply)
    print(f"  Input: {fake_reply}")
    print(f"  Parsed: {scores}")
    print(f"  Reward: {reward_from_scores(scores)}")

    print("\n" + "=" * 60)
    print("TEST 5: Best possible score (sanity check on scaling)")
    fake_reply = '{"relevance": 5, "clarity": 5, "completeness": 5}'
    scores = parse_judge_response(fake_reply)
    print(f"  Parsed: {scores}")
    print(f"  Reward: {reward_from_scores(scores)} (should be 1.0 - the max)")

    print("\n" + "=" * 60)
    print("TEST 6: Worst possible score (sanity check on scaling)")
    fake_reply = '{"relevance": 1, "clarity": 1, "completeness": 1}'
    scores = parse_judge_response(fake_reply)
    print(f"  Parsed: {scores}")
    print(f"  Reward: {reward_from_scores(scores)} (should be 0.0 - the min)")
