"""
evaluate.py
RUN THIS IN KAGGLE/COLAB (needs the real LLM + your trained Q-table).

This is the FAIR TEST: we take our 20 held-out test questions (which
the agent NEVER saw during training) and run them through 3 strategies:

  1. RANDOM   - pick a template completely at random each time
  2. FIXED    - always use ONE "reasonable default" template (step_by_step)
  3. LEARNED  - use our trained agent's best-known template per category

Whoever scores highest on average, on these NEW unseen questions,
wins - and by how much tells us if the RL agent actually learned
something useful, not just memorized the training set.
"""

import json
import csv
import random
import time
import numpy as np

from templates import apply_template, TEMPLATE_NAMES
from classifier import classify_query, CATEGORIES
from llm_utils import generate_response
from reward_model import build_judge_prompt, parse_judge_response, reward_from_scores
from q_learning_agent import QLearningAgent

random.seed(123)  # different seed than training, for a truly independent test

FIXED_TEMPLATE_ID = 0  # "step_by_step" - a reasonable, commonly-used default


def score_query(query: str, template_id: int) -> dict:
    """Runs one query through the full pipeline and returns timing + scores."""
    start_time = time.time()

    prompt = apply_template(query, template_id)
    response = generate_response(prompt)

    judge_prompt = build_judge_prompt(query, response)
    judge_reply = generate_response(judge_prompt, max_new_tokens=100)
    scores = parse_judge_response(judge_reply)
    reward = reward_from_scores(scores)

    elapsed = time.time() - start_time

    return {
        "response": response,
        "response_length_words": len(response.split()),
        "reward": reward,
        "time_seconds": round(elapsed, 2),
    }


def main():
    with open("test_queries.json") as f:
        test_queries = json.load(f)

    # Load our trained agent's knowledge
    agent = QLearningAgent(num_states=5, num_actions=5)
    agent.q_table = np.load("trained_q_table.npy")

    print("Loaded trained Q-table:")
    print(agent.get_q_table_readable(CATEGORIES, TEMPLATE_NAMES))
    print()

    results = []

    for i, item in enumerate(test_queries):
        query = item["query"]
        category_id = classify_query(query)
        category_name = CATEGORIES[category_id]

        print(f"[{i+1}/{len(test_queries)}] {query}  (category: {category_name})")

        # ---- Strategy 1: RANDOM ----
        random_template = random.randint(0, 4)
        random_result = score_query(query, random_template)

        # ---- Strategy 2: FIXED ----
        fixed_result = score_query(query, FIXED_TEMPLATE_ID)

        # ---- Strategy 3: LEARNED ----
        learned_template = agent.best_action_for_state(category_id)
        learned_result = score_query(query, learned_template)

        for strategy_name, template_used, result in [
            ("random", random_template, random_result),
            ("fixed", FIXED_TEMPLATE_ID, fixed_result),
            ("learned", learned_template, learned_result),
        ]:
            results.append({
                "query": query,
                "category": category_name,
                "strategy": strategy_name,
                "template_used": TEMPLATE_NAMES[template_used],
                "reward": result["reward"],
                "response_length_words": result["response_length_words"],
                "time_seconds": result["time_seconds"],
            })

        print(f"    random={random_result['reward']:.2f}  "
              f"fixed={fixed_result['reward']:.2f}  "
              f"learned={learned_result['reward']:.2f}")

    with open("evaluation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to evaluation_results.csv")
    print("Next: run analyze_results.py to see the final comparison.")


if __name__ == "__main__":
    main()
