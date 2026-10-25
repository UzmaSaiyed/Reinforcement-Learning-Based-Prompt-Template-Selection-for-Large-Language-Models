"""
generate_and_score.py
RUN THIS IN COLAB (needs the real LLM).

For a sample of 20 queries, this:
  1. Generates an LLM response (using template 0: step-by-step)
  2. Asks the LLM judge to score that response
  3. Saves everything to a CSV, with an empty column for YOU to fill in
     your own manual score later.

We only test ONE template here (not all 5) because the goal right now
is just to check: does the judge agree with a human? Not yet comparing
templates - that comes later once we trust the judge.
"""

import json
import csv
import random

from templates import apply_template
from classifier import classify_query, CATEGORIES
from llm_utils import generate_response
from reward_model import build_judge_prompt, parse_judge_response, reward_from_scores

# Load our training queries and randomly sample 20 for this validation check
with open("train_queries.json") as f:
    all_queries = json.load(f)

random.seed(42)  # fixed seed so results are reproducible if we rerun
sample = random.sample(all_queries, min(20, len(all_queries)))

TEMPLATE_ID = 0  # step-by-step, just for this validation test

rows = []
for i, item in enumerate(sample):
    query = item["query"]
    print(f"[{i+1}/{len(sample)}] Processing: {query}")

    # Step 1: classify and generate a response
    category_id = classify_query(query)
    prompt = apply_template(query, TEMPLATE_ID)
    response = generate_response(prompt)

    # Step 2: ask the judge to score it
    judge_prompt = build_judge_prompt(query, response)
    judge_reply = generate_response(judge_prompt, max_new_tokens=100)
    scores = parse_judge_response(judge_reply)
    judge_reward = reward_from_scores(scores)

    rows.append({
        "index": i,
        "query": query,
        "category": CATEGORIES[category_id],
        "response": response,
        "judge_raw_reply": judge_reply,
        "judge_relevance": scores["relevance"] if scores else "PARSE_FAILED",
        "judge_clarity": scores["clarity"] if scores else "PARSE_FAILED",
        "judge_completeness": scores["completeness"] if scores else "PARSE_FAILED",
        "judge_reward": judge_reward,
        "MY_MANUAL_SCORE_1_TO_5": "",  # <-- YOU fill this in by hand later
    })

# Save to CSV so it's easy to open and fill in manually (e.g. in Excel/Sheets)
with open("judge_validation.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to judge_validation.csv")
print("Next step: open this file, read each query+response, and fill in")
print("your own 1-5 score in the MY_MANUAL_SCORE_1_TO_5 column.")
