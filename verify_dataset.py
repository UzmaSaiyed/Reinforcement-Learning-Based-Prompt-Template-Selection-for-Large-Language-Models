"""
verify_dataset.py
This checks: does our classifier correctly sort each question
into the category we INTENDED when we wrote it?

Since we wrote the dataset in 5 clear blocks (16 factual, 16 how_to, etc.),
we know what category each question SHOULD get. This script checks
if our classifier agrees.
"""

import json
from classifier import classify_query, CATEGORIES

with open("queries.json") as f:
    queries = json.load(f)

# We wrote the dataset in blocks of 16, in this exact order:
# 0-15: factual, 16-31: how_to, 32-47: conceptual, 48-63: comparison, 64-79: troubleshooting
intended_categories = (
    [0] * 16 +  # factual
    [1] * 16 +  # how_to
    [2] * 16 +  # conceptual
    [3] * 16 +  # comparison
    [4] * 16    # troubleshooting
)

mismatches = []
for i, (item, intended) in enumerate(zip(queries, intended_categories)):
    query = item["query"]
    predicted = classify_query(query)
    if predicted != intended:
        mismatches.append({
            "index": i,
            "query": query,
            "intended": CATEGORIES[intended],
            "predicted": CATEGORIES[predicted],
        })

print(f"Total queries checked: {len(queries)}")
print(f"Correctly classified: {len(queries) - len(mismatches)}")
print(f"Mismatches: {len(mismatches)}\n")

if mismatches:
    print("MISMATCHED QUERIES (need fixing):")
    for m in mismatches:
        print(f"  [{m['index']}] '{m['query']}'")
        print(f"      intended: {m['intended']}  |  classifier said: {m['predicted']}")
else:
    print("All queries classified correctly! Dataset is ready to use.")
