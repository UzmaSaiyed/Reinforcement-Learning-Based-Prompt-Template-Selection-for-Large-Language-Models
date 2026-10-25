"""
split_dataset.py
Splits our 80 queries into:
  - train_queries.json  (60 queries) - used to TEACH the RL agent
  - test_queries.json   (20 queries) - held back, used ONLY for final evaluation

We take 4 out of every 16 from each category for the test set, so both
train and test have a balanced mix of all 5 question types.
"""

import json

with open("queries.json") as f:
    queries = json.load(f)

train = []
test = []

# Each category has 16 queries in a contiguous block (see build_dataset.py).
# We take the last 4 of each block of 16 for testing, first 12 for training.
for block_start in range(0, 80, 16):
    block = queries[block_start: block_start + 16]
    train.extend(block[:12])
    test.extend(block[12:])

with open("train_queries.json", "w") as f:
    json.dump(train, f, indent=2)

with open("test_queries.json", "w") as f:
    json.dump(test, f, indent=2)

print(f"Train set: {len(train)} queries -> train_queries.json")
print(f"Test set: {len(test)} queries -> test_queries.json")
