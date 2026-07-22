"""
classifier.py
This figures out "what type of question is this?"
We use simple keyword rules - no AI needed for this part.

The 5 categories (states) are:
  0 - factual        ("What is X?")
  1 - how_to          ("How do I do X?")
  2 - conceptual      ("Why does X happen?")
  3 - comparison      ("What's the difference between X and Y?")
  4 - troubleshooting ("Why isn't X working?")
"""

CATEGORIES = {
    0: "factual",
    1: "how_to",
    2: "conceptual",
    3: "comparison",
    4: "troubleshooting",
}


def classify_query(query: str) -> int:
    """
    Takes a question (text) and returns a category number (0-4).
    Uses simple keyword matching - checked in a specific order
    so more specific patterns are caught before general ones.
    """
    q = query.lower().strip()

    # Troubleshooting patterns (check first - often overlaps with "why")
    # We check for "isn't" / "not" / "doesn't" TOGETHER with "work" anywhere
    # in the sentence, since the exact words may not be right next to each other
    # (e.g. "isn't my code working" instead of "isn't working").
    has_negation = any(word in q for word in ["isn't", "doesn't", "won't", "not "])
    has_work_or_fail = any(word in q for word in ["work", "fail", "error", "broken", "issue", "problem", "bug"])
    if (has_negation and has_work_or_fail) or any(
        word in q for word in ["error", "broken", "issue", "problem", "bug"]
    ):
        return 4

    # Comparison patterns
    comparison_words = ["difference between", " vs ", "versus", "compared to",
                        "better than", "which is better"]
    if any(word in q for word in comparison_words):
        return 3

    # How-to patterns
    if q.startswith("how do") or q.startswith("how to") or q.startswith("how can"):
        return 1

    # Conceptual patterns (why/what causes/what happens)
    # NOTE: each condition must explicitly check "in q" - combining them with
    # bare "or" like `"what causes" or "what happens" in q` is a bug, because
    # a non-empty string is always True in Python regardless of its content.
    if q.startswith("why") or "what causes" in q or "what happens" in q:
        return 2

    # Factual patterns (what is / who is / when / where)
    if q.startswith("what is") or q.startswith("what are") or q.startswith("who") \
       or q.startswith("when") or q.startswith("where"):
        return 0

    # Default fallback: treat unclear questions as factual
    return 0


if __name__ == "__main__":
    # Quick test with example questions
    test_queries = [
        "What is photosynthesis?",
        "How do I bake a cake?",
        "Why does ice float on water?",
        "What's the difference between a virus and bacteria?",
        "Why isn't my code working?",
        "How to learn Python?",
        "What causes earthquakes?",
    ]

    print("Testing the classifier:\n")
    for query in test_queries:
        cat_id = classify_query(query)
        cat_name = CATEGORIES[cat_id]
        print(f"  '{query}' -> [{cat_id}] {cat_name}")
