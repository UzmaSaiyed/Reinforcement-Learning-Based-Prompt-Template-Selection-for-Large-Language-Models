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
    # Real troubleshooting questions use MANY different words, not just "work" -
    # e.g. "won't turn on", "keeps crashing", "so slow", "strange noise".
    # We combine a broader set of signals: does the question mention "my"
    # (a personal device/thing) AND show some sign of malfunction?
    mentions_my_thing = "my " in q or q.startswith("my")

    negation_words = ["isn't", "doesn't", "won't", "wont", "aren't", "not "]
    has_negation = any(word in q for word in negation_words)

    repeats_failing = "keep " in q or "keeps " in q  # "keeps crashing", "keep freezing"

    trouble_phrases = ["so slow", "so fast", "strange noise", "what could be wrong",
                       "what could cause", "issue", "error", "problem", "bug", "broken", "fail"]
    has_trouble_phrase = any(word in q for word in trouble_phrases)

    if mentions_my_thing and (has_negation or repeats_failing or has_trouble_phrase):
        return 4
    # Catch standalone trouble words even without "my" (e.g. "there's a bug in X")
    if any(word in q for word in ["error", "broken", "issue", "problem", "bug"]):
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
