"""
test_pipeline.py
Run this in Google Colab AFTER uploading templates.py, classifier.py, and llm_utils.py.

This tests the full flow:
  question -> classify it -> apply a template -> get LLM response
"""

from templates import apply_template, TEMPLATE_NAMES
from classifier import classify_query, CATEGORIES
from llm_utils import generate_response

test_queries = [
    "What is photosynthesis?",
    "How do I bake a cake?",
    "Why does ice float on water?",
]

for query in test_queries:
    category_id = classify_query(query)
    category_name = CATEGORIES[category_id]

    print("=" * 60)
    print(f"QUERY: {query}")
    print(f"Detected category: [{category_id}] {category_name}")
    print()

    # Try template 0 (step-by-step) as a test
    template_id = 0
    prompt = apply_template(query, template_id)
    print(f"Using template: {TEMPLATE_NAMES[template_id]}")
    print(f"Full prompt sent to LLM: {prompt}")
    print()

    response = generate_response(prompt)
    print(f"LLM Response:\n{response}")
    print()
