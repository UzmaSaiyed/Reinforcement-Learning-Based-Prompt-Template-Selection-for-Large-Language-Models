"""
templates.py
This file holds our 5 "recipes" for asking a question.
Each recipe (template) takes a query and wraps it in different instructions.
"""

TEMPLATES = {
    0: "Explain step-by-step: {query}",
    1: "Give a short, direct answer: {query}",
    2: "Explain with a simple example: {query}",
    3: "Explain this as if teaching a complete beginner: {query}",
    4: "Give a detailed, thorough explanation: {query}",
}

TEMPLATE_NAMES = {
    0: "step_by_step",
    1: "concise",
    2: "example_based",
    3: "beginner_friendly",
    4: "detailed",
}


def apply_template(query: str, template_id: int) -> str:
    """
    Takes a raw question and a template number,
    and returns the final prompt to send to the LLM.

    Example:
        apply_template("How do airplanes fly?", 0)
        -> "Explain step-by-step: How do airplanes fly?"
    """
    if template_id not in TEMPLATES:
        raise ValueError(f"Unknown template_id: {template_id}")
    return TEMPLATES[template_id].format(query=query)


if __name__ == "__main__":
    # Quick test to make sure everything works
    test_query = "How do airplanes fly?"
    print("Testing all 5 templates on one query:\n")
    for tid in TEMPLATES:
        prompt = apply_template(test_query, tid)
        print(f"[Template {tid} - {TEMPLATE_NAMES[tid]}]")
        print(f"  {prompt}\n")
