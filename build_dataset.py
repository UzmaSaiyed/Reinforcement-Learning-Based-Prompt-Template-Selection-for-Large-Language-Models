"""
build_dataset.py
This creates our dataset of 80 questions, spread across 5 categories.
We write them by hand (not scraped/generated) so we know exactly what
each one is testing.

Run this file to generate queries.json.
"""

import json

# 16 questions per category x 5 categories = 80 total
queries = [
    # ---- FACTUAL (0): "What is X?" type questions ----
    {"query": "What is photosynthesis?"},
    {"query": "What is the capital of Japan?"},
    {"query": "What is machine learning?"},
    {"query": "What is inflation?"},
    {"query": "What is DNA?"},
    {"query": "What is the speed of light?"},
    {"query": "Who is the president of France?"},
    {"query": "What is a black hole?"},
    {"query": "What is the boiling point of water?"},
    {"query": "What is an ecosystem?"},
    {"query": "What are enzymes?"},
    {"query": "When was the internet invented?"},
    {"query": "Where is Mount Everest located?"},
    {"query": "What is cloud computing?"},
    {"query": "Who wrote Romeo and Juliet?"},
    {"query": "What is a supply chain?"},

    # ---- HOW_TO (1): "How do I...?" type questions ----
    {"query": "How do I bake a chocolate cake?"},
    {"query": "How do I start learning Python?"},
    {"query": "How do I improve my public speaking skills?"},
    {"query": "How do I set up a budget?"},
    {"query": "How to fix a flat bicycle tire?"},
    {"query": "How do I write a good resume?"},
    {"query": "How to train for a marathon?"},
    {"query": "How do I meditate properly?"},
    {"query": "How to grow tomatoes at home?"},
    {"query": "How do I negotiate a salary?"},
    {"query": "How can I improve my sleep quality?"},
    {"query": "How to make a paper airplane?"},
    {"query": "How do I create a study schedule?"},
    {"query": "How to change a car tire?"},
    {"query": "How do I apply for a passport?"},
    {"query": "How can I reduce screen time?"},

    # ---- CONCEPTUAL (2): "Why...?" / "What causes...?" type questions ----
    {"query": "Why does ice float on water?"},
    {"query": "Why is the sky blue?"},
    {"query": "Why do we dream when we sleep?"},
    {"query": "What causes earthquakes?"},
    {"query": "Why do leaves change color in autumn?"},
    {"query": "What causes inflation to rise?"},
    {"query": "Why does the moon affect ocean tides?"},
    {"query": "What causes rainbows to form?"},
    {"query": "Why do we get muscle soreness after exercise?"},
    {"query": "What causes climate change?"},
    {"query": "Why do birds migrate?"},
    {"query": "What causes jet lag?"},
    {"query": "Why does bread rise when baked?"},
    {"query": "What causes allergies?"},
    {"query": "Why do metals conduct electricity?"},
    {"query": "What happens when a volcano erupts?"},

    # ---- COMPARISON (3): "difference between X and Y" type questions ----
    {"query": "What's the difference between a virus and bacteria?"},
    {"query": "What is the difference between weather and climate?"},
    {"query": "Python vs JavaScript, which is better for beginners?"},
    {"query": "What's the difference between a CV and a resume?"},
    {"query": "What is the difference between a crocodile and an alligator?"},
    {"query": "Renewable energy vs fossil fuels, which is more sustainable?"},
    {"query": "What's the difference between machine learning and deep learning?"},
    {"query": "What is the difference between a stock and a bond?"},
    {"query": "Tea vs coffee, which has more caffeine?"},
    {"query": "What's the difference between a hurricane and a typhoon?"},
    {"query": "What is the difference between HTTP and HTTPS?"},
    {"query": "What's the difference between a lawyer and a paralegal?"},
    {"query": "SQL vs NoSQL databases, which is better for structured data?"},
    {"query": "What is the difference between an introvert and an extrovert?"},
    {"query": "What's the difference between a resume and a portfolio?"},
    {"query": "What is the difference between civil law and criminal law?"},

    # ---- TROUBLESHOOTING (4): "why isn't X working" type questions ----
    {"query": "Why isn't my laptop turning on?"},
    {"query": "My Wi-Fi isn't working, what should I check?"},
    {"query": "Why does my phone battery drain so fast?"},
    {"query": "My code isn't compiling, what could be wrong?"},
    {"query": "Why is my printer not connecting?"},
    {"query": "My website isn't loading properly, why could this happen?"},
    {"query": "Why does my car engine make a strange noise?"},
    {"query": "My email isn't sending, what's the issue?"},
    {"query": "Why is my internet connection so slow?"},
    {"query": "My washing machine isn't draining water, why?"},
    {"query": "Why does my computer keep freezing?"},
    {"query": "My app keeps crashing, what could cause this?"},
    {"query": "Why won't my plant grow despite watering it?"},
    {"query": "My headphones aren't producing sound, why?"},
    {"query": "Why does my browser keep showing errors?"},
    {"query": "My phone screen isn't responding to touch, why?"},
]

if __name__ == "__main__":
    # Save the dataset
    with open("queries.json", "w") as f:
        json.dump(queries, f, indent=2)

    print(f"Total queries created: {len(queries)}")
    print("Saved to queries.json")
