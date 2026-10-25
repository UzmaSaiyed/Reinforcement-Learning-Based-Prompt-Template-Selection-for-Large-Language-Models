"""
train_agent.py
RUN THIS IN COLAB (needs the real LLM).

This is the main event: connects everything we've built so far and
actually trains the RL agent on your 60 real training questions.

What happens each step:
  1. Pick a query from the training set
  2. Classify it (what category is this question?)
  3. Agent picks a template (mostly the best-known one, sometimes random)
  4. Generate the LLM's response using that template
  5. Judge scores the response -> becomes the reward
  6. Agent updates its Q-table based on this reward
  7. Repeat, many times, gradually exploring less and exploiting more
"""

import json
import csv
import random

from templates import apply_template, TEMPLATE_NAMES
from classifier import classify_query, CATEGORIES
from llm_utils import generate_response
from reward_model import build_judge_prompt, parse_judge_response, reward_from_scores
from q_learning_agent import QLearningAgent

# ---- Settings ----
NUM_EPOCHS = 8              # how many full passes over the training data
INITIAL_EPSILON = 0.4       # start with 40% random exploration
EPSILON_DECAY = 0.85        # shrink exploration by 15% after each epoch
MIN_EPSILON = 0.05          # never go below 5% exploration
LEARNING_RATE = 0.15

random.seed(42)


def get_reward_for_query(query: str, template_id: int) -> tuple:
    """
    Runs one query through the full pipeline: generate a response with
    the given template, then judge it. Returns (reward, response, scores).
    """
    prompt = apply_template(query, template_id)
    response = generate_response(prompt)

    judge_prompt = build_judge_prompt(query, response)
    judge_reply = generate_response(judge_prompt, max_new_tokens=100)
    scores = parse_judge_response(judge_reply)
    reward = reward_from_scores(scores)

    return reward, response, scores


def train():
    with open("train_queries.json") as f:
        train_queries = json.load(f)

    agent = QLearningAgent(num_states=5, num_actions=5, learning_rate=LEARNING_RATE)

    epsilon = INITIAL_EPSILON
    log_rows = []          # detailed log of every single step
    epoch_summaries = []   # one row per epoch, for the reward-over-time plot

    print(f"Starting training: {NUM_EPOCHS} epochs x {len(train_queries)} queries "
          f"= {NUM_EPOCHS * len(train_queries)} total LLM calls (this will take a while)\n")

    for epoch in range(NUM_EPOCHS):
        shuffled = train_queries.copy()
        random.shuffle(shuffled)

        epoch_rewards = []
        for step, item in enumerate(shuffled):
            query = item["query"]
            state = classify_query(query)
            action = agent.choose_action(state, epsilon)

            reward, response, scores = get_reward_for_query(query, action)
            agent.update(state, action, reward)

            epoch_rewards.append(reward)
            log_rows.append({
                "epoch": epoch + 1,
                "step": step + 1,
                "query": query,
                "category": CATEGORIES[state],
                "template_used": TEMPLATE_NAMES[action],
                "reward": reward,
                "epsilon": round(epsilon, 3),
            })

            if (step + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, query {step+1}/{len(shuffled)} "
                      f"- reward so far this epoch: {sum(epoch_rewards)/len(epoch_rewards):.3f}")

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        epoch_summaries.append({"epoch": epoch + 1, "avg_reward": avg_reward, "epsilon": epsilon})
        print(f"\n== Epoch {epoch+1} complete. Average reward: {avg_reward:.3f}, "
              f"epsilon was {epsilon:.3f} ==\n")

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

    # ---- Save everything ----
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    with open("epoch_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_summaries[0].keys())
        writer.writeheader()
        writer.writerows(epoch_summaries)

    # Save the trained Q-table so we can reuse it later without retraining
    import numpy as np
    np.save("trained_q_table.npy", agent.q_table)

    print("Training complete!")
    print("Saved: training_log.csv (every step), epoch_summary.csv (per-epoch averages),")
    print("       trained_q_table.npy (the trained agent's knowledge)\n")

    print("=" * 60)
    print("FINAL LEARNED POLICY (best template per query category):")
    print(agent.get_q_table_readable(CATEGORIES, TEMPLATE_NAMES))

    return agent, log_rows, epoch_summaries


if __name__ == "__main__":
    train()
