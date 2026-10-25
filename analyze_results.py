"""
analyze_results.py
Run this AFTER evaluate.py has produced evaluation_results.csv.

This produces:
  1. A summary table: average reward per strategy (random vs fixed vs learned)
  2. A breakdown by category (does "learned" help more for some question types?)
  3. Two plots: bar chart comparing strategies, and a category breakdown chart
"""

import csv
from collections import defaultdict
import matplotlib.pyplot as plt


def load_results(path="evaluation_results.csv"):
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize_by_strategy(rows):
    totals = defaultdict(list)
    lengths = defaultdict(list)
    times = defaultdict(list)

    for r in rows:
        totals[r["strategy"]].append(float(r["reward"]))
        lengths[r["strategy"]].append(int(r["response_length_words"]))
        times[r["strategy"]].append(float(r["time_seconds"]))

    print("=" * 60)
    print("OVERALL RESULTS: Average reward per strategy")
    print("=" * 60)
    for strategy in ["random", "fixed", "learned"]:
        avg_reward = sum(totals[strategy]) / len(totals[strategy])
        avg_len = sum(lengths[strategy]) / len(lengths[strategy])
        avg_time = sum(times[strategy]) / len(times[strategy])
        print(f"{strategy:10s} | avg reward: {avg_reward:.3f} | "
              f"avg length: {avg_len:.1f} words | avg time: {avg_time:.2f}s")

    return totals


def summarize_by_category(rows):
    print("\n" + "=" * 60)
    print("BREAKDOWN BY CATEGORY: Does 'learned' help more for some question types?")
    print("=" * 60)

    by_cat_strategy = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cat_strategy[r["category"]][r["strategy"]].append(float(r["reward"]))

    category_data = {}
    for category in sorted(by_cat_strategy.keys()):
        print(f"\n{category}:")
        cat_results = {}
        for strategy in ["random", "fixed", "learned"]:
            values = by_cat_strategy[category][strategy]
            avg = sum(values) / len(values) if values else 0
            cat_results[strategy] = avg
            print(f"  {strategy:10s}: {avg:.3f}")
        category_data[category] = cat_results

    return category_data


def make_plots(totals, category_data):
    # Plot 1: overall strategy comparison
    strategies = ["random", "fixed", "learned"]
    avg_rewards = [sum(totals[s]) / len(totals[s]) for s in strategies]

    plt.figure(figsize=(6, 4))
    colors = ["#cccccc", "#7fa8d9", "#4caf50"]
    plt.bar(strategies, avg_rewards, color=colors)
    plt.ylabel("Average Reward (0-1 scale)")
    plt.title("Prompt Selection Strategy Comparison")
    plt.ylim(0, 1)
    for i, v in enumerate(avg_rewards):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("strategy_comparison.png", dpi=150)
    print("\nSaved plot: strategy_comparison.png")

    # Plot 2: category breakdown
    categories = list(category_data.keys())
    x = range(len(categories))
    width = 0.25

    plt.figure(figsize=(9, 5))
    for i, strategy in enumerate(strategies):
        values = [category_data[c][strategy] for c in categories]
        positions = [xi + i * width for xi in x]
        plt.bar(positions, values, width=width, label=strategy, color=colors[i])

    plt.xticks([xi + width for xi in x], categories, rotation=20)
    plt.ylabel("Average Reward")
    plt.title("Strategy Comparison by Query Category")
    plt.legend()
    plt.tight_layout()
    plt.savefig("category_breakdown.png", dpi=150)
    print("Saved plot: category_breakdown.png")


if __name__ == "__main__":
    rows = load_results()
    totals = summarize_by_strategy(rows)
    category_data = summarize_by_category(rows)
    make_plots(totals, category_data)

    print("\n" + "=" * 60)
    print("DONE. Use the printed numbers above + the two PNG files for your")
    print("results table and plots in your final write-up.")
