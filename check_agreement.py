"""
check_agreement.py
Run this AFTER you've filled in the MY_MANUAL_SCORE_1_TO_5 column
in judge_validation.csv by hand.

This checks: does the AI judge roughly agree with human judgment?
We don't need PERFECT agreement - just close enough that we can
trust the judge to guide our RL agent's learning.
"""

import csv


def load_and_check(csv_path: str):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Separate out rows where manual score was actually filled in
    scored_rows = []
    skipped = 0
    for row in rows:
        manual_str = row["MY_MANUAL_SCORE_1_TO_5"].strip()
        judge_relevance = row["judge_relevance"]

        if not manual_str:
            skipped += 1
            continue
        if judge_relevance == "PARSE_FAILED":
            skipped += 1
            continue

        try:
            manual_score = float(manual_str)
        except ValueError:
            skipped += 1
            continue

        if manual_score < 1 or manual_score > 5:
            print(f"  WARNING: row '{row['query']}' has manual score {manual_score}, "
                  f"outside 1-5 range - skipping")
            skipped += 1
            continue

        # Judge's "total" isn't in the CSV directly, so we recompute it
        # from the 3 sub-scores that ARE saved.
        try:
            judge_total = (
                float(row["judge_relevance"])
                + float(row["judge_clarity"])
                + float(row["judge_completeness"])
            ) / 3
        except ValueError:
            skipped += 1
            continue

        scored_rows.append({
            "query": row["query"],
            "manual": manual_score,
            "judge": round(judge_total, 2),
            "diff": round(abs(manual_score - judge_total), 2),
        })

    return scored_rows, skipped


def summarize(scored_rows: list):
    if not scored_rows:
        print("No valid scored rows found - did you fill in the manual score column?")
        return

    n = len(scored_rows)
    avg_diff = sum(r["diff"] for r in scored_rows) / n

    # "Close agreement" = within 1 point on the 1-5 scale
    close_agreement = sum(1 for r in scored_rows if r["diff"] <= 1.0)
    exact_agreement = sum(1 for r in scored_rows if r["diff"] <= 0.5)

    print(f"Total rows compared: {n}")
    print(f"Average difference (manual vs judge): {avg_diff:.2f} points (on a 1-5 scale)")
    print(f"Rows within 1 point of each other: {close_agreement}/{n} ({100*close_agreement/n:.0f}%)")
    print(f"Rows within 0.5 points of each other: {exact_agreement}/{n} ({100*exact_agreement/n:.0f}%)")

    print("\nBiggest disagreements (check these manually):")
    worst = sorted(scored_rows, key=lambda r: -r["diff"])[:3]
    for r in worst:
        print(f"  '{r['query'][:50]}...' | manual={r['manual']} judge={r['judge']} diff={r['diff']}")

    print("\n--- VERDICT ---")
    if close_agreement / n >= 0.7:
        print("GOOD: Judge agrees closely with you on 70%+ of cases. Safe to proceed.")
    else:
        print("CAUTION: Judge disagrees with you often. Consider revising the judge")
        print("prompt (in reward_model.py's build_judge_prompt) before trusting it")
        print("to train the RL agent.")


if __name__ == "__main__":
    scored_rows, skipped = load_and_check("judge_validation.csv")
    print(f"(Skipped {skipped} rows with missing/invalid data)\n")
    summarize(scored_rows)
