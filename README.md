# RL-Based Prompt Template Selection for LLMs

A Q-learning agent that learns which of five prompt templates works best for each type of user query. Responses come from an open-source LLM (Qwen2.5-1.5B-Instruct) and are scored by an LLM-as-judge whose score is the reward.

Authors: Saiyed Uzma Banu Nasirhusen, Kasodiya Tanvi Bharatbhai (MSc AI, St. Xavier's College (Autonomous), Ahmedabad)

## Problem

LLM output quality depends on how a question is phrased ("explain step-by-step" vs. "give a short answer"). Choosing the phrasing is usually done by habit. Fixed rules cannot correct themselves when evidence shows a different template is better, and supervised learning would need labels saying which template is correct for each query. Reinforcement learning needs only feedback after acting, so an agent can learn the best template by trial and error.

## Approach

1. **Query classification** - a rule-based keyword classifier assigns each query to one of 5 categories: factual, how-to, conceptual, comparison, troubleshooting. This category is the RL **state**.
2. **Prompt templates** - 5 fixed templates (step-by-step, concise, example-based, beginner-friendly, detailed). Choosing one is the RL **action**.
3. **Q-learning agent** - a 5 x 5 Q-table updated with `Q(s,a) += alpha * (reward - Q(s,a))`, with `alpha = 0.15` and epsilon-greedy exploration (epsilon 0.4, decayed by 0.85 per epoch, minimum 0.05; 8 epochs over 60 training queries). There is no discount factor because each query is an independent one-shot decision, so the problem is a contextual bandit, the single-step case of Q-learning.
4. **LLM-as-judge reward** - the model scores each response on relevance, clarity and completeness (1-5); the mean is rescaled to a 0-1 reward with `(mean - 1) / 4`. Judge output is parsed defensively and out-of-range scores are rejected.
5. **Validation before training**
   - *Judge vs. human:* 20 responses were scored by hand. The first judge prompt agreed within 1 point on 80% of cases but was lenient (for example 5/5 where the human gave 3/5). After adding explicit score definitions, agreement within 1 point reached 90%. Agreement within 0.5 points fell from 45% to 30%, and one outlier remained (human 4, judge 1).
   - *Learning loop:* a synthetic environment with a hidden best template per category and noisy rewards. The agent found the right template for 5 of 5 categories and average reward rose from 0.531 to 0.757.

## Dataset

80 hand-written queries (16 per category), split into 60 training and 20 held-out test queries (12 and 4 per category). The classifier labels all 80 queries correctly, but its rules were adjusted on this same dataset, so this does not show how it performs on unseen phrasing.

## Results

Average reward on the 20 held-out test queries, compared with two baselines:

- **Random** - pick a template at random
- **Fixed** - always use `step_by_step`
- **Learned** - the trained agent's policy

| Strategy | Avg. Reward |
|---|---|
| Random | 0.662 |
| Fixed | 0.638 |
| **Learned** | **0.692** |

The learned policy is about 4.5% above random and 8.5% above the fixed baseline.

| Category | Random | Fixed | Learned | Learned template |
|---|---|---|---|---|
| Factual | 0.792 | 0.813 | **0.917** | detailed |
| Conceptual | 0.750 | 0.667 | **0.771** | example-based |
| How-to | 0.688 | 0.583 | 0.688 | step-by-step |
| Comparison | 0.583 | 0.625 | 0.583 | step-by-step |
| Troubleshooting | 0.500 | 0.500 | 0.500 | step-by-step |

Plots (created by `analyze_results.py`): `strategy_comparison.png`, `category_breakdown.png`.

### How to read these results

- The clearest gain is on **factual** queries: 0.917 vs. 0.813 for the fixed baseline (about 13% higher) and vs. 0.792 for random (about 16% higher).
- For how-to, comparison and troubleshooting the learned policy chose `step_by_step`, which is also the fixed baseline. Learned and fixed therefore used the same template there, and any difference between them (for example 0.688 vs. 0.583 on how-to) is sampling and judge noise, not a policy effect.
- Troubleshooting scored exactly 0.500 for all three strategies. This may mean template choice matters little there, or that the reward is not sensitive enough; the raw judge outputs for that category have not been inspected.
- `step_by_step` won in 3 of 5 categories. This may reflect a real advantage of structured answers, or the known tendency of LLM judges to favour longer, more structured responses (Zheng et al., 2023).

## Limitations

- Only 20 test queries (4 per category) from a single run, with no confidence intervals or significance test. The overall gains (0.03-0.05) are about the same size as the run-to-run noise visible on categories where learned and fixed use the same template.
- Because each query is a one-shot decision, the agent is a contextual bandit over 5 states. Running all 5 templates on all training queries and picking the best per category would also work; an exhaustive "oracle" baseline is not included.
- The same small model (Qwen2.5-1.5B-Instruct) generates and judges responses, so self-preference and verbosity bias are possible. Judge validation used 20 samples and one human rater.
- The classifier is keyword-based and was tuned on this dataset.
- Generation samples at temperature 0.7 and only the Python `random` module is seeded, so a re-run will give slightly different numbers.

## How to run

Requires Python 3.9+ and, for training and evaluation, a GPU (for example Colab or Kaggle). The scripts read and write files in the current directory, so keep all files in one folder.

```
pip install -r requirements.txt

python verify_dataset.py        # checks the classifier on all 80 queries (no GPU)
python split_dataset.py         # creates train_queries.json / test_queries.json (no GPU)
python simulate_training.py     # sanity-checks the Q-learning loop (no GPU)
python test_pipeline.py         # end-to-end smoke test (downloads the model)

python generate_and_score.py    # writes judge_validation.csv; fill in your manual scores,
python check_agreement.py       # then measure judge-vs-human agreement

python train_agent.py           # trains the agent -> trained_q_table.npy, training_log.csv, epoch_summary.csv
python evaluate.py              # evaluates on test queries -> evaluation_results.csv
python analyze_results.py       # prints result tables and saves the two plots
```

`queries.json`, `train_queries.json` and `test_queries.json` are already included, so you can start at `train_agent.py`. The first run downloads Qwen2.5-1.5B-Instruct from Hugging Face.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── templates.py            # 5 prompt templates
├── classifier.py           # rule-based query classifier
├── llm_utils.py            # model loading and generation
├── reward_model.py         # judge prompt, JSON parsing, reward conversion
├── q_learning_agent.py     # Q-learning agent
├── build_dataset.py        # generates queries.json
├── verify_dataset.py       # classifier check on the dataset
├── split_dataset.py        # train/test split
├── simulate_training.py    # synthetic validation of the learning loop
├── test_pipeline.py        # smoke test
├── generate_and_score.py   # creates judge_validation.csv
├── check_agreement.py      # judge-vs-human agreement
├── train_agent.py          # RL training
├── evaluate.py             # baselines vs. learned policy
├── analyze_results.py      # tables and plots
├── queries.json  train_queries.json  test_queries.json
```

## Related work

- Deng et al. (2022), RLPrompt - RL for discrete prompt generation.
- Zhang et al. (2023), TEMPERA - test-time prompt editing with RL.
- Sun, Hüyük and van der Schaar (2024), Prompt-OIRL - query-dependent prompt evaluation with offline inverse RL (ICLR 2024).
- Hu, Leung and Farnia (2024), PAK-UCB - contextual-bandit prompt-aware model selection.
- Zheng et al. (2023), MT-Bench and Chatbot Arena - biases of LLM judges.

This project is a much smaller online, tabular setup with an explicitly validated judge.

## Future work

- Larger dataset, multiple seeds, confidence intervals and significance tests.
- An exhaustive oracle baseline and a comparison with contextual bandits (UCB, Thompson Sampling, LinUCB).
- An embedding-based classifier in place of keyword rules.
- A second, stronger or independent judge model, and more human raters.
- Testing whether the learned policy transfers to another LLM (for example Llama-3.2).
