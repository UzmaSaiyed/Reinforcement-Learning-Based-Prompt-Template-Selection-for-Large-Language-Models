"""
simulate_training.py
This is a critical validation step: BEFORE we trust the training loop
with the real, slow, expensive LLM, we test it on a FAKE environment
where we already secretly know the right answer.

We set up: "for category X, template Y secretly gives higher reward
(with some random noise, like a real LLM would have)."

Then we run full training and check: did the agent actually FIGURE
THIS OUT on its own, purely through trial and error?

If this doesn't work here, it definitely won't work with the real LLM -
so this is our safety check before spending real time/compute.
"""

import numpy as np
import random
from q_learning_agent import QLearningAgent

random.seed(0)
np.random.seed(0)

NUM_STATES = 5
NUM_ACTIONS = 5

# THE SECRET GROUND TRUTH (only we know this - the agent must discover it):
# For each state (query category), one specific template secretly performs
# better on average than the others.
SECRET_BEST_ACTION = {
    0: 3,  # for category 0, template 3 is secretly best
    1: 0,  # for category 1, template 0 is secretly best
    2: 4,  # for category 2, template 4 is secretly best
    3: 1,  # for category 3, template 1 is secretly best
    4: 2,  # for category 4, template 2 is secretly best
}


def fake_reward(state: int, action: int) -> float:
    """
    Simulates what a real LLM+judge reward might look like:
    - the "secretly best" action gives a HIGHER average reward
    - all actions have random noise added (real rewards are never
      perfectly consistent, just like our real judge won't be)
    """
    if action == SECRET_BEST_ACTION[state]:
        base = 0.8  # secretly-best action: high average reward
    else:
        base = 0.4  # other actions: lower average reward
    noise = np.random.normal(0, 0.15)  # random noise, like real-world messiness
    return float(np.clip(base + noise, 0.0, 1.0))


def train(num_epochs: int = 30, queries_per_epoch: int = 50, initial_epsilon: float = 0.4):
    agent = QLearningAgent(num_states=NUM_STATES, num_actions=NUM_ACTIONS, learning_rate=0.1)
    reward_history = []

    epsilon = initial_epsilon
    for epoch in range(num_epochs):
        epoch_rewards = []
        for _ in range(queries_per_epoch):
            state = random.randint(0, NUM_STATES - 1)          # a random query category
            action = agent.choose_action(state, epsilon)         # agent picks a template
            reward = fake_reward(state, action)                  # simulate the outcome
            agent.update(state, action, reward)                  # agent learns from it
            epoch_rewards.append(reward)

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        reward_history.append(avg_reward)
        epsilon = max(0.05, epsilon * 0.95)  # gradually explore less over time

    return agent, reward_history


if __name__ == "__main__":
    print("Training on FAKE (synthetic) data where we know the correct answer...\n")
    agent, reward_history = train()

    print("Reward per epoch (should trend UPWARD if learning is happening):")
    # Print every 5th epoch so it's not too long
    for i in range(0, len(reward_history), 5):
        print(f"  Epoch {i+1:3d}: avg reward = {reward_history[i]:.3f}")
    print(f"  Epoch {len(reward_history):3d}: avg reward = {reward_history[-1]:.3f} (final)")

    first_5_avg = sum(reward_history[:5]) / 5
    last_5_avg = sum(reward_history[-5:]) / 5
    print(f"\nFirst 5 epochs average: {first_5_avg:.3f}")
    print(f"Last 5 epochs average:  {last_5_avg:.3f}")
    assert last_5_avg > first_5_avg, "FAILED: reward should improve over training, but it didn't!"
    print("PASSED: reward clearly improved over training.\n")

    print("=" * 60)
    print("Did the agent correctly discover the SECRET best template for each state?\n")
    all_correct = True
    for state in range(NUM_STATES):
        learned_best = agent.best_action_for_state(state)
        actual_best = SECRET_BEST_ACTION[state]
        status = "CORRECT" if learned_best == actual_best else "WRONG"
        if learned_best != actual_best:
            all_correct = False
        print(f"  State {state}: agent learned template {learned_best}, "
              f"secretly-best was template {actual_best}  -> {status}")

    print()
    if all_correct:
        print("SUCCESS: The agent correctly discovered the best template for EVERY")
        print("state, purely from trial and error. The training loop works correctly.")
    else:
        print("WARNING: The agent didn't fully converge. May need more epochs or")
        print("a different learning rate. (This is still useful info to know!)")
