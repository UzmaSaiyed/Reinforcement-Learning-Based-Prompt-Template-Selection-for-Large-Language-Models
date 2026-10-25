"""
q_learning_agent.py
This is our "learning brain" - it keeps a notebook (Q-table) of
"when I see this type of question, this template earned this much reward."

A quick honest note on the math: normally Q-learning also factors in
"what happens next" (future rewards), using a discount factor gamma.
In OUR project, each question is independent - answering one question
doesn't lead to a "next state" the way moves in a game do. So this is
technically closer to what's called a "contextual bandit" - we simplify
the update rule accordingly (no future-reward term). This is a normal,
correct simplification for this kind of one-shot decision problem.
"""

import numpy as np
import random


class QLearningAgent:
    def __init__(self, num_states: int, num_actions: int, learning_rate: float = 0.1):
        """
        num_states: how many query categories we have (5 in our case)
        num_actions: how many templates we have (5 in our case)
        learning_rate (alpha): how much each new experience updates our beliefs
                                 (0 = never learn, 1 = fully trust the newest result)
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        # The Q-table: rows = states (query types), columns = actions (templates)
        # Starts at all zeros - the agent knows nothing yet.
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state: int, epsilon: float) -> int:
        """
        Decide which template to use for this query type.

        epsilon = chance of picking a RANDOM template instead of the
        best-known one (this is called "exploration" - trying new things
        instead of always doing what seemed best so far).

        Example: epsilon=0.3 means 30% of the time, try something random;
        70% of the time, use the best template we've found so far.
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)  # explore
        else:
            return int(np.argmax(self.q_table[state]))       # exploit (best known)

    def update(self, state: int, action: int, reward: float):
        """
        This is the core learning step. After we see how good a template
        actually worked for a query type, we nudge our belief closer to
        that result.

        Formula: Q(state, action) = Q(state, action) + alpha * (reward - Q(state, action))

        In plain words: "move my current belief a little bit toward
        what I just observed."
        """
        current_estimate = self.q_table[state, action]
        self.q_table[state, action] = current_estimate + self.alpha * (reward - current_estimate)

    def best_action_for_state(self, state: int) -> int:
        """Returns the currently-best-known template for a given query type."""
        return int(np.argmax(self.q_table[state]))

    def get_q_table_readable(self, state_names: dict, action_names: dict) -> str:
        """Pretty-prints the Q-table with real names instead of numbers."""
        lines = []
        for s in range(self.num_states):
            best_a = self.best_action_for_state(s)
            lines.append(f"{state_names[s]:20s} -> best template: {action_names[best_a]}")
            for a in range(self.num_actions):
                marker = " <-- BEST" if a == best_a else ""
                lines.append(f"    {action_names[a]:20s}: {self.q_table[s, a]:.3f}{marker}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Basic sanity test: does a single update move the Q-value in the right direction?
    agent = QLearningAgent(num_states=5, num_actions=5, learning_rate=0.1)

    print("Before any updates, all Q-values should be 0:")
    print(agent.q_table[0])
    assert np.all(agent.q_table == 0), "FAILED: Q-table should start at all zeros"
    print("PASSED\n")

    print("After ONE update with reward=1.0 on state=0, action=2:")
    agent.update(state=0, action=2, reward=1.0)
    print(agent.q_table[0])
    # With alpha=0.1: new value = 0 + 0.1*(1.0 - 0) = 0.1
    expected = 0.1
    actual = agent.q_table[0, 2]
    assert abs(actual - expected) < 1e-6, f"FAILED: expected {expected}, got {actual}"
    print(f"PASSED: Q-value moved to {actual:.3f} as expected (0.1)\n")

    print("After MANY updates with reward=1.0 on the same state/action,")
    print("the Q-value should get closer and closer to 1.0:")
    for _ in range(50):
        agent.update(state=0, action=2, reward=1.0)
    print(f"Q-value after 50 more updates: {agent.q_table[0, 2]:.4f} (should be very close to 1.0)")
    assert agent.q_table[0, 2] > 0.99, "FAILED: Q-value should have converged near 1.0"
    print("PASSED\n")

    print("Testing choose_action with epsilon=0 (should ALWAYS pick the best known action):")
    picks = [agent.choose_action(state=0, epsilon=0.0) for _ in range(20)]
    print(f"20 picks: {picks}")
    assert all(p == 2 for p in picks), "FAILED: with epsilon=0, should always pick action 2"
    print("PASSED: always picked action 2, the one with highest Q-value\n")

    print("Testing choose_action with epsilon=1.0 (should be random every time):")
    picks = [agent.choose_action(state=0, epsilon=1.0) for _ in range(20)]
    print(f"20 picks: {picks}")
    unique_picks = set(picks)
    assert len(unique_picks) > 1, "FAILED: with epsilon=1.0, should see variety, not always the same action"
    print(f"PASSED: saw {len(unique_picks)} different actions picked, confirms randomness works\n")

    print("ALL TESTS PASSED - the Q-learning agent's core logic is correct.")
