import numpy as np
import random
import pandas as pd
from IPython.display import display, HTML


class StochasticEnvModel:

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.transitions = {(s, a): [] for s in range(n_states) for a in range(n_actions)}
        self.rewards = {(s, a): [] for s in range(n_states) for a in range(n_actions)}
        self.visited = set()

    def update(self, state, action, next_state, reward):
        """ Update the env model with observed (s, a, s', r) tuples. """
        self.transitions[(state, action)].append(next_state)
        self.rewards[(state, action)].append(reward)
        self.visited.add((state, action))

    def sample(self):
        """ Sample a random (state, action) pair from observed. """
        return random.choice(list(self.visited))

    def step(self, state, action):
        """ Return a (next_state, reward) pair based on observed probabilities. """
        if (state, action) not in self.visited:
            return None, None  # If there's no such (s, a)

        next_states = self.transitions[(state, action)]
        rewards = self.rewards[(state, action)]

        # Compute transition probabilities
        unique_next_states, counts = np.unique(next_states, return_counts=True)
        probs = counts / len(next_states)

        # Choose the next state based on probs
        next_state = np.random.choice(unique_next_states, p=probs)

        # Filter the corresponding rewards and calculate mean
        mask = np.array(next_states) == next_state
        avg_reward = np.mean(np.array(rewards)[mask])

        return next_state, avg_reward

    def display_transitions(self):
        """ Display transition probs for each action at each state. """
        data = np.zeros((self.n_states, self.n_actions), dtype=object)
        for (s, a), next_states in self.transitions.items():
            if next_states:
                unique, counts = np.unique(next_states, return_counts=True)
                probs = {s_: round(c / sum(counts), 2) for s_, c in zip(unique, counts)}
                data[s, a] = str(probs)
        df = pd.DataFrame(data, columns=['L', 'D', 'R', 'U'])
        display(HTML(df.T.to_html()))

    def display_rewards(self):
        """ Display expected rewards for each action at each state. """
        data = np.zeros((self.n_states, self.n_actions))
        for (s, a), rewards in self.rewards.items():
            if rewards:
                data[s, a] = np.mean(rewards)
        df = pd.DataFrame(data, columns=['L', 'D', 'R', 'U'])
        display(HTML(df.T.to_html()))
