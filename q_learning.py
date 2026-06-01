import numpy as np
import random
from maze import Maze
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent for solving maze navigation problems.
    Uses epsilon-greedy exploration strategy.
    """

    def __init__(
        self,
        maze,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,
        decay_rate=0.995,
    ):
        """
        Initialize Q-Learning agent.

        Args:
            maze: Maze instance
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy
            decay_rate: Rate at which epsilon decays
        """
        self.maze = maze
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.decay_rate = decay_rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.episode_rewards = []
        self.episode_steps = []

    def get_state_key(self, position):
        """
        Convert position to state key for Q-table.

        Args:
            position: Tuple (row, col)

        Returns:
            State key (tuple)
        """
        return position

    def get_action_space(self, state):
        """
        Get all possible actions from a state.

        Args:
            state: Current state

        Returns:
            List of possible next positions (actions)
        """
        return self.maze.get_neighbors(state)

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy strategy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action (next position)
        """
        possible_actions = self.get_action_space(state)

        if not possible_actions:
            return None

        if training and random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(possible_actions)
        else:
            # Exploit: choose action with highest Q-value
            q_values = [
                self.q_table[state][action] for action in possible_actions
            ]
            max_q = max(q_values)
            # Break ties randomly
            best_actions = [
                action
                for action, q in zip(possible_actions, q_values)
                if q == max_q
            ]
            return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-Learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Get current Q-value
        current_q = self.q_table[state][action]

        # Calculate maximum Q-value for next state
        next_actions = self.get_action_space(next_state)
        if next_actions:
            max_next_q = max(
                [self.q_table[next_state][a] for a in next_actions]
            )
        else:
            max_next_q = 0

        # Q-Learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q

    def get_reward(self, state, next_state):
        """
        Calculate reward for transitioning to next state.

        Args:
            state: Current state
            next_state: Next state

        Returns:
            Reward value
        """
        # Large reward for reaching goal
        if next_state == self.maze.goal:
            return 100

        # Small penalty for each step (encourages shorter paths)
        return -1

    def train_episode(self):
        """
        Run one training episode.

        Returns:
            Total reward and number of steps in episode
        """
        state = self.maze.start
        total_reward = 0
        steps = 0
        max_steps = self.maze.width * self.maze.height * 2

        while state != self.maze.goal and steps < max_steps:
            action = self.select_action(state, training=True)

            if action is None:
                break

            next_state = action
            reward = self.get_reward(state, next_state)
            self.update_q_value(state, action, reward, next_state)

            total_reward += reward
            state = next_state
            steps += 1

        # Decay epsilon
        self.epsilon = self.epsilon_start * (self.decay_rate ** len(self.episode_rewards))

        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)

        return total_reward, steps

    def train(self, num_episodes=500):
        """
        Train the agent for multiple episodes.

        Args:
            num_episodes: Number of episodes to train
        """
        print(f"Training Q-Learning agent for {num_episodes} episodes...")
        for episode in range(num_episodes):
            reward, steps = self.train_episode()
            if (episode + 1) % max(1, num_episodes // 10) == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                print(
                    f"Episode {episode + 1}/{num_episodes} - "
                    f"Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}"
                )

    def test_episode(self):
        """
        Run one test episode using greedy policy.

        Returns:
            Path taken and number of steps
        """
        state = self.maze.start
        path = [state]
        steps = 0
        max_steps = self.maze.width * self.maze.height * 2

        while state != self.maze.goal and steps < max_steps:
            action = self.select_action(state, training=False)

            if action is None:
                break

            state = action
            path.append(state)
            steps += 1

        return path, steps

    def get_policy_path(self):
        """
        Get the path following the learned policy from start to goal.

        Returns:
            Path as list of positions
        """
        path, _ = self.test_episode()
        return path
