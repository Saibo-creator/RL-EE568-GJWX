import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, env, gamma=0.99, lr=1e-3, batch_size=64, epsilon=1.0, min_epsilon=0.01, decay=0.995):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        q_values = self.q_net(states).gather(1, actions).squeeze()

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def train(self, episodes=500, target_update=10):
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0

            for t in range(500):  # max steps per episode
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.push((state, action, reward, next_state, float(done)))
                self.train_step()

                state = next_state
                total_reward += reward
                if done:
                    break

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode % target_update == 0:
                self.update_target()

            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")

# Main
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)
    agent.train()
